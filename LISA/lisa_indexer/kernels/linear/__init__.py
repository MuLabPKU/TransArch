"""Head-fused FP8x2 scoring with a HOT key permutation.

Keys are gathered in HOT order on the caller stream. select_permuted accepts
keys already in that order; policy_rule selects a kernel by head count and
sequence length.
"""
from functools import lru_cache
import json
from pathlib import Path

import torch
import triton

from ...build import extension
from ...api import PreparedFP8
from .. import linear_status
from .score import pack

HOT = 12288
CAPACITY = 196608


@lru_cache(maxsize=2)
def _policy_rules(heads):
    name = 'policy_h32.json' if heads == 32 else 'policy_h64.json'
    return tuple(json.loads(Path(__file__).with_name(name).read_text())['rules'])


def policy_rule(heads, n):
    if heads not in (32, 64) or not 0 < n <= 1048576:
        raise ValueError("linear policy supports H32/H64 and N<=1M")
    # Cache the policies to avoid reading them for every layer.
    return next(rule for rule in _policy_rules(heads) if n <= rule['max_n']).copy()


class LinearRuntime:
    def __init__(self, max_n):
        self.max_n = max_n
        self.ws = None
        self.static_ws = {}
        self.planner = None
        self.carries = {}
        self.last_stats = {}

    @staticmethod
    def scorer(heads):
        from . import score
        return score

    def workspace(self, prepared):
        rows, device = len(prepared.q), prepared.q.device
        if self.ws is not None and self.ws['q2'].shape[0] == rows:
            return self.ws
        self.static_ws.clear()
        def alloc(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)
        self.ws = dict(
            q2=alloc((rows, 2, 128), prepared.q.dtype), w4=alloc((rows, 4), torch.float32),
            origin=alloc(rows, torch.float32), inv=alloc(rows, torch.float32),
            threshold=alloc(rows, torch.int32), histogram=alloc((rows, 256), torch.int32),
            values=alloc((rows, CAPACITY), torch.float16), ids=alloc((rows, CAPACITY), torch.int32),
            counts=alloc(rows, torch.int32), status=alloc(rows, torch.int32),
            zeros=torch.zeros(rows, dtype=torch.int32, device=device),
            prefix_ends=torch.full((rows,), HOT, dtype=torch.int32, device=device),
            run_max=torch.zeros(1, dtype=torch.int32, device=device),
            over_events=torch.zeros(1, dtype=torch.int32, device=device),
        )
        return self.ws

    def _planner(self, device):
        if self.planner is None:
            self.planner = dict(
                permutation=torch.arange(self.max_n, dtype=torch.int32, device=device),
                hot_epoch=torch.zeros(self.max_n, dtype=torch.int32, device=device),
                swap_a=torch.empty(HOT, dtype=torch.int32, device=device),
                swap_b=torch.empty(HOT, dtype=torch.int32, device=device),
                counts=torch.zeros(4, dtype=torch.int32, device=device), epoch=0,
            )
        return self.planner

    def _vote_workspace(self, prepared):
        n, rows, dev = len(prepared.keys), len(prepared.q), prepared.q.device
        recent = min(rows, 1536)
        max_vote = min(recent, recent * 2048 // HOT + 1)
        if not hasattr(self, 'votes'):
            self.votes = torch.zeros(self.max_n, dtype=torch.int32, device=dev)
            self.partial = torch.empty((128, 1537), dtype=torch.int16, device=dev)
            self.vote_state = torch.zeros(136, dtype=torch.int32, device=dev)
        return self.votes[:n], recent, max_vote

    def publish_carry(self, prepared, layer_id, out, *, permutation=None, status=None):
        """Select the most frequent recent winners for the next HOT permutation."""
        n, rows, dev = len(prepared.keys), len(prepared.q), prepared.q.device
        if n < HOT:
            return
        native = extension('linear_native')
        votes, recent, max_vote = self._vote_workspace(prepared)
        if permutation is None:
            recent_ids = out[-recent:].to(torch.int64).reshape(-1)
            # Short causal rows contain -1; they contribute no votes.
            valid = recent_ids >= 0
            votes.scatter_add_(0, recent_ids.clamp_min(0), valid.to(torch.int32))
        else:
            ws = self.ws
            native.map_topk_vote_stats_litetopk_(out, permutation, status, votes,
                recent, ws['counts'], ws['run_max'], ws['over_events'], CAPACITY)
        hot = self.carries.get(layer_id, (None, 0))[0]
        if hot is None:
            hot = torch.empty(HOT, dtype=torch.int64, device=dev)
        # The extension requires contiguous [128,max_vote+1], not a strided view.
        partial = self.partial.view(-1)[:128 * (max_vote + 1)].view(128, max_vote + 1)
        native.carry_votes_topk_reset_(votes, hot, partial, self.vote_state, HOT, max_vote, 0)
        self.carries[layer_id] = (hot, n)

    def select_permuted(self, prepared, ends, out, *, permutation, layer_id=0, publish=True):
        """Select from keys already ordered by the HOT permutation.

        Every row must see [0,12288). permutation maps physical rows to logical
        request indices and must be a bijection. The HOT prefix must lie within
        the common causal prefix.
        """
        n, rows = len(prepared.keys), len(prepared.q)
        native = extension('linear_native')
        score = self.scorer(prepared.q.shape[1])
        ws = self.workspace(prepared)
        g = score.collapse(prepared, components=2)
        pack[(triton.cdiv(rows * 128, 256),)](g.high, g.low, g.high_scale, g.low_scale,
            ws['q2'], ws['w4'], rows, B=256)
        prefix = PreparedFP8(prepared.q, prepared.keys[:HOT], prepared.key_scales[:HOT],
                             prepared.effective_weights)
        sample = score.scan_prepared(g, prefix, ws['zeros'], ws['prefix_ends'], config=score.ScanConfig())
        native.seed_prep_litetopk_(sample, 256, 2048, CAPACITY, HOT, 0.0, 0, 1,
            ws['origin'], ws['inv'], ws['threshold'], ws['histogram'], ws['values'], ws['ids'], ws['counts'])
        scales = prepared.key_scales
        if n % 4:
            scales = torch.nn.functional.pad(scales, (0, 4 - n % 4))
        rule = policy_rule(prepared.q.shape[1], n)
        variant, config = rule['variant'], rule['config']
        args = (ws['q2'], prepared.keys, scales, ws['w4'], ws['prefix_ends'], ends,
                ws['origin'], ws['inv'], ws['threshold'], ws['values'], ws['ids'], ws['counts'], ws['histogram'])
        if variant == 'static':
            from .sampled_scan import scan
            scan(*args, config, workspace=self.static_ws)
        else:
            scan_ext = extension('linear_' + variant)
            scan_ext.scan(*args, (30 if n <= 131072 else 31 if n <= 262144 else 29)
                          if variant == 'recovery' else config)
        final_args = (ws['values'], ws['ids'], ws['counts'], out, ws['status'], ws['histogram'], n)
        if variant == 'gatefinal':
            extension('linear_gatefinal').finalize_(*final_args, ws['threshold'])
        else:
            extension('linear_final').finalize_(*final_args)
        linear_status[(triton.cdiv(rows, 256),)](ws['status'], ws['counts'], rows, CAPACITY, 256, debug=True)
        if publish:
            self.publish_carry(prepared, layer_id, out, permutation=permutation, status=ws['status'])
        else:
            native.map_topk_indices_and_accumulate_votes_litetopk_(out, permutation,
                ws['status'], ws['counts'][:0], 1)
        self.last_stats = dict(route='hot', variant=variant, config=config,
                               status=ws['status'], counts=ws['counts'])
        return out

    def select(self, prepared, starts, ends, out, *, layer_id, query_start):
        n, rows = len(prepared.keys), len(prepared.q)
        native = extension('linear_native')
        if n <= 65536 or query_start + 1 < HOT:
            # Bound logits memory even for a cold request with a long K cache.
            score = self.scorer(prepared.q.shape[1])
            tile = max(1, min(rows, (256 << 20) // (4 * n)))
            for start in range(0, rows, tile):
                stop = min(rows, start + tile)
                p = PreparedFP8(prepared.q[start:stop], prepared.keys, prepared.key_scales,
                                 prepared.effective_weights[start:stop])
                logits = score.stage1(p, starts[start:stop], ends[start:stop], components=2)
                native.dense_topk_litetopk_(logits, starts[start:stop], ends[start:stop], out[start:stop],
                    stop - start, logits.stride(0), logits.stride(1), 2048, 0, 0)
            self.publish_carry(prepared, layer_id, out)
            self.last_stats = dict(route='dense_fp8x2', row_tile=tile)
            return out
        plan = self._planner(prepared.q.device)
        hot, extent = self.carries.get(layer_id, (None, 0))
        if hot is None or extent > query_start + 1:
            hot = torch.arange(HOT, device=prepared.q.device, dtype=torch.int64)
        plan['epoch'] += 1
        native.plan_hot_permutation_out(hot, plan['hot_epoch'], plan['permutation'],
            plan['swap_a'], plan['swap_b'], plan['counts'], query_start + 1, plan['epoch'])
        permutation = plan['permutation'][:n]
        # Gather keys and scales in the current HOT order.
        keys = torch.index_select(prepared.keys.view(torch.uint8), 0, permutation).view(prepared.q.dtype)
        scales = torch.index_select(prepared.key_scales, 0, permutation)
        p = PreparedFP8(prepared.q, keys, scales, prepared.effective_weights)
        return self.select_permuted(p, ends, out, permutation=permutation, layer_id=layer_id)
