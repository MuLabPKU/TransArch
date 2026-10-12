"""Share H64 threshold candidate pools across groups of four layers."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from types import SimpleNamespace
from . import native, recovery, score as p  # native installs the scan adapter
from ...api import validate_prepared, validate_ranges
from ...api import ForwardLedger
from ..linear.score import _collapse_kernel

def _validate_h64(prepared):
    validate_prepared(prepared)
    if prepared.q.shape[1] != 64:
        raise ValueError("shared4 requires 64 query heads")

device = SimpleNamespace(Workspace=recovery.Workspace,
    DeviceSelection=recovery.DeviceSelection, _validate_prepared=_validate_h64,
    _validate_ranges=validate_ranges, _collapse_kernel=_collapse_kernel,
    _initial_keys=recovery._initial_keys, _pack_status=recovery._pack_status)

@triton.jit
def _guard(Starts, Ends, SavedStarts, SavedEnds, Counts, Status, rows,
           MIN: tl.constexpr, CAP: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0) * B + tl.arange(0, B)
    valid = r < rows
    start = tl.load(Starts + r, valid, 0)
    end = tl.load(Ends + r, valid, 0)
    saved_start = tl.load(SavedStarts + r, valid, 0)
    saved_end = tl.load(SavedEnds + r, valid, 0)
    count = tl.load(Counts + r, valid, 0)
    status = tl.load(Status + r, valid, 0)
    tl.device_assert((~valid) | ((start == saved_start) & (end == saved_end)),
                     'shared threshold pool causal ranges changed')
    tl.device_assert((~valid) | ((status == 0) & (count >= tl.minimum(end - start, MIN)) &
                     (count <= tl.minimum(end - start, CAP))), 'invalid threshold pool')


def rerank_into(prepared, candidates, starts, ends, out, splits=4):
    rows, cap = candidates.ids.shape
    assert splits in (1, 2, 4, 8, 16)
    p._rerank_counted_kernel[(rows, splits)](
        prepared.q, prepared.keys, prepared.effective_weights, prepared.key_scales,
        candidates.ids, candidates.counts, candidates.status, starts, ends, out,
        len(prepared.keys), CAP=cap, BN=128, SPLITS=splits, num_warps=4, num_stages=3)
    return out


def finish_into(candidates, logits, starts, out, cached=False):
    assert out.shape == (len(starts), 2048) and out.dtype == torch.int32
    assert out.device == logits.device and out.is_contiguous()
    assert all(out.untyped_storage().data_ptr() != t.untyped_storage().data_ptr()
               for t in (candidates.ids, candidates.counts, candidates.status, logits, starts))
    p.load_finish()
    torch.ops.relu_hisa_counted_finish.run(
        logits, candidates.ids, candidates.counts, candidates.status, starts, out, cached)
    return out


_TUNED_SCAN = None


def scan(prepared, ws, config=None):
    global _TUNED_SCAN
    rows, n = len(prepared.q), len(prepared.keys)
    args = (ws.g.high, ws.g.high_scale, ws.g.low, ws.g.low_scale,
            prepared.keys, prepared.key_scales, ws.starts, ws.ends, ws.thresholds,
            ws.ids, ws.counts, ws.ids, rows, n)
    if config is None:
        return p._scan_compact_kernel[(triton.cdiv(n, 64), triton.cdiv(rows, 64))](
            *args, CAP=ws.budget.capacity, BQ=64, BN=64, ACC=32, DEBUG=False,
            RESERVE=128, ThresholdKeys=None, KEY_THRESHOLD=False, Q_FIRST=False,
            num_warps=4, enable_fp_fusion=False)
    if _TUNED_SCAN is None:
        from .scan import compact_scan_tuned as compact_scan
        _TUNED_SCAN = compact_scan
    group = config['group_q']
    return _TUNED_SCAN[(triton.cdiv(n, 64) * group, triton.cdiv(rows, 64 * group))](
        *args, CAP=ws.budget.capacity, BQ=64, BN=64, GROUP_Q=group, DEBUG=False,
        ThresholdKeys=None, KEY_THRESHOLD=False, num_warps=4,
        enable_fp_fusion=False, maxnreg=config['maxnreg'])


def coarse(prepared, starts, ends, ws, *, scan_config=None, thresholds=None):
    """Build threshold candidates and recover overflow rows using the supplied workspace."""
    device._validate_prepared(prepared)
    device._validate_ranges(prepared, starts, ends)
    rows, n, budget = len(prepared.q), len(prepared.keys), ws.budget
    assert ws.rows == rows and ws.max_n >= n and ws.device == prepared.q.device
    ws.starts.copy_(starts)
    ws.ends.copy_(ends)
    ws.counts.zero_()
    ws.status.zero_()
    device._collapse_kernel[(rows,)](
        prepared.q, prepared.effective_weights, ws.g.high, ws.g.high_scale,
        ws.g.low, ws.g.low_scale, COMPONENTS=2, num_warps=4)
    if thresholds is None:
        p._pilot_kernel[(triton.cdiv(budget.samples, 64), triton.cdiv(rows, 64))](
            ws.g.high, ws.g.high_scale, ws.g.low, ws.g.low_scale,
            prepared.keys, prepared.key_scales, ws.starts, ws.ends, ws.pilot,
            rows, n, S=budget.samples, BQ=64, BN=64, ACC=32,
            num_warps=4, enable_fp_fusion=False)
        p._threshold_kernel[(rows,)](
            ws.pilot, ws.starts, ws.ends, ws.thresholds, ws.status, n,
            S=budget.samples, TARGET=budget.target, num_warps=4)
    else:
        assert thresholds.shape == (rows,) and thresholds.dtype == torch.float32
        ws.thresholds.copy_(thresholds)
    if n:
        scan(prepared, ws, scan_config)
    device._initial_keys[(triton.cdiv(rows, 256),)](ws.thresholds, ws.threshold_keys, rows, B=256)
    device._pack_status[(1,)](
        ws.counts, ws.starts, ws.ends, ws.status, ws.rowmap, ws.summary, rows, n,
        MINIMUM=budget.minimum, CAP=budget.capacity, B=triton.next_power_of_2(rows),
        num_warps=8 if rows > 2048 else 4)
    ws.graph(prepared, n, False).launch()
    return p.Candidates(ws.ids, ws.counts, ws.status, ws.thresholds,
                        ws.pilot if thresholds is None else None, None, budget, ws.threshold_keys)


@dataclass
class Lease:
    generation: int
    n: int
    next_layer: int
    last_layer: int
    workspace: object
    candidates: object


class PoolEngine:
    def __init__(self, *, max_n=1048576, group_size=4, splits=4, scan_config=None,
                 policy=None, cached_finish=False, num_layers=61):
        assert group_size in (1, 2, 4)
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self.max_n = max_n
        self.group_size = group_size
        self.splits = splits
        self.scan_config = scan_config
        self.policy = policy
        self.cached_finish = cached_finish
        self.workspaces = {}
        self.logits = {}
        self.pools = {}
        self.stream = None
        self.generation = 0
        self.calls = self.coarse_calls = self.member_calls = 0
        self.ledger = ForwardLedger(num_layers)

    def begin_forward(self):
        if self.pools:
            raise RuntimeError('previous forward has unconsumed shared pools')
        self.ledger.begin()
        self.generation = self.ledger.generation

    def end_forward(self):
        if self.pools:
            raise RuntimeError('shared4 forward ended with unconsumed candidate pools')
        self.ledger.finish()

    def workspace_slot(self, phase, rows, device):
        # A framework can execute A and B at layer 1 before either slice
        # reaches layer 4. Each live lease needs distinct candidate storage.
        # Recycle slots only after leases end, independent of absolute position.
        live = {id(lease.workspace) for lease in self.pools.values()}
        slot = 0
        while True:
            key = (phase, rows, device, slot)
            if key not in self.workspaces or id(self.workspaces[key]) not in live:
                return key
            slot += 1

    def role(self, layer):
        if not 0 <= layer < self.num_layers:
            raise ValueError("layer out of configured range")
        donor = 0 if layer == 0 else 1 + (layer - 1) // self.group_size * self.group_size
        last = 0 if layer == 0 else min(donor + self.group_size - 1, self.num_layers - 1)
        return donor, last

    def select(self, prepared, starts, ends, out, *, layer, phase='prefill',
               chunk=(0, 8192), thresholds=None):
        device._validate_prepared(prepared)
        device._validate_ranges(prepared, starts, ends)
        rows, n = len(prepared.q), len(prepared.keys)
        if not rows or rows > 8192 or n > self.max_n:
            raise ValueError('unsupported shared pool shape')
        ledger_key = (phase, chunk)
        layout = (rows, n, prepared.q.device)
        self.ledger.check(layer, ledger_key, layout)
        current = (prepared.q.device, torch.cuda.current_stream(prepared.q.device).cuda_stream)
        if self.stream is None:
            self.stream = current
        if self.stream != current or not self.generation:
            raise RuntimeError('pool requires one stream and begin_forward')
        donor, last = self.role(layer)
        key = (donor, phase, chunk)
        rule = None if self.policy is None else next(r for r in self.policy if n <= r['max_n'])
        scan_config = self.scan_config if rule is None else rule['scan']
        splits = self.splits if rule is None else rule['splits']
        if layer == donor:
            wskey = self.workspace_slot(phase, rows, prepared.q.device)
            if key in self.pools:
                raise RuntimeError('duplicate donor')
            if wskey not in self.workspaces:
                self.workspaces[wskey] = device.Workspace(rows, prepared.q.device, p.Budget(), self.max_n)
            ws = self.workspaces[wskey]
            candidates = coarse(prepared, starts, ends, ws, scan_config=scan_config, thresholds=thresholds)
            if last != donor:
                self.pools[key] = Lease(self.generation, n, layer + 1, last, ws, candidates)
            self.coarse_calls += 1
        else:
            lease = self.pools.get(key)
            if lease is None or lease.generation != self.generation or lease.next_layer != layer:
                raise RuntimeError('missing, stale, or out-of-order shared pool')
            ws, candidates = lease.workspace, lease.candidates
            if ws.rows != rows or lease.n != n or ws.device != prepared.q.device:
                raise ValueError('member shape differs from donor')
            _guard[(triton.cdiv(rows, 256),)](
                starts, ends, ws.starts, ws.ends, candidates.counts, candidates.status,
                rows, MIN=p.Budget().minimum, CAP=p.Budget().capacity, B=256, debug=True)
            lease.next_layer += 1
            self.member_calls += 1
        scratch_key = (rows, prepared.q.device)
        if scratch_key not in self.logits:
            self.logits[scratch_key] = torch.empty_like(candidates.ids, dtype=torch.float32)
        logits = rerank_into(prepared, candidates, starts, ends, self.logits[scratch_key], splits)
        finish_into(candidates, logits, starts, out, self.cached_finish)
        if layer != donor and layer == last:
            del self.pools[key]
        self.calls += 1
        self.ledger.commit(layer, ledger_key, layout)
        return device.DeviceSelection(p.Selection(candidates, logits, out), ws.summary)
