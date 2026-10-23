"""Full September HOT prefix/carry around the received fused1+residual7 scorer.

Only disjoint KV pairs inside the common causal prefix are swapped. The same
pairs restore gathered K/scales before return, and final IDs map to the caller's
original corpus order before carry publication. No decode path is installed.
"""
import threading

import torch

from . import carry, native
from .extensions import extension

_PLANS = {}
_MAP = {}


def _stream_key(device):
    return (str(device), torch.cuda.current_stream(device).cuda_stream, threading.get_ident())


def plan_workspace(n, device):
    key = _stream_key(device)
    state = _PLANS.get(key)
    if state is None or state["capacity"] < n or state["epoch"] >= (1 << 31) - 1:
        capacity = max(16384, 1 << (n - 1).bit_length())
        state = dict(capacity=capacity, epoch=0,
            marks=torch.zeros(capacity, dtype=torch.int32, device=device),
            permutation=torch.arange(capacity, dtype=torch.int32, device=device),
            a=torch.empty(12288, dtype=torch.int32, device=device),
            b=torch.empty(12288, dtype=torch.int32, device=device),
            counts=torch.zeros(4, dtype=torch.int32, device=device))
        _PLANS[key] = state
    state["epoch"] += 1
    return state


def _map_workspace(m, n, device):
    key = _stream_key(device)
    state = _MAP.get(key)
    if state is None or state["n"] < n or state["m"] < m:
        nc = max(16384, 1 << (n - 1).bit_length())
        mc = max(128, 1 << (m - 1).bit_length())
        state = dict(n=nc, m=mc,
            identity=torch.arange(nc, dtype=torch.int32, device=device),
            status=torch.zeros(mc, dtype=torch.int32, device=device))
        _MAP[key] = state
    return state["identity"][:n], state["status"][:m]


def select_prepared(qf, qscale, qs, weights, key, kscale, starts, ends, *, layer):
    m, n = qf.shape[0], key.shape[0]
    common_end = n - m + 1
    if not (0 < m <= 8192 and 0 < n <= 135168 and common_end > 0):
        raise ValueError("HOT expects a trailing query chunk in context<=135168")
    if not key.is_contiguous() or not kscale.is_contiguous() or kscale.dtype != torch.float32:
        raise ValueError("HOT requires contiguous K and FP32 scales")
    key_name = ("resmisa_hot", layer)
    cached = carry._HOT_CARRY.get((str(key.device), key_name))
    # Request/layer rollback is handled explicitly before preparation. A stale
    # carry reaching this point is an integration error, not a cold fallback.
    if cached is not None and cached[1] > n:
        raise RuntimeError("Stale HOT carry: request lifecycle was not established")
    hot_valid = (cached is not None and cached[1] <= common_end and cached[3] == 0
                 and cached[0].numel() >= carry.HOT_PREFIX and carry.HOT_PREFIX <= common_end)
    plan = None
    event = None
    if n <= 12288 or common_end < 8192:
        scores = native._scores(qf, qscale, qs, weights, key, kscale, starts, ends)
        out = native.dense_select(scores, starts, ends, 2048, initial=True)
    else:
        if hot_valid:
            event = cached[2]
            if event is not None:
                torch.cuda.current_stream(key.device).wait_event(event)
            hot = cached[0][:carry.HOT_PREFIX]
            hot.record_stream(torch.cuda.current_stream(key.device))
            plan = plan_workspace(n, key.device)
            extension("pair_swap").plan_and_swap_(hot, plan["marks"], plan["permutation"],
                plan["a"], plan["b"], plan["counts"], 0, common_end, plan["epoch"],
                key.view(torch.uint8), kscale)
        try:
            out = native.select(qf, qscale, qs, weights, key, kscale, starts, ends, 2048,
                hot_valid=hot_valid)
        finally:
            if plan is not None:
                extension("pair_swap").swap_gathered_(key.view(torch.uint8), kscale,
                    plan["a"], plan["b"], plan["counts"])

    # Underfilled early causal rows may return -1; they do not publish a carry.
    publish = n >= carry.HOT_PREFIX and common_end >= 2048
    if plan is not None or publish:
        identity, status = _map_workspace(m, n, key.device)
        permutation = plan["permutation"][:n] if plan is not None else identity
        recent = min(m, carry.CARRY_RECENT_ROWS)
        votes = carry._carry_vote_hist(n, key.device, key_name, event) if publish else status[:0]
        # This fresh status belongs to completed/recovered output. Passing a
        # stale native failure flag would suppress correctly recovered rows.
        extension("carry").map_topk_indices_and_accumulate_votes_litetopk_(
            out, permutation, status, votes, recent)
        if publish:
            vote_cap = min(recent, recent * 2048 // carry.HOT_PREFIX + 1)
            carry._publish_carry(key_name, votes, n, 0, vote_cap)
    return out
