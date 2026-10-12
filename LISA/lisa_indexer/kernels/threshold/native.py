"""H32/H64 threshold selection, overflow recovery, and candidate reranking."""
import torch
import triton
import triton.language as tl
from . import score as p
from . import recovery
from .score import PreparedFP8, _quantize_row, _validate_ranges
from .scan import ScanAdapter, install

if not isinstance(p._scan_compact_kernel, ScanAdapter):
    install(p)

@triton.jit
def _collapse_h32(Q, W, Hi, HS, Lo, LS):
    row = tl.program_id(0)
    h, d = tl.arange(0, 32), tl.arange(0, 128)
    q = tl.load(Q + row * 32 * 128 + h[:, None] * 128 + d[None, :]).to(tl.float32)
    w = tl.load(W + row * 32 + h).to(tl.float32)
    g = tl.sum(q * w[:, None], axis=0)
    hi, hs = _quantize_row(g)
    low, ls = _quantize_row(g - hi.to(tl.float32) * hs)
    tl.store(Hi + row * 128 + d, hi)
    tl.store(HS + row, hs)
    tl.store(Lo + row * 128 + d, low)
    tl.store(LS + row, ls)


@triton.jit
def _rerank_h32(Q, K, W, KS, IDs, Counts, Status, Starts, Ends, Out,
                 n, CAP: tl.constexpr, BN: tl.constexpr, SPLITS: tl.constexpr):
    row = tl.program_id(0)
    count = tl.minimum(tl.load(Counts + row), CAP)
    status = tl.load(Status + row)
    if status == 0:
        h, d = tl.arange(0, 32), tl.arange(0, 128)
        qt = tl.load(Q + row * 32 * 128 + h[None, :] * 128 + d[:, None])
        w = tl.load(W + row * 32 + h)
        start, end = tl.load(Starts + row), tl.load(Ends + row)
        blocks = tl.cdiv(count, BN)
        if SPLITS == 1:
            first, last = 0, blocks
        else:
            per_split = tl.cdiv(blocks, SPLITS)
            first = tl.program_id(1) * per_split
            last = tl.minimum(first + per_split, blocks)
        for block in tl.range(first, last, num_stages=3):
            col = block * BN + tl.arange(0, BN)
            ids = tl.load(IDs + row.to(tl.int64) * CAP + col, col < count, 0)
            valid = (col < count) & (ids >= 0) & (ids < n)
            safe = tl.where(valid, ids, 0)
            kt = tl.load(K + safe[:, None] * 128 + d[None, :], valid[:, None], 0.0)
            ks = tl.load(KS + safe, valid, 0.0)
            dot = tl.dot(kt, qt, out_dtype=tl.float32, max_num_imprecise_acc=32)
            value = tl.sum(tl.maximum(dot * ks[:, None], 0.0) * w[None, :], 1)
            value = tl.where(valid & (ids >= start) & (ids < end), value, float("-inf"))
            tl.store(Out + row.to(tl.int64) * CAP + col, value, col < count)


from ...api import validate_prepared
from ..linear.score import _collapse_kernel as _collapse_h64


def rerank(prepared, selected, starts, ends, *, splits=None):
    rows, cap = selected.ids.shape
    if splits is None:
        splits = 4 if rows >= 1024 else 1
    if isinstance(splits, bool) or splits not in (1, 2, 4, 8, 16):
        raise ValueError("rerank splits must be one of 1, 2, 4, 8, 16")
    out = torch.empty((rows, cap), dtype=torch.float32, device=prepared.q.device)
    if rows:
        (_rerank_h32 if prepared.q.shape[1] == 32 else p._rerank_counted_kernel)[(rows, splits)](
            prepared.q, prepared.keys, prepared.effective_weights, prepared.key_scales,
            selected.ids, selected.counts, selected.status, starts, ends, out,
            len(prepared.keys), CAP=cap, BN=128, SPLITS=splits, num_warps=4, num_stages=3)
    return out


# Each shape and budget owns its workspace and recovery graph.
_WORKSPACES = {}


def get_workspace(prepared, budget, debug, max_n):
    key = (prepared.q.device, prepared.q.shape[0], tuple(vars(budget).values()), debug, max_n)
    if key not in _WORKSPACES:
        _WORKSPACES[key] = recovery.Workspace(
            prepared.q.shape[0], prepared.q.device, budget, max_n, debug)
    return _WORKSPACES[key]


def checked(prepared, starts, ends, *, budget=p.Budget(), final_k=2048,
            thresholds=None, debug_scores=False, poison_ids=False,
            workspace=None, max_n=1048576):
    """Compact threshold candidates, recover overflow rows, and rerank."""
    validate_prepared(prepared)
    _validate_ranges(prepared, starts, ends)
    if not isinstance(budget, p.Budget):
        raise TypeError("Budget required")
    if (isinstance(final_k, bool) or not isinstance(final_k, int)
            or not 0 < final_k <= min(8192, budget.minimum)):
        raise ValueError("invalid final K")
    rows, n = prepared.q.shape[0], len(prepared.keys)
    with torch.cuda.device(prepared.q.device):
        if rows == 0:
            ids = torch.empty((0, budget.capacity), dtype=torch.int32, device=prepared.q.device)
            count = torch.empty((0,), dtype=torch.int32, device=prepared.q.device)
            tau = torch.empty((0,), dtype=torch.float32, device=prepared.q.device)
            selected = p.Candidates(ids, count, count.clone(), tau, None, None, budget)
            logits = torch.empty_like(ids, dtype=torch.float32)
            indices = torch.empty((0, final_k), dtype=torch.int32, device=prepared.q.device)
            return recovery.DeviceSelection(p.Selection(selected, logits, indices),
                torch.zeros(16, dtype=torch.int32, device=prepared.q.device))
        max_n = max(max_n, n)
        ws = workspace or get_workspace(prepared, budget, debug_scores, max_n)
        assert ws.rows == rows and ws.max_n >= n and ws.budget == budget and ws.device == prepared.q.device
        assert not debug_scores or ws.debug is not None
        if thresholds is not None:
            if (thresholds.shape != (rows,) or thresholds.dtype != torch.float32
                    or thresholds.device != prepared.q.device or not thresholds.is_contiguous()):
                raise ValueError("thresholds must be contiguous FP32 [Q] on the input device")
        ws.starts.copy_(starts)
        ws.ends.copy_(ends)
        ws.counts.zero_()
        ws.status.zero_()
        if debug_scores or poison_ids:
            ws.ids.fill_(-1)
        if debug_scores:
            ws.debug.fill_(torch.nan)
        if prepared.q.shape[1] == 32:
            _collapse_h32[(rows,)](prepared.q, prepared.effective_weights,
                ws.g.high, ws.g.high_scale, ws.g.low, ws.g.low_scale, num_warps=4)
        else:
            _collapse_h64[(rows,)](prepared.q, prepared.effective_weights,
                ws.g.high, ws.g.high_scale, ws.g.low, ws.g.low_scale,
                COMPONENTS=2, num_warps=4)
        operands = (ws.g.high, ws.g.high_scale, ws.g.low, ws.g.low_scale,
                    prepared.keys, prepared.key_scales)
        if thresholds is None:
            p._pilot_kernel[(triton.cdiv(budget.samples, 64), triton.cdiv(rows, 64))](
                *operands, ws.starts, ws.ends, ws.pilot, rows, n, S=budget.samples,
                BQ=64, BN=64, ACC=32, num_warps=4, enable_fp_fusion=False)
            p._threshold_kernel[(rows,)](ws.pilot, ws.starts, ws.ends,
                ws.thresholds, ws.status, n, S=budget.samples, TARGET=budget.target, num_warps=4)
        else:
            ws.thresholds.copy_(thresholds)
        if n:
            p._scan_compact_kernel[(triton.cdiv(n, 64), triton.cdiv(rows, 64))](
                *operands, ws.starts, ws.ends, ws.thresholds, ws.ids, ws.counts,
                ws.debug if debug_scores else ws.ids, rows, n, CAP=budget.capacity,
                BQ=64, BN=64, ACC=32, DEBUG=debug_scores, RESERVE=128,
                ThresholdKeys=None, KEY_THRESHOLD=False, Q_FIRST=False,
                num_warps=4, enable_fp_fusion=False)
        recovery._initial_keys[(triton.cdiv(rows, 256),)](
            ws.thresholds, ws.threshold_keys, rows, B=256)
        recovery._pack_status[(1,)](
            ws.counts, ws.starts, ws.ends, ws.status, ws.rowmap, ws.summary, rows, n,
            MINIMUM=budget.minimum, CAP=budget.capacity, B=triton.next_power_of_2(rows),
            num_warps=8 if rows > 2048 else 4)
        ws.graph(prepared, n, debug_scores).launch()
        selected = p.Candidates(ws.ids, ws.counts, ws.status, ws.thresholds,
            ws.pilot if thresholds is None else None, ws.debug if debug_scores else None,
            budget, ws.threshold_keys)
        logits = rerank(prepared, selected, ws.starts, ws.ends)
        indices = p.finish(selected, logits, ws.starts, k=final_k)
        return recovery.DeviceSelection(p.Selection(selected, logits, indices), ws.summary)


def select(q, keys, scales, weights, starts, ends, *, max_n=1048576):
    prepared = PreparedFP8(q.contiguous(), keys.contiguous(), scales.contiguous(), weights.contiguous())
    return checked(prepared, starts.contiguous(), ends.contiguous(), max_n=max_n)
