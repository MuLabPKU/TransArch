"""September cooperative SM100 scan with explicit complete-row recovery.

The affine high24 candidate key is unchanged. This is not a claim of bitwise
FP32 TopK: approximation recall is measured separately on real operands.
"""
from collections import OrderedDict
import threading

import torch
import triton
import triton.language as tl

from ..triton_kernel import resmisa_mqa_fused_merged_8h_v2_logits_triton as _scores
from .extensions import extension

_WS = OrderedDict()
MAX_WORKSPACES_PER_DEVICE = 2


def workspace(device, m, capacity):
    key = (device, torch.cuda.current_stream(device).cuda_stream,
           threading.get_ident(), m, capacity)
    if key in _WS:
        _WS.move_to_end(key)
        return _WS[key]
    peers = [k for k in _WS if k[0] == device]
    # A scheduler using more than two row shapes still has a finite bound.
    # Drain the current device before dropping any asynchronously used storage.
    if len(peers) >= MAX_WORKSPACES_PER_DEVICE:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot retire Blackwell workspace during capture")
        torch.cuda.synchronize(device)
        del _WS[peers[0]]
    alloc = lambda shape, dtype: torch.empty(shape, device=device, dtype=dtype)
    ix = lambda *shape: alloc(shape, torch.int32)
    fp = lambda *shape: alloc(shape, torch.float32)
    _WS[key] = dict(q=alloc((m, 8, 128), torch.float8_e4m3fn), weights=fp(m, 8),
        zeros=ix(m), prefix=ix(m), suffix=ix(m), origin=fp(m), inv=fp(m),
        th=ix(m), hist=ix(m, 256), cv=alloc((m, capacity), torch.float16),
        ci=ix(m, capacity), cc=ix(m), seed_status=ix(m), status=ix(m), geometry=ix(5))
    return _WS[key]


def dense_select(scores, starts, ends, k=2048, *, initial=False):
    out = torch.empty((scores.shape[0], k), dtype=torch.int32, device=scores.device)
    extension("carry" if initial else "seed").dense_topk_litetopk_(
        scores, starts, ends, out, scores.shape[0], scores.stride(0), scores.stride(1), k, 0, 0)
    return out


@triton.jit
def _inspect_windows(Starts, Ends, Geometry, N: tl.constexpr,
                     ROWS: tl.constexpr, B: tl.constexpr):
    row = tl.arange(0, B)
    start = tl.load(Starts + row, row < ROWS, 0)
    end = tl.load(Ends + row, row < ROWS, 0)
    lo = tl.min(tl.where(row < ROWS, start, 2147483647), 0)
    hi = tl.max(start, 0)
    first = tl.min(tl.where(row < ROWS, end, 2147483647), 0)
    last = tl.max(end, 0)
    bad = tl.sum(((row < ROWS) & ((start < 0) | (end > N) | (start > end))).to(tl.int32), 0)
    tl.store(Geometry, lo)
    tl.store(Geometry + 1, hi)
    tl.store(Geometry + 2, first)
    tl.store(Geometry + 3, last)
    tl.store(Geometry + 4, bad)


@triton.jit
def _prepare_scan(QF, QS, Scale, Weights, PackedQ, PackedWeights,
                  Zeros, PrefixEnds, SuffixStarts, ROWS: tl.constexpr,
                  LO: tl.constexpr, WIDTH: tl.constexpr, B: tl.constexpr):
    j = tl.program_id(0) * B + tl.arange(0, B)
    d = j % 128
    head = j // 128 % 8
    row = j // 1024
    ptr = tl.where(head == 0, QF + row * 128 + d,
                   QS + (row * 7 + head - 1) * 128 + d)
    q = tl.load(ptr, row < ROWS, 0.0)
    tl.store(PackedQ + j, q, row < ROWS)
    wp = tl.where(head == 0, Scale + row, Weights + row * 7 + head - 1)
    w = tl.load(wp, (row < ROWS) & (d == 0), 0)
    tl.store(PackedWeights + row * 8 + head, w, (row < ROWS) & (d == 0))
    first = (row < ROWS) & (head == 0) & (d == 0)
    tl.store(Zeros + row, 0, first)
    tl.store(PrefixEnds + row, WIDTH, first)
    tl.store(SuffixStarts + row, LO + WIDTH, first)


@triton.jit
def _seed_check(Sample,Status,WIDTH:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,B)
    v=tl.load(Sample+row*WIDTH+i,i<WIDTH,0.)
    low=tl.min(tl.where(i<WIDTH,v,float('inf')),0)
    high=tl.max(tl.where(i<WIDTH,v,-float('inf')),0)
    bad=tl.sum(((i<WIDTH)&((v!=v)|(tl.abs(v)==float('inf')))).to(tl.int32),0)>0
    tl.store(Status+row,bad|((high-low)<1e-5*tl.maximum(tl.max(tl.abs(v),0),1.)))


@triton.jit
def _check_ids(Out,Status,SeedStatus,Starts,Ends,N:tl.constexpr,K:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,K)
    ids=tl.load(Out+row*K+i)
    start=tl.load(Starts+row);end=tl.load(Ends+row)
    status=tl.load(Status+row)|tl.load(SeedStatus+row)
    bad=(ids<start)|(ids>=end)|(ids>=N)
    status|=tl.sum(bad.to(tl.int32),0)>0
    tl.store(Status+row,status)
    tl.store(Out+row*K+i,ids-start)


@triton.jit
def _summarize_status(Status, Summary, M:tl.constexpr, B:tl.constexpr):
    i=tl.arange(0,B)
    flags=tl.load(Status+i,i<M,0)
    tl.store(Summary,tl.max(flags,0))


def select(qf, scale, qs, weights, key, kscale, starts, ends, k=2048, *, hot_valid):
    m, n = qf.shape[0], key.shape[0]
    capacity = 65536
    seed = 12288 if n - m + 1 >= 12288 else 8192
    if k != 2048 or qs.shape != (m, 7, 128) or qf.shape != (m, 128) or key.shape != (n, 128):
        raise ValueError("Blackwell scan requires Top2048, fused1+residual7, D128")
    if (weights.shape != (m, 7) or scale.numel() != m or kscale.numel() != n
            or starts.shape != (m,) or ends.shape != (m,)):
        raise ValueError("Invalid Blackwell scale/weight/window shape")
    if not 0 < m <= 8192 or not 0 < n <= 135168:
        raise ValueError("Blackwell scan supports Q<=8192 and context<=135168")
    if not qf.is_cuda or any(t.device != qf.device for t in (scale, qs, weights, key, kscale, starts, ends)):
        raise ValueError("All Blackwell tensors must share one CUDA device")
    if any(t.dtype != torch.float8_e4m3fn for t in (qf, qs, key)):
        raise ValueError("Blackwell scan requires original FP8 Q/K")
    if torch.cuda.get_device_capability(qf.device) != (10, 0):
        raise ValueError("Blackwell scan requires SM100")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Blackwell prefill requires eager execution")
    qf, qs, key = [t.contiguous() for t in (qf, qs, key)]
    scale, weights, kscale = [t.float().contiguous() for t in (scale, weights, kscale)]
    starts, ends = [t.int().contiguous() for t in (starts, ends)]
    ws = workspace(qf.device, m, capacity)
    args = (qf, scale, qs, weights, key, kscale, starts, ends)
    _inspect_windows[(1,)](starts, ends, ws["geometry"], n, m, triton.next_power_of_2(m))
    lo, hi, first_end, last_end, invalid = ws["geometry"].tolist()
    if invalid:
        raise ValueError("Bounds must satisfy 0<=start<=end<=N")
    # The public entry supports a single causal trailing query chunk. Do not
    # apply HOT's CPU bound hints to an arbitrary/mixed-window invocation.
    if lo != 0 or hi != 0 or first_end != n - m + 1 or last_end != n:
        raise ValueError("Blackwell HOT requires zero-start causal chunk bounds")
    if first_end - lo < seed or last_end <= lo + seed:
        raw = _scores(*args)
        out = dense_select(torch.nan_to_num(raw, neginf=-1e30, posinf=1e30), starts, ends, k)
        return out
    _prepare_scan[(triton.cdiv(m * 1024, 256),)](
        qf, qs, scale, weights, ws["q"], ws["weights"], ws["zeros"],
        ws["prefix"], ws["suffix"], m, lo, seed, 256)
    sample = _scores(qf, scale, qs, weights, key[lo:lo+seed], kscale[lo:lo+seed],
                     ws["zeros"], ws["prefix"])
    _seed_check[(m,)](sample, ws["seed_status"], seed, triton.next_power_of_2(seed))
    extension("seed").seed_prep_(sample,
        ws["origin"], ws["inv"], ws["th"], ws["hist"], ws["cv"], ws["ci"], ws["cc"])
    padded = kscale if n % 4 == 0 else torch.nn.functional.pad(kscale, (0, 4 - n % 4))
    extension("scan").scan(ws["q"], key, padded, ws["weights"], ws["suffix"], ends,
        ws["origin"], ws["inv"], ws["th"], ws["cv"], ws["ci"], ws["cc"],
        ws["hist"], hot_valid)
    out = torch.empty((m, k), device=qf.device, dtype=torch.int32)
    extension("final").safe_topk_(ws["cv"], ws["ci"], ws["cc"], out,
                                ws["status"], n)
    _check_ids[(m,)](out, ws["status"], ws["seed_status"], starts, ends, n, k)
    _summarize_status[(1,)](ws["status"], ws["geometry"], m, triton.next_power_of_2(m))
    any_failure = ws["geometry"][0].item()
    flags = ws["status"].tolist() if any_failure else []
    bad = [i for i, flag in enumerate(flags) if flag]
    # Preserve the accepted row grouping and arithmetic. Recovery recomputes
    # complete rows; it cannot silently accept truncated candidate storage.
    for first in range(0, len(bad), 64):
        rows = torch.tensor(bad[first:first+64], device=qf.device, dtype=torch.long)
        gather = lambda t: t.index_select(0, rows)
        fq = gather(qf.view(torch.uint8)).view(qf.dtype)
        sq = gather(qs.view(torch.uint8)).view(qs.dtype)
        st, en = gather(starts), gather(ends)
        values = _scores(fq, gather(scale), sq, gather(weights), key, kscale, st, en)
        ids = dense_select(torch.nan_to_num(values, neginf=-1e30, posinf=1e30), st, en, k)
        out.index_copy_(0, rows, ids)
    return out


