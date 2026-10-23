"""Flat-prefill negative-residual scores for offline and online callers.

Full physical blocks share FP32 statistics. Maxmin can build block-local prefix
and suffix extrema once, then gather legal boundaries; internal same-block
intervals retain an exact reduction fallback. The explicit micro backend instead
combines complete microblock extrema with two disjoint raw-key fragments.
Paged scoring lives separately; legal mean retains the original reduction.
Prefix workspace is O((N + ceil(N/pool) + M)*D + M*H); micro workspace is
O((ceil(N/pool)*ceil(pool/micro) + M)*D + M*H), never M*Nb*H.
Ties prefer lower head IDs. Parallel scoring is enabled only for H64/D128,
whose FP32 reduction order has been checked against the reference.
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


# Lengths/counts are runtime scalars: neither exact values nor alignment should
# create JIT variants for variable-length prefill. Layout parameters stay constexpr.
@triton.jit(do_not_specialize=['N', 'NB'])
def _pool(K, Scale, Starts, Ends, Out,
          N, D: tl.constexpr, NB,
          POOL: tl.constexpr, TN: tl.constexpr, TD: tl.constexpr,
          K_STRIDE: tl.constexpr):
    p = tl.program_id(0)
    dims = tl.program_id(1) * TD + tl.arange(0, TD)
    if p < NB:
        block = p
        lo = block * POOL
        hi = tl.minimum(lo + POOL, N)
    else:
        row = (p - NB) // 2
        side = (p - NB) % 2
        start = tl.load(Starts + row)
        end = tl.load(Ends + row)
        first = start // POOL
        last = tl.maximum(end - 1, 0) // POOL
        block = tl.where(side == 0, first, last)
        lo = tl.maximum(block * POOL, start)
        hi = tl.minimum((block + 1) * POOL, end)
        # First slot handles both boundaries when they fall in the same block.
        hi = tl.where((side == 1) & (first == last), lo, hi)
    tok = block * POOL + tl.arange(0, TN)
    valid = (tok >= lo) & (tok < hi) & (tok < N)
    key = tl.load(K + tok[:, None] * K_STRIDE + dims[None, :],
                  valid[:, None] & (dims[None, :] < D), other=0.0).to(tl.float32)
    scale = tl.load(Scale + tok, valid, other=0.0).to(tl.float32)
    key = key * scale[:, None]
    count = tl.sum(valid.to(tl.int32), 0)
    mx = tl.max(tl.where(valid[:, None], key, -float('inf')), 0)
    mn = tl.min(tl.where(valid[:, None], key, float('inf')), 0)
    mean = tl.sum(key, 0) / tl.maximum(count, 1)
    tl.store(Out + p * 3 * D + dims, tl.where(count > 0, mx, 0.), dims < D)
    tl.store(Out + p * 3 * D + D + dims, tl.where(count > 0, mn, 0.), dims < D)
    tl.store(Out + p * 3 * D + 2 * D + dims, mean, dims < D)


@triton.jit(do_not_specialize=['NB'])
def _score(Q, W, Bounds, Starts, Ends, Scores,
           H: tl.constexpr, D: tl.constexpr, BH: tl.constexpr, BD: tl.constexpr,
           NB, POOL: tl.constexpr, MEAN: tl.constexpr):
    row = tl.program_id(0)
    hs, ds = tl.arange(0, BH), tl.arange(0, BD)
    q = tl.load(Q + row * H * D + hs[:, None] * D + ds[None, :],
                (hs[:, None] < H) & (ds[None, :] < D), other=0.0).to(tl.float32)
    pos, neg = tl.maximum(q, 0.), tl.maximum(-q, 0.)
    start, end = tl.load(Starts + row), tl.load(Ends + row)
    first, last = start // POOL, tl.maximum(end - 1, 0) // POOL
    best = tl.full((BH,), 0., tl.float32)
    for block in range(first, tl.cdiv(end, POOL)):
        p = block
        p = tl.where((block == last) & (end % POOL != 0), NB + row * 2 + 1, p)
        p = tl.where((block == first) & ((start % POOL != 0) | (first == last)),
                     NB + row * 2, p)
        if MEAN:
            mean = tl.load(Bounds + p * 3 * D + 2 * D + ds, ds < D, other=0.0)
            response = tl.maximum(-tl.sum(q * mean[None, :], 1), 0.)
            best += response
        else:
            mx = tl.load(Bounds + p * 3 * D + ds, ds < D, other=0.0)
            mn = tl.load(Bounds + p * 3 * D + D + ds, ds < D, other=0.0)
            upper = tl.sum(neg * mx[None, :] - pos * mn[None, :], 1)
            best = tl.maximum(best, upper)
    w = tl.load(W + row * H + hs, hs < H, other=0.0).to(tl.float32)
    tl.store(Scores + row * H + hs, best * tl.abs(w), hs < H)


@gluon.jit(do_not_specialize=['NB'])
def _score_parallel(Q, W, Bounds, Starts, Ends, Scores,
                    H: gl.constexpr, D: gl.constexpr, BD: gl.constexpr,
                    NB, POOL: gl.constexpr, HH: gl.constexpr, BB: gl.constexpr,
                    HW: gl.constexpr = 1, COALESCED: gl.constexpr = False):
    if COALESCED:
        # Consecutive lanes read consecutive dimensions; tree reduction permits
        # small FP32 rounding differences relative to the serial reference.
        layout: gl.constexpr = BlockedLayout([1, 1, 1], [1, 1, 32], [HW, 4 // HW, 1], [2, 0, 1])
    else:
        layout: gl.constexpr = BlockedLayout([1, 1, 16], [4, 1, 8], [HW, 4 // HW, 1], [2, 0, 1])
    row = gl.program_id(0)
    hs = gl.program_id(1) * HH + gl.arange(0, HH, layout=SliceLayout(1, SliceLayout(2, layout)))
    ds = gl.arange(0, BD, layout=SliceLayout(0, SliceLayout(0, layout)))
    bs = gl.arange(0, BB, layout=SliceLayout(0, SliceLayout(2, layout)))
    q = gl.load(Q + row * H * D + hs[:, None, None] * D + ds[None, :][None, :, :],
                (hs[:, None, None] < H) & (ds[None, :][None, :, :] < D), other=0.).to(gl.float32)
    pos, neg = gl.maximum(q, 0.), gl.maximum(-q, 0.)
    start, end = gl.load(Starts + row), gl.load(Ends + row)
    first, last = start // POOL, gl.maximum(end - 1, 0) // POOL
    best = gl.full((HH, BB), 0., gl.float32, SliceLayout(2, layout))
    for base in range(first, gl.cdiv(end, POOL), BB):
        block = base + bs
        valid = block < gl.cdiv(end, POOL)
        p = gl.where((block == last) & (end % POOL != 0), NB + row * 2 + 1, block)
        p = gl.where((block == first) & ((start % POOL != 0) | (first == last)),
                     NB + row * 2, p)
        mask = valid[None, :, None] & (ds[None, :][None, :, :] < D)
        mx = gl.load(Bounds + p[None, :, None] * 3 * D + ds[None, :][None, :, :], mask, other=0.)
        mn = gl.load(Bounds + p[None, :, None] * 3 * D + D + ds[None, :][None, :, :], mask, other=0.)
        upper = gl.sum(neg * mx - pos * mn, 2)
        best = gl.maximum(best, gl.where(valid[None, :], upper, 0.))
    result = gl.max(best, 1)
    w = gl.load(W + row * H + hs, hs < H, other=0.).to(gl.float32)
    gl.store(Scores + row * H + hs, result * gl.abs(w), hs < H)


@triton.jit(do_not_specialize=['NB'])
def _score_tensor(Q, W, Bounds, Starts, Ends, Scores,
                  H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                  NB, POOL: tl.constexpr, HH: tl.constexpr, BB: tl.constexpr):
    row = tl.program_id(0)
    hs = tl.program_id(1) * HH + tl.arange(0, HH)
    ds, bs = tl.arange(0, BD), tl.arange(0, BB)
    q = tl.load(Q + row * H * D + hs[:, None] * D + ds[None, :],
                (hs[:, None] < H) & (ds[None, :] < D), other=0.).to(tl.float32)
    pos, neg = tl.maximum(q, 0.), tl.maximum(-q, 0.)
    start, end = tl.load(Starts + row), tl.load(Ends + row)
    first, last = start // POOL, tl.maximum(end - 1, 0) // POOL
    best = tl.full((HH, BB), 0., tl.float32)
    for base in range(first, tl.cdiv(end, POOL), BB):
        block = base + bs
        valid = block < tl.cdiv(end, POOL)
        p = tl.where((block == last) & (end % POOL != 0), NB + row * 2 + 1, block)
        p = tl.where((block == first) & ((start % POOL != 0) | (first == last)),
                     NB + row * 2, p)
        mask = (ds[:, None] < D) & valid[None, :]
        mx = tl.load(Bounds + p[None, :] * 3 * D + ds[:, None], mask, other=0.)
        mn = tl.load(Bounds + p[None, :] * 3 * D + D + ds[:, None], mask, other=0.)
        # Three TF32 products retain FP32 input precision, unlike plain TF32.
        upper = tl.dot(neg, mx, input_precision='tf32x3') - tl.dot(pos, mn, input_precision='tf32x3')
        best = tl.maximum(best, tl.where(valid[None, :], upper, 0.))
    result = tl.max(best, 1)
    w = tl.load(W + row * H + hs, hs < H, other=0.).to(tl.float32)
    tl.store(Scores + row * H + hs, result * tl.abs(w), hs < H)


@triton.jit(do_not_specialize=['M', 'NB'])
def _score_shared(Q, W, Bounds, Starts, Ends, Scores,
                  M, H: tl.constexpr, D: tl.constexpr, NB,
                  POOL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  SPLIT_BF16: tl.constexpr):
    # Matrix rows span multiple queries; only complete blocks enter the GEMM.
    ids = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = ids // H
    bs = tl.arange(0, BN)
    # Keep dot operands small; load query fragments inside the K loop.
    kd = tl.arange(0, 32)
    start = tl.load(Starts + rows, rows < M, other=0)
    end = tl.load(Ends + rows, rows < M, other=0)
    first, last = start // POOL, tl.maximum(end - 1, 0) // POOL
    # Match the reference's special same-block first-boundary slot.
    lo = first + ((start % POOL != 0) | (first == last)).to(tl.int32)
    hi = end // POOL
    lower = tl.min(tl.where((rows < M) & (lo < hi), lo, NB), 0)
    upper = tl.max(tl.where(rows < M, hi, 0), 0)
    best = tl.full((BM, BN), 0., tl.float32)
    for base in range(lower, upper, BN):
        block = base + bs
        positive = tl.full((BM, BN), 0., tl.float32)
        negative = tl.full((BM, BN), 0., tl.float32)
        for db in range(tl.cdiv(D, 32)):
            dims = db * 32 + kd
            qpart = tl.load(Q + ids[:, None] * D + dims[None, :],
                (rows[:, None] < M) & (dims[None, :] < D), other=0.).to(tl.float32)
            pos, neg = tl.maximum(qpart, 0.), tl.maximum(-qpart, 0.)
            mask = (dims[:, None] < D) & (block[None, :] < NB)
            mx = tl.load(Bounds + block[None, :] * 3 * D + dims[:, None], mask, other=0.)
            mn = tl.load(Bounds + block[None, :] * 3 * D + D + dims[:, None], mask, other=0.)
            if SPLIT_BF16:
                # FP8 Q is exact in BF16; split only the FP32 K extrema.
                xh, nh = mx.to(tl.bfloat16), mn.to(tl.bfloat16)
                xl = (mx - xh.to(tl.float32)).to(tl.bfloat16)
                nl = (mn - nh.to(tl.float32)).to(tl.bfloat16)
                a, b = neg.to(tl.bfloat16), pos.to(tl.bfloat16)
                positive = tl.dot(a, xh, positive)
                positive = tl.dot(a, xl, positive)
                negative = tl.dot(b, nh, negative)
                negative = tl.dot(b, nl, negative)
            else:
                positive = tl.dot(neg, mx, positive, input_precision='tf32x3')
                negative = tl.dot(pos, mn, negative, input_precision='tf32x3')
        score = positive - negative
        valid = (block[None, :] >= lo[:, None]) & (block[None, :] < hi[:, None])
        best = tl.maximum(best, tl.where(valid, score, 0.))
    result = tl.max(best, 1)
    # Query-specific boundaries use FP32 products with a compact 32-D tile.
    for side in tl.static_range(2):
        used = (rows < M) & (start < end) & tl.where(side == 0,
            (start % POOL != 0) | (first == last),
            (end % POOL != 0) & (first != last))
        p = NB + rows * 2 + side
        accum = tl.full((BM, 32), 0., tl.float32)
        for db in range(tl.cdiv(D, 32)):
            dims = db * 32 + kd
            mask = used[:, None] & (dims[None, :] < D)
            qpart = tl.load(Q + ids[:, None] * D + dims[None, :], mask, other=0.).to(tl.float32)
            mx = tl.load(Bounds + p[:, None] * 3 * D + dims[None, :], mask, other=0.)
            mn = tl.load(Bounds + p[:, None] * 3 * D + D + dims[None, :], mask,other=0.)
            accum += tl.maximum(-qpart, 0.) * mx - tl.maximum(qpart, 0.) * mn
        boundary = tl.sum(accum, 1)
        result = tl.maximum(result, tl.where(used, boundary, 0.))
    weight = tl.load(W + ids, rows < M, other=0.).to(tl.float32)
    tl.store(Scores + ids, result * tl.abs(weight), rows < M)


@triton.jit(do_not_specialize=['NB'])
def _pack_extrema(Bounds, Packed, NB, D: tl.constexpr, NP: tl.constexpr,
                  B: tl.constexpr):
    x = tl.program_id(0) * B + tl.arange(0, B)
    block = x % NP
    dim = x // NP
    value = tl.load(Bounds + block * 3 * D + dim % D + (dim // D) * D,
                    (block < NB) & (dim < 2 * D), other=0.)
    value = tl.where(dim < D, value, -value)
    high = value.to(tl.bfloat16)
    low = (value - high.to(tl.float32)).to(tl.bfloat16)
    tl.store(Packed + x, high, dim < 2 * D)
    tl.store(Packed + 2 * D * NP + x, low, dim < 2 * D)


@triton.jit(do_not_specialize=['M', 'NB'])
def _score_packed_full(Q, Packed, Starts, Ends, Partial,
                       M, NB, H: tl.constexpr, D: tl.constexpr,
                       POOL: tl.constexpr, NP: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       SPLITS: tl.constexpr, LOW: tl.constexpr):
    ids = tl.program_id(0) * BM + tl.arange(0, BM)
    row = ids // H
    split = tl.program_id(1)
    start = tl.load(Starts + row, row < M, other=0)
    end = tl.load(Ends + row, row < M, other=0)
    first, last = start // POOL, tl.maximum(end - 1, 0) // POOL
    lo = first + ((start % POOL != 0) | (first == last)).to(tl.int32)
    hi = end // POOL
    lower = tl.min(tl.where((row < M) & (lo < hi), lo, NB), 0)
    upper = tl.max(tl.where(row < M, hi, 0), 0)
    best = tl.full((BM, BN), 0., tl.float32)
    kd = tl.arange(0, BK)
    for base in range(lower + split * BN, upper, BN * SPLITS):
        blocks = base + tl.arange(0, BN)
        acc = tl.full((BM, BN), 0., tl.float32)
        for db in range(tl.cdiv(2 * D, BK)):
            dims = db * BK + kd
            q = tl.load(Q + ids[:, None] * D + dims[None, :] % D,
                        (row[:, None] < M) & (dims[None, :] < 2 * D), other=0.).to(tl.float32)
            a = tl.maximum(tl.where(dims[None, :] < D, -q, q), 0.).to(tl.bfloat16)
            offsets = dims[:, None] * NP + blocks[None, :]
            mask = (dims[:, None] < 2 * D) & (blocks[None, :] < NB)
            b = tl.load(Packed + offsets, mask, other=0.)
            acc = tl.dot(a, b, acc)
            if LOW:
                low = tl.load(Packed + 2 * D * NP + offsets, mask, other=0.)
                acc = tl.dot(a, low, acc)
        valid = (blocks[None, :] >= lo[:, None]) & (blocks[None, :] < hi[:, None])
        best = tl.maximum(best, tl.where(valid, acc, 0.))
    tl.store(Partial + split * M * H + ids, tl.max(best, 1), row < M)


@triton.jit(do_not_specialize=['M', 'NB'])
def _score_packed_finish(Q, W, Bounds, Starts, Ends, Partial, Scores,
                         M, NB, H: tl.constexpr, D: tl.constexpr,
                         POOL: tl.constexpr, SPLITS: tl.constexpr,
                         BH: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    dims = tl.arange(0, BD)
    q = tl.load(Q + row * H * D + heads[:, None] * D + dims[None, :],
                (heads[:, None] < H) & (dims[None, :] < D), other=0.).to(tl.float32)
    start, end = tl.load(Starts + row), tl.load(Ends + row)
    first, last = start // POOL, tl.maximum(end - 1, 0) // POOL
    result = tl.full((BH,), 0., tl.float32)
    for split in tl.static_range(SPLITS):
        value = tl.load(Partial + split * M * H + row * H + heads, heads < H, other=0.)
        result = tl.maximum(result, value)
    for side in tl.static_range(2):
        used = (start < end) & tl.where(side == 0,
            (start % POOL != 0) | (first == last),
            (end % POOL != 0) & (first != last))
        p = NB + row * 2 + side
        mx = tl.load(Bounds + p * 3 * D + dims, used & (dims < D), other=0.)
        mn = tl.load(Bounds + p * 3 * D + D + dims, used & (dims < D), other=0.)
        value = tl.sum(tl.maximum(-q, 0.) * mx[None, :] - tl.maximum(q, 0.) * mn[None, :], 1)
        result = tl.maximum(result, tl.where(used, value, 0.))
    w = tl.load(W + row * H + heads, heads < H, other=0.).to(tl.float32)
    tl.store(Scores + row * H + heads, result * tl.abs(w), heads < H)


@triton.jit
def _max_pair(a, b):
    return tl.maximum(a, b)


@triton.jit
def _min_pair(a, b):
    return tl.minimum(a, b)


@triton.jit(do_not_specialize=['N'])
def _prefix_extrema(K, Scale, Prefix, Suffix, Bounds,
                    N, D: tl.constexpr, POOL: tl.constexpr,
                    TN: tl.constexpr, TD: tl.constexpr):
    block = tl.program_id(0)
    ds = tl.program_id(1) * TD + tl.arange(0, TD)
    ts = block * POOL + tl.arange(0, TN)
    valid = (ts < N) & (tl.arange(0, TN) < POOL)
    key = tl.load(K + ts[:, None] * D + ds[None, :],
                  valid[:, None] & (ds[None, :] < D), other=0.).to(tl.float32)
    scale = tl.load(Scale + ts, valid, other=0.)
    key = key * scale[:, None]
    mx = tl.where(valid[:, None], key, -float('inf'))
    mn = tl.where(valid[:, None], key, float('inf'))
    pmx = tl.associative_scan(mx, 0, _max_pair)
    pmn = tl.associative_scan(mn, 0, _min_pair)
    smx = tl.associative_scan(mx, 0, _max_pair, reverse=True)
    smn = tl.associative_scan(mn, 0, _min_pair, reverse=True)
    offsets = ts[:, None] * 2 * D + ds[None, :]
    mask = valid[:, None] & (ds[None, :] < D)
    tl.store(Prefix + offsets, pmx, mask)
    tl.store(Prefix + offsets + D, pmn, mask)
    tl.store(Suffix + offsets, smx, mask)
    tl.store(Suffix + offsets + D, smn, mask)
    tl.store(Bounds + block * 3 * D + ds, tl.max(mx, 0), ds < D)
    tl.store(Bounds + block * 3 * D + D + ds, tl.min(mn, 0), ds < D)


@triton.jit(do_not_specialize=['M', 'N', 'NB'])
def _boundary_lookup(Starts, Ends, Prefix, Suffix, Bounds,
                     M, N, D: tl.constexpr,
                     NB, POOL: tl.constexpr, BD: tl.constexpr,
                     BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    ds = tl.arange(0, BD)
    start = tl.load(Starts + rows, rows < M, other=0)
    end = tl.load(Ends + rows, rows < M, other=0)
    first, last = start // POOL, tl.maximum(end - 1, 0) // POOL
    for side in tl.static_range(2):
        block = tl.where(side == 0, first, last)
        lo = tl.maximum(block * POOL, start)
        hi = tl.minimum((block + 1) * POOL, end)
        used = (rows < M) & (start < end) & tl.where(side == 0,
            (start % POOL != 0) | (first == last),
            (end % POOL != 0) & (first != last))
        is_prefix = used & (lo == block * POOL)
        is_suffix = used & ~is_prefix & (hi == tl.minimum((block + 1) * POOL, N))
        po = (hi - 1)[:, None] * 2 * D + ds[None, :]
        so = lo[:, None] * 2 * D + ds[None, :]
        pm = is_prefix[:, None] & (ds[None, :] < D)
        sm = is_suffix[:, None] & (ds[None, :] < D)
        mx = tl.load(Prefix + po, pm, other=0.)
        mn = tl.load(Prefix + po + D, pm, other=0.)
        sx = tl.load(Suffix + so, sm, other=0.)
        sn = tl.load(Suffix + so + D, sm, other=0.)
        mx = tl.where(is_suffix[:, None], sx, mx)
        mn = tl.where(is_suffix[:, None], sn, mn)
        out = (NB + rows * 2 + side)[:, None] * 3 * D + ds[None, :]
        mask = (rows[:, None] < M) & (ds[None, :] < D)
        tl.store(Bounds + out, mx, mask)
        tl.store(Bounds + out + D, mn, mask)


@triton.jit(do_not_specialize=['N', 'NB'])
def _boundary_extrema(K, Scale, Starts, Ends, Prefix, Suffix, Bounds,
                      N, D: tl.constexpr, NB,
                      POOL: tl.constexpr, TN: tl.constexpr, TD: tl.constexpr):
    row = tl.program_id(0)
    side = tl.program_id(1)
    ds = tl.program_id(2) * TD + tl.arange(0, TD)
    start, end = tl.load(Starts + row), tl.load(Ends + row)
    first, last = start // POOL, tl.maximum(end - 1, 0) // POOL
    block = tl.where(side == 0, first, last)
    lo = tl.maximum(block * POOL, start)
    hi = tl.minimum((block + 1) * POOL, end)
    used = ((start < end) & (side == 0) & (first == last)
            & (start % POOL != 0)
            & (end < tl.minimum((first + 1) * POOL, N)))
    if used:
        if lo == block * POOL:
            mx = tl.load(Prefix + (hi - 1) * 2 * D + ds, ds < D, other=0.)
            mn = tl.load(Prefix + (hi - 1) * 2 * D + D + ds, ds < D, other=0.)
        elif hi == tl.minimum((block + 1) * POOL, N):
            mx = tl.load(Suffix + lo * 2 * D + ds, ds < D, other=0.)
            mn = tl.load(Suffix + lo * 2 * D + D + ds, ds < D, other=0.)
        else:
            # An internal same-block interval cannot use prefix subtraction.
            ts = block * POOL + tl.arange(0, TN)
            valid = (ts >= lo) & (ts < hi) & (ts < N)
            key = tl.load(K + ts[:, None] * D + ds[None, :],
                          valid[:, None] & (ds[None, :] < D), other=0.).to(tl.float32)
            scale = tl.load(Scale + ts, valid, other=0.)
            key = key * scale[:, None]
            mx = tl.max(tl.where(valid[:, None], key, -float('inf')), 0)
            mn = tl.min(tl.where(valid[:, None], key, float('inf')), 0)
    else:
        mx = tl.full((TD,), 0., tl.float32)
        mn = tl.full((TD,), 0., tl.float32)
    # Other slots were populated by the batched lookup kernel.
    if used:
        p = NB + row * 2 + side
        tl.store(Bounds + p * 3 * D + ds, mx, ds < D)
        tl.store(Bounds + p * 3 * D + D + ds, mn, ds < D)


@triton.jit(do_not_specialize=['N'])
def _micro_extrema(K, Scale, Micro, N, D: tl.constexpr, POOL: tl.constexpr,
                   MICRO: tl.constexpr, C: tl.constexpr, TD: tl.constexpr):
    block, chunk = tl.program_id(0), tl.program_id(1)
    ds = tl.program_id(2) * TD + tl.arange(0, TD)
    ts = block * POOL + chunk * MICRO + tl.arange(0, MICRO)
    valid = (ts < N) & (ts < (block + 1) * POOL)
    key = tl.load(K + ts[:, None] * D + ds[None, :],
                  valid[:, None] & (ds[None, :] < D), other=0.).to(tl.float32)
    scale = tl.load(Scale + ts, valid, other=0.)
    key = key * scale[:, None]
    mx = tl.max(tl.where(valid[:, None], key, -float('inf')), 0)
    mn = tl.min(tl.where(valid[:, None], key, float('inf')), 0)
    offset = (block * C + chunk) * 2 * D + ds
    tl.store(Micro + offset, mx, ds < D)
    tl.store(Micro + offset + D, mn, ds < D)


@triton.jit(do_not_specialize=['N', 'NB'])
def _micro_bounds(K, Scale, Micro, Starts, Ends, Bounds, N, NB,
                  D: tl.constexpr, POOL: tl.constexpr, MICRO: tl.constexpr,
                  C: tl.constexpr, BC: tl.constexpr, TD: tl.constexpr):
    p = tl.program_id(0)
    ds = tl.program_id(1) * TD + tl.arange(0, TD)
    if p < NB:
        block = p
        lo = 0
        hi = tl.minimum(POOL, N - block * POOL)
    else:
        row, side = (p - NB) // 2, (p - NB) % 2
        start, end = tl.load(Starts + row), tl.load(Ends + row)
        first, last = start // POOL, tl.maximum(end - 1, 0) // POOL
        block = tl.where(side == 0, first, last)
        lo = tl.maximum(start - block * POOL, 0)
        hi = tl.minimum(end - block * POOL, POOL)
        hi = tl.where((side == 1) & (first == last), lo, hi)
    cs = tl.arange(0, BC)
    a, z = tl.cdiv(lo, MICRO), hi // MICRO
    complete = (cs >= a) & (cs < z) & (cs < C) & (lo < hi)
    off = (block * C + cs[:, None]) * 2 * D + ds[None, :]
    mx = tl.load(Micro + off, complete[:, None] & (ds[None, :] < D), other=-float('inf'))
    mn = tl.load(Micro + off + D, complete[:, None] & (ds[None, :] < D), other=float('inf'))
    vmax, vmin = tl.max(mx, 0), tl.min(mn, 0)
    # Two disjoint fragments, including same-microblock intervals exactly once.
    rs = tl.arange(0, MICRO)
    for side in tl.static_range(2):
        begin = tl.where(side == 0, lo, tl.maximum(lo, z * MICRO))
        stop = tl.where(side == 0, tl.minimum(hi, a * MICRO), hi)
        used = (lo < hi) & tl.where(side == 0, lo % MICRO != 0,
                                                   (hi % MICRO != 0) & (z >= a))
        ts = block * POOL + begin + rs
        valid = used & (begin + rs < stop) & (ts < N)
        key = tl.load(K + ts[:, None] * D + ds[None, :],
                      valid[:, None] & (ds[None, :] < D), other=0.).to(tl.float32)
        scale = tl.load(Scale + ts, valid, other=0.)
        key = key * scale[:, None]
        vmax = tl.maximum(vmax, tl.max(tl.where(valid[:, None], key, -float('inf')), 0))
        vmin = tl.minimum(vmin, tl.min(tl.where(valid[:, None], key, float('inf')), 0))
    tl.store(Bounds + p * 3 * D + ds, tl.where(lo < hi, vmax, 0.), ds < D)
    tl.store(Bounds + p * 3 * D + D + ds, tl.where(lo < hi, vmin, 0.), ds < D)


def maxmin_head_scores(q, k, scales, weights, starts, ends, pool_size,
                       scoring="maxmin", maxmin_backend="auto", *, validate_bounds=True,
                       score_backend="strict"):
    """Return FP32 [M,H], using maxmin max or legal-mean sum over blocks.

    Inputs are flat Q [M,H,D], K [N,D], FP32 per-token K scales and effective
    signed weights [M,H] (Q scale already absorbed). Invalid bounds are errors;
    start == end is empty. No implicit scale bit reinterpretation orclamping.
    score_backend defaults to strict. Explicit shared_bf16/shared_tf32 are
    experimental H64/D128 paths: neither guarantees absolute error <= 1e-3
    for arbitrary input magnitudes. Other shapes and reference retain fallback.
    """
    if q.ndim != 3 or k.ndim != 2 or q.shape[-1] != k.shape[-1]:
        raise ValueError("flat Q [M,H,D] and K [N,D] required")
    if scoring not in ("maxmin", "legal_mean"):
        raise ValueError("scoring must be maxmin or legal_mean")
    if not isinstance(pool_size, int) or pool_size <= 0:
        raise ValueError("positive integer pool_size required")
    m, h, d = q.shape
    n = k.shape[0]
    if scales.dtype != torch.float32 or scales.numel() != n:
        raise ValueError("one FP32 dequantization scale per K token required")
    if weights.shape != (m, h) or starts.numel() != m or ends.numel() != m:
        raise ValueError("weights/bounds shape mismatch")
    if not all(t.is_cuda and t.device == q.device for t in (q, k, scales, weights, starts, ends)):
        raise ValueError("all inputs must be on the same CUDA device")
    if starts.dtype not in (torch.int32, torch.int64) or ends.dtype not in (torch.int32, torch.int64):
        raise ValueError("integer token bounds required")
    # Online callers supply validated attention metadata; skip host reads so
    # changing GPU boundaries remains compatible with CUDA Graph replay.
    if validate_bounds and bool(((starts < 0) | (ends < starts) | (ends > n)).any()):
        raise ValueError("require 0 <= start <= end <= N")
    starts, ends = starts.reshape(-1).int().contiguous(), ends.reshape(-1).int().contiguous()
    nb = triton.cdiv(n, pool_size)
    bounds = torch.empty((nb + 2 * m, 3, d), device=q.device, dtype=torch.float32)
    scores = torch.empty((m, h), device=q.device, dtype=torch.float32)
    if m == 0:
        return scores
    if maxmin_backend not in ("auto", "reference", "prefix", "micro"):
        raise ValueError("unknown maxmin backend")
    use_micro = scoring == "maxmin" and n > 0 and maxmin_backend == "micro"
    # Small batches use the original reduction to avoid scan/workspace overhead.
    use_prefix = scoring == "maxmin" and (maxmin_backend == "prefix" or
        (maxmin_backend == "auto" and m >= nb and pool_size >= 16 and n > 0))
    if use_micro:
        micro_size = min(32, triton.next_power_of_2(pool_size))
        chunks = triton.cdiv(pool_size, micro_size)
        micro = torch.empty((nb, chunks, 2, d), device=q.device, dtype=torch.float32)
        _micro_extrema[(nb, chunks, triton.cdiv(d, 32))](
            k.contiguous(), scales.contiguous(), micro, n, d, pool_size,
            micro_size, chunks, 32, num_warps=4, enable_fp_fusion=False)
        _micro_bounds[(nb + 2 * m, triton.cdiv(d, 32))](
            k.contiguous(), scales.contiguous(), micro, starts, ends, bounds,
            n, nb, d, pool_size, micro_size, chunks, triton.next_power_of_2(chunks),
            32, num_warps=4, enable_fp_fusion=False)
    elif use_prefix and n > 0:
        prefix = torch.empty((n, 2, d), device=q.device, dtype=torch.float32)
        suffix = torch.empty_like(prefix)
        _prefix_extrema[(nb, triton.cdiv(d, 8))](
            k.contiguous(), scales.contiguous(), prefix, suffix, bounds,
            n, d, pool_size, triton.next_power_of_2(pool_size), 8, num_warps=8)
        _boundary_lookup[(triton.cdiv(m, 16),)](
            starts, ends, prefix, suffix, bounds, m, n, d, nb, pool_size,
            triton.next_power_of_2(d), 16, num_warps=4)
        _boundary_extrema[(m, 1, triton.cdiv(d, 32))](
            k.contiguous(), scales.contiguous(), starts, ends, prefix, suffix, bounds,
            n, d, nb, pool_size, triton.next_power_of_2(pool_size), 32, num_warps=8)
    else:
        _pool[(nb + 2 * m, triton.cdiv(d, 16))](
            k.contiguous(), scales.contiguous(), starts, ends, bounds,
            n, d, nb, pool_size, triton.next_power_of_2(pool_size), 16, d, num_warps=8)
    if score_backend not in ("strict", "shared_bf16", "shared_tf32"):
        raise ValueError("unknown score backend")
    if (score_backend == "shared_bf16" and scoring == "maxmin" and
            maxmin_backend != "reference" and q.dtype not in (torch.float8_e4m3fn, torch.bfloat16)):
        raise ValueError("shared_bf16 requires Q exactly representable in BF16")
    if scoring == "maxmin" and maxmin_backend != "reference" and d == 128 and h == 64 and score_backend != "strict":
        _score_shared[(m,)](
            q.contiguous(), weights.contiguous(), bounds, starts, ends, scores,
            m, h, d, nb, pool_size, 64, 32, score_backend == "shared_bf16",
            num_warps=4, num_stages=1, enable_fp_fusion=False)
    elif scoring == "maxmin" and maxmin_backend != "reference" and d == 128 and h == 64:
        _score_parallel[(m, triton.cdiv(h, 32))](
            q.contiguous(), weights.contiguous(), bounds, starts, ends, scores,
            h, d, triton.next_power_of_2(d), nb, pool_size, 32, 4,
            num_warps=4, enable_fp_fusion=False)
    else:
        _score[(m,)](q.contiguous(), weights.contiguous(), bounds, starts, ends, scores,
                     h, d, triton.next_power_of_2(h), triton.next_power_of_2(d),
                     nb, pool_size, scoring == "legal_mean", num_warps=8, enable_fp_fusion=False)
    return scores


@triton.jit(do_not_specialize=['SIZE'])
def _weighted_q(Q, W, Product, SIZE, H: tl.constexpr,
                D: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    q = tl.load(Q + i, i < SIZE, other=0.).to(tl.float32)
    w = tl.load(W + i // D, i < SIZE, other=0.).to(tl.float32)
    tl.store(Product + i, q * w, i < SIZE)


@triton.jit(do_not_specialize=['M'])
def _quantize_fused(Fused, Out, Scale, M, D: tl.constexpr,
                    BD: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    ds = tl.arange(0, BD)
    mask = (rows[:, None] < M) & (ds[None, :] < D)
    x = tl.load(Fused + rows[:, None] * D + ds[None, :], mask, other=0.)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1.e-12)
    # PyTorch scalar division is lowered to multiplication by an FP32 reciprocal.
    scale = amax * (1. / 448.)
    quantized = tl.div_rn(x, scale[:, None])
    tl.store(Out + rows[:, None] * D + ds[None, :], quantized, mask)
    tl.store(Scale + rows, scale, rows < M)


def build_fused_head_q_exact(q, weights):
    """Flat Q; retain PyTorch's original FP32 head reduction."""
    m, h, d = q.shape
    product = torch.empty((m, h, d), device=q.device, dtype=torch.float32)
    _weighted_q[(triton.cdiv(m * h * d, 1024),)](
        q.contiguous(), weights.contiguous(), product, m * h * d, h, d, 1024)
    fused = product.sum(dim=-2)
    out = torch.empty((m, d), device=q.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((m,), device=q.device, dtype=torch.float32)
    if m:
        _quantize_fused[(triton.cdiv(m, 16),)](
            fused, out, scale, m, d, triton.next_power_of_2(d), 16,
            enable_fp_fusion=False)
    return out, scale


def deterministic_heads(scores, count):
    """Descending score, then ascending head ID, including zero/empty ties."""
    if not 0 <= count <= scores.shape[-1]:
        raise ValueError("invalid residual head count")
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :count].contiguous()