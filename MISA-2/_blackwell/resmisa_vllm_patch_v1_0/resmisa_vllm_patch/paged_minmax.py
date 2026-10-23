"""FP32 paged negative-residual head selection, independent of mean pooling.

KV pages are segmented E4M3FN bytes then FP32 token scales, NOT interleaved.
Cache correctness requires ordered calls on one CUDA stream per KV resource.
Decode/verify may append, but tokens before the previous minimum visible end
must remain immutable. Speculative suffixes are always recomputed. Prefill,
page reuse or arbitrary in-place rewrites MUST call invalidate_minmax_cache
AFTER KV writes. Identical metadata cannot reveal a new sequence's identity.
Reset/invalidation retain every allocation referenced by captured graphs.
"""
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(Lens, Tables, Offsets, Previous, OldTables, Valid, Clock, Stamp,
             Reuse, B: tl.constexpr, S: tl.constexpr, C: tl.constexpr,
             CAP: tl.constexpr, TC: tl.constexpr, TS: tl.constexpr):
    b = tl.program_id(0)
    if b >= B:
        # A smaller replay must not make dormant rows fresh via the shared stamp.
        tl.store(Valid + b, 0)
    else:
        cs = tl.arange(0, TC)
        js = tl.arange(0, TS)
        length = tl.load(Lens + b).to(tl.int64)
        off = tl.load(Offsets + b * S + js, js < S, other=0).to(tl.int64)
        safe = tl.minimum(tl.maximum(length - tl.max(off, 0), 0), CAP)
        table = tl.load(Tables + b * C + cs, cs < C, other=-1)
        old = tl.load(OldTables + b * C + cs, cs < C, other=-1)
        same = tl.sum((table != old).to(tl.int32), 0) == 0
        prev_len = tl.load(Previous + b * 2)
        prev_safe = tl.load(Previous + b * 2 + 1)
        valid = (tl.load(Valid + b) != 0) & (tl.load(Clock) == tl.load(Stamp))
        valid = valid & same & (length >= prev_len) & (safe >= prev_safe)
        tl.store(Reuse + b, tl.where(valid, prev_safe, 0))
        tl.store(Previous + b * 2, length)
        tl.store(Previous + b * 2 + 1, safe)
        tl.store(OldTables + b * C + cs, table, cs < C)
        tl.store(Valid + b, 1)


@triton.jit
def _finish(Clock, Stamp):
    value = tl.load(Clock) + 1
    tl.store(Clock, value)
    tl.store(Stamp, value)


@triton.jit
def _pool(KV, Tables, Lens, Offsets, Reuse, Bounds,
          S: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
          P: tl.constexpr, NP: tl.constexpr, NB: tl.constexpr,
          POOL: tl.constexpr, TN: tl.constexpr, TD: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    b = row // S
    ds = tl.program_id(2) * TD + tl.arange(0, TD)
    reuse = tl.load(Reuse + b)
    if (block + 1) * POOL > reuse:
        length = tl.load(Lens + b).to(tl.int64)
        offset = tl.load(Offsets + row).to(tl.int64)
        end = tl.minimum(tl.maximum(length - offset, 0), C * P)
        mx = tl.full((TD,), -float('inf'), tl.float32)
        mn = tl.full((TD,), float('inf'), tl.float32)
        # Chunking bounds register use even for very large pooling blocks.
        for start in range(block * POOL, tl.minimum((block + 1) * POOL, end), TN):
            ts = start + tl.arange(0, TN)
            valid = (ts < end) & (ts < (block + 1) * POOL)
            page = tl.load(Tables + b * C + ts // P, valid, other=-1).to(tl.int64)
            valid = valid & (page >= 0) & (page < NP)
            base = page * P * (D + 4)
            raw = tl.load(KV + base[:, None] + (ts % P)[:, None] * D + ds[None, :],
                          valid[:, None] & (ds[None, :] < D), other=0)
            key = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scale_ptr = (KV + base + P * D).to(tl.pointer_type(tl.float32))
            scale = tl.load(scale_ptr + ts % P, valid, other=0.)
            key = key * scale[:, None]
            mx = tl.maximum(mx, tl.max(tl.where(valid[:, None], key, -float('inf')), 0))
            mn = tl.minimum(mn, tl.min(tl.where(valid[:, None], key, float('inf')), 0))
        empty = mx == -float('inf')
        out = (row * NB + block) * 2 * D + ds
        tl.store(Bounds + out, tl.where(empty, 0., mx), ds < D)
        tl.store(Bounds + out + D, tl.where(empty, 0., mn), ds < D)


@triton.jit
def _score(Q, W, Bounds, Lens, Offsets, Scores,
           S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
       NB: tl.constexpr, CAP: tl.constexpr, POOL: tl.constexpr,
           BH: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    hs = tl.program_id(1) * BH + tl.arange(0, BH)
    ds = tl.arange(0, BD)
    q = tl.load(Q + row * H * D + hs[:, None] * D + ds[None, :],
                (hs[:, None] < H) & (ds[None, :] < D), other=0.).to(tl.float32)
    pos, neg = tl.maximum(q, 0.), tl.maximum(-q, 0.)
    end = tl.minimum(tl.maximum(tl.load(Lens + row // S).to(tl.int64) -
                               tl.load(Offsets + row).to(tl.int64), 0), CAP)
    best = tl.full((BH,), 0., tl.float32)
    for block in range(tl.cdiv(end, POOL)):
        p = (row * NB + block) * 2 * D + ds
        mx = tl.load(Bounds + p, ds < D, other=0.)
        mn = tl.load(Bounds + p + D, ds < D, other=0.)
        score = tl.sum(neg * mx[None, :] - pos * mn[None, :], 1)
        best = tl.maximum(best, score)
    w = tl.load(W + row * H + hs, hs < H, other=0.).to(tl.float32)
    tl.store(Scores + row * H + hs, best * tl.abs(w), hs < H)


class _Entry:
    def __init__(self, b, s, c, nb, d, device):
        self.bounds = torch.empty((b * s, nb, 2, d), device=device, dtype=torch.float32)
        self.previous = torch.zeros((b, 2), device=device, dtype=torch.int64)
        self.tables = torch.zeros((b, c), device=device, dtype=torch.int64)
        self.valid = torch.zeros(b, device=device, dtype=torch.int32)
        self.stamp = torch.full((), -1, device=device, dtype=torch.int64)


# Strong KV references prevent allocator address reuse from aliasing old state.
# Batch sizes share one fixed-capacity allocation per (S,C,pool,D).
# Other shape buckets remain isolated; the clock prevents stale cross-bucket reuse.
_STATES = {}
_COLD_RESOURCES = {}
_COLD_WRITE_PREFIXES = set()


@triton.jit
def _written_pages(Slots, FirstWrite, N: tl.constexpr, STRIDE: tl.constexpr,
                   P: tl.constexpr, NP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + i * STRIDE, i < N, other=-1).to(tl.int64)
    page = slot // P
    valid = (i < N) & (slot >= 0) & (page < NP)
    tl.atomic_min(FirstWrite + page, (slot % P).to(tl.int32), valid,
                  sem='relaxed')


@triton.jit
def _invalidate_written_rows(FirstWrite, Previous, Tables, Valid, Clock, Stamp,
                             C: tl.constexpr, P: tl.constexpr,
                             NP: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    # Carry only the currently fresh bucket across this write event. A captured
    # invalidation cannot enumerate buckets created later; advancing the clock
    # makes those buckets stale without sacrificing known rows' append reuse.
    if b == 0:
        clock = tl.load(Clock)
        if tl.load(Stamp) == clock:
            tl.store(Stamp, clock + 1)
    if tl.load(Valid + b) != 0:
        safe = tl.load(Previous + b * 2 + 1)
        hit = tl.full((), False, tl.int1)
        for start in range(0, tl.minimum(tl.cdiv(safe, P), C), BLOCK):
            j = start + tl.arange(0, BLOCK)
            page = tl.load(Tables + b * C + j, j < C, other=-1)
            first = tl.load(FirstWrite + page,
                            (j < C) & (page >= 0) & (page < NP), other=P)
            visible = tl.minimum(tl.maximum(safe - j * P, 0), P)
            hit = hit | (tl.sum((first < visible).to(tl.int32), 0) != 0)
        if hit:
            tl.store(Valid + b, 0)


def invalidate_minmax_cache(cache_prefix, slot_mapping=None, kv_cache=None):
    """Invalidate after ordered KV writes, preserving graph-owned addresses.

    Without slots, invalidate the entire layer (or the selected KV resource).
    Slots are a CUDA int32/int64 vector: physical_page * page_size + offset;
    negative padding is ignored. Strided vectors are supported. Only rows whose
    cached common-visible prefix overlaps any write lose validity, including
    rows sharing a physical page with another sequence. Append/speculative
    suffix writes retain validity and carry fresh buckets across the clock tick.

    All decisions use GPU data and are replayable in CUDA graphs. The scratch
    page map is initialized on every invocation/replay, never persistent state.
    Use the writer/reader stream, or establish explicit stream ordering. As for
    scoring, warm the required resources/buckets before graph capture.
    """
    if slot_mapping is not None:
        if (not isinstance(slot_mapping, torch.Tensor) or not slot_mapping.is_cuda
                or slot_mapping.ndim != 1
                or slot_mapping.dtype not in (torch.int32, torch.int64)):
            raise ValueError('slot_mapping must be a CUDA int32/int64 vector')
        if kv_cache is not None and kv_cache.device != slot_mapping.device:
            raise ValueError('slot_mapping and kv_cache must share a CUDA device')
    if kv_cache is None and torch.cuda.is_current_stream_capturing():
        # Unknown future KV resources cannot be enumerated by this graph.
        # Existing states remain incremental; future resources stay uncached.
        _COLD_WRITE_PREFIXES.add(cache_prefix)
    if kv_cache is not None and torch.cuda.is_current_stream_capturing():
        resource = (cache_prefix, kv_cache.device, kv_cache.data_ptr(), tuple(kv_cache.shape))
        if resource not in _STATES:
            # A prefill-only graph may write KV without ever running scoring.
            # No clock can be advanced by that replay: quarantine before a
            # later eager decode has a chance to create reusable cache state.
            _COLD_RESOURCES[resource] = kv_cache
    for key, state in _STATES.items():
        if key[0] != cache_prefix:
            continue
        kv = state['kv']
        if kv_cache is not None and (kv.device != kv_cache.device
                or kv.data_ptr() != kv_cache.data_ptr() or kv.shape != kv_cache.shape):
            continue
        if slot_mapping is None:
            state['clock'].add_(1)
            for entry in state['entries'].values():
                entry.valid.zero_()
        elif kv.device == slot_mapping.device and slot_mapping.numel():
            np, p = kv.shape[:2]
            first_write = torch.full((np,), p, device=kv.device, dtype=torch.int32)
            _written_pages[(triton.cdiv(slot_mapping.numel(), 256),)](
                slot_mapping, first_write, slot_mapping.numel(), slot_mapping.stride(0),
                p, np, 256)
            for entry in state['entries'].values():
                _invalidate_written_rows[(entry.valid.numel(),)](
                    first_write, entry.previous, entry.tables, entry.valid,
                    state['clock'], entry.stamp, entry.tables.shape[1], p, np, 128)
            state['clock'].add_(1)


def _inputs(q, kv, w, lens, tables, pool, offsets):
    if q.ndim != 4 or q.dtype != torch.float8_e4m3fn:
        raise ValueError('q must be E4M3FN [B,S,H,D]')
    b, s, h, d = q.shape
    if min(s, h, d) <= 0 or not isinstance(pool, int) or pool <= 0:
        raise ValueError('positive S,H,D and integer pool size required')
    if (kv.ndim != 4 or kv.dtype != torch.uint8 or not kv.is_contiguous()
            or kv.shape[2:] != (1, d + 4) or kv.shape[1] <= 0):
        raise ValueError('contiguous uint8 KV [pages,page_size,1,D+4] required')
    if kv.shape[1] * d % 4 or kv.data_ptr() % 4:
        raise ValueError('FP32 scale segment must be four-byte aligned')
    if tables.ndim != 2 or tables.shape[0] != b or lens.shape != (b,):
        raise ValueError('lens [B], block_tables [B,C] required')
    if lens.dtype not in (torch.int32, torch.int64) or tables.dtype not in (torch.int32, torch.int64):
        raise ValueError('integer lengths/tables required')
    if w.shape == (b, h):
        w = w[:, None, :].expand(b, s, h)
    elif w.shape != (b, s, h):
        raise ValueError('weights [B,H] or [B,S,H] required')
    if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('FP16/BF16/FP32 weights required')
    if offsets is None:
        offsets = torch.arange(s - 1, -1, -1, device=q.device, dtype=torch.int64)
    if not isinstance(offsets, torch.Tensor):
        raise ValueError('offsets must be a CUDA integer tensor')
    if offsets.shape == (s,):
        offsets = offsets[None, :].expand(b, s)
    if offsets.shape != (b, s) or offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError('integer offsets [S] or [B,S] required')
    if not all(t.is_cuda and t.device == q.device for t in (q, kv, w, lens, tables, offsets)):
        raise ValueError('all inputs must share one CUDA device')
    return q.contiguous(), w.contiguous(), lens.contiguous(), tables.contiguous(), offsets.contiguous()


def _paged_minmax_head_scores(q_fp8, kv_cache_fp8, weights, context_lens,
                              block_tables, k_block_size, offsets=None,
                              cache_prefix=None):
    """Internal FP32 scores for independent reference validation."""
    q, w, lens, tables, off = _inputs(q_fp8, kv_cache_fp8, weights,
                                    context_lens, block_tables, k_block_size, offsets)
    b, s, h, d = q.shape
    c, p = tables.shape[1], kv_cache_fp8.shape[1]
    nb = triton.cdiv(c * p, k_block_size)
    scores = torch.empty((b, s, h), device=q.device, dtype=torch.float32)
    if b == 0 or c == 0:
        return scores.zero_()
    enabled = cache_prefix is not None
    resource = (cache_prefix, q.device, kv_cache_fp8.data_ptr(), tuple(kv_cache_fp8.shape))
    state = _STATES.get(resource)
    if state is None and cache_prefix in _COLD_WRITE_PREFIXES:
        _COLD_RESOURCES[resource] = kv_cache_fp8
    signature = (s, c, k_block_size, d)
    # Never capture persistent initialization: its replay could reset the clock
    # and accidentally resurrect a stale bucket's matching stamp. Cold capture
    # uses full recomputation forever; eager warmup enables incremental graphs.
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and state is None:
        # No persistent clock exists for this graph to advance on replay.
        # Permanently quarantine this resource from caching, even if the env
        # is later enabled. Keep KV alive to prevent pointer-address aliasing.
        _COLD_RESOURCES[resource] = kv_cache_fp8
    enabled = enabled and resource not in _COLD_RESOURCES
    if enabled:
        max_batch = (state['max_batch'] if state is not None else
                     int(os.environ.get('ResMISA_POOL_MAX_BATCH', '256')))
        if max_batch <= 0 or b > max_batch:
            raise ValueError(
                f'minmax cache batch {b} exceeds capacity {max_batch}; '
                'set ResMISA_POOL_MAX_BATCH to the largest runtime/capture batch '
                'before warmup (default 256); cached buffers cannot be resized')
    if enabled and capturing:
        enabled = state is not None and signature in state['entries']
    if state is None and enabled:
        state = {'kv': kv_cache_fp8, 'clock': torch.zeros((), device=q.device, dtype=torch.int64),
                 'entries': {}, 'max_batch': max_batch}
        _STATES[resource] = state
    reuse = torch.zeros(b, device=q.device, dtype=torch.int64)
    if enabled:
        signature = (s, c, k_block_size, d)
        entry = state['entries'].get(signature)
        if entry is None:
            entry = _Entry(state['max_batch'], s, c, nb, d, q.device)
            state['entries'][signature] = entry
        bounds = entry.bounds
        _prepare[(entry.valid.numel(),)](lens, tables, off, entry.previous, entry.tables, entry.valid,
                       state['clock'], entry.stamp, reuse, b, s, c, c * p,
                       triton.next_power_of_2(c), triton.next_power_of_2(s))
    else:
        bounds = torch.empty((b * s, nb, 2, d), device=q.device, dtype=torch.float32)
        if state is not None:
            state['clock'].add_(1)
    _pool[(b * s, nb, triton.cdiv(d, 32))](
        kv_cache_fp8, tables, lens, off, reuse, bounds, s, d, c, p,
        kv_cache_fp8.shape[0], nb, k_block_size, 32, 32, enable_fp_fusion=False)
    if enabled:
        _finish[(1,)](state['clock'], entry.stamp)
    _score[(b * s, triton.cdiv(h, 8))](
        q, w, bounds, lens, off, scores, s, h, d, nb, c * p,
        k_block_size, 8, triton.next_power_of_2(d), enable_fp_fusion=False)
    return scores


def paged_minmax_head_indices(q_fp8, kv_cache_fp8, weights, context_lens,
                              block_tables, k_block_size, topk_heads,
                              offsets=None, cache_prefix=None):
    if q_fp8.ndim != 4 or not isinstance(topk_heads, int):
        raise ValueError('q [B,S,H,D] and integer topk_heads required')
    if not 0 <= topk_heads <= q_fp8.shape[2]:
        raise ValueError('topk_heads must be between zero and H')
    scores = _paged_minmax_head_scores(q_fp8, kv_cache_fp8, weights,
                                      context_lens, block_tables, k_block_size,
                   offsets, cache_prefix)
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :topk_heads]
    return indices[:, 0] if q_fp8.shape[1] == 1 else indices


def reset_cache():
    """Invalidate all minmax state, without dropping any graph references."""
    for state in _STATES.values():
        state['clock'].add_(1)
        for entry in state['entries'].values():
            entry.valid.zero_()


