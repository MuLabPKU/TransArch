"""LISA threshold score preparation and reranking."""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl

from ...api import PreparedFP8

@dataclass(frozen=True)
class QuantizedG:
    high: torch.Tensor
    high_scale: torch.Tensor
    low: torch.Tensor
    low_scale: torch.Tensor
    components: int


@triton.jit
def _quantize_row(values):
    # Quantize with a per-row power-of-two scale.
    # A floor makes the all-zero row well-defined; scaling/rounding is explicit.
    maximum = tl.maximum(tl.max(tl.abs(values), 0), 1.0e-12)
    scale = tl.exp2(tl.ceil(tl.log2(maximum / 448.0)))
    quantized = tl.minimum(tl.maximum(values / scale, -448.0), 448.0).to(
        tl.float8e4nv, fp_downcast_rounding="rtne"
    )
    return quantized, scale


@triton.jit
def _dot(a, b, ACC: tl.constexpr):
    if ACC < 0:
        return tl.dot(a, b, out_dtype=tl.float32)
    return tl.dot(a, b, out_dtype=tl.float32, max_num_imprecise_acc=ACC)

def _validate_ranges(prepared, starts, ends):
    if any(
        x.shape != (prepared.q.shape[0],)
        or x.dtype != torch.int32
        or not x.is_contiguous()
        or x.device != prepared.q.device
        for x in (starts, ends)
    ):
        raise ValueError("contiguous int32 ranges[rows] on the input device required")


UNDERFLOW, OVERFLOW, BAD_PILOT, BAD_RANGE = 1, 2, 4, 8
_FINISH_LOADED = False

@dataclass(frozen=True)
class Budget:
    minimum: int = 2048
    target: int = 8192
    capacity: int = 16384
    samples: int = 2048

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, int) for x in vars(self).values()):
            raise ValueError("budget parameters must be integers")
        if not 0 < self.minimum <= self.target <= self.capacity <= 1048576:
            raise ValueError("require 0 < minimum <= target <= capacity <= 1048576")
        if self.samples not in (256, 512, 1024, 2048, 4096):
            raise ValueError("samples must be a power of two in [256,4096]")


@dataclass
class Candidates:
    ids: torch.Tensor
    counts: torch.Tensor  # True count, including records that exceeded capacity.
    status: torch.Tensor
    thresholds: torch.Tensor
    pilot_scores: torch.Tensor | None
    coarse_scores: torch.Tensor | None
    budget: Budget
    threshold_keys: torch.Tensor | None = None  # Ordered score + token tie break, recovery only.


@dataclass
class Selection:
    candidates: Candidates
    logits: torch.Tensor
    indices: torch.Tensor


@triton.jit
def _score_tile(Hi, HS, Lo, LS, K, KS, qi, ki, qmask, kmask, ACC: tl.constexpr):
    d = tl.arange(0, 128)
    high = tl.load(Hi + qi[:, None] * 128 + d[None, :], qmask[:, None], 0.0)
    low = tl.load(Lo + qi[:, None] * 128 + d[None, :], qmask[:, None], 0.0)
    kt = tl.load(K + ki[None, :] * 128 + d[:, None], kmask[None, :], 0.0)
    hs = tl.load(HS + qi, qmask, 0.0)
    ls = tl.load(LS + qi, qmask, 0.0)
    ks = tl.load(KS + ki, kmask, 0.0)
    score = _dot(high, kt, ACC) * hs[:, None]
    score = score + _dot(low, kt, ACC) * ls[:, None]
    return score * ks[None, :]


@triton.jit
def _pilot_kernel(Hi, HS, Lo, LS, K, KS, Starts, Ends, Samples,
                  rows, n, S: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr,
                  ACC: tl.constexpr):
    qi = tl.program_id(1) * BQ + tl.arange(0, BQ)
    si = tl.program_id(0) * BN + tl.arange(0, BN)
    start = tl.load(Starts + qi, qi < rows, n)
    end = tl.load(Ends + qi, qi < rows, 0)
    lower = tl.minimum(tl.maximum(tl.min(start, 0), 0), n)
    upper = tl.minimum(tl.maximum(tl.max(end, 0), lower), n)
    span = upper - lower
    size = tl.minimum(span, S)
    # One unique stratified sample per bin. Shared within a query tile so the
    # pilot uses the same native FP8 tensor-core math as the full scan.
    ki = lower + ((2 * si.to(tl.int64) + 1) * span // (2 * tl.maximum(size, 1)))
    kmask = (si < size) & (ki >= 0) & (ki < n)
    score = _score_tile(Hi, HS, Lo, LS, K, KS, qi, ki, qi < rows, kmask, ACC)
    valid = kmask[None, :] & (ki[None, :] >= start[:, None]) & (ki[None, :] < end[:, None])
    score = tl.where(valid, score, float("-inf"))
    tl.store(Samples + qi[:, None].to(tl.int64) * S + si[None, :], score,
             (qi[:, None] < rows) & (si[None, :] < S))


@triton.jit
def _threshold_kernel(Samples, Starts, Ends, Tau, Status, n,
                      S: tl.constexpr, TARGET: tl.constexpr, HISTOGRAM: tl.constexpr = True):
    row = tl.program_id(0)
    i = tl.arange(0, S)
    values = tl.load(Samples + row.to(tl.int64) * S + i)
    valid = (values > float("-inf")) & (values < float("inf"))
    count = tl.sum(valid.to(tl.int32), 0)
    start, end = tl.load(Starts + row), tl.load(Ends + row)
    length = tl.maximum(end - start, 0)
    rank = (count.to(tl.int64) * tl.minimum(TARGET, length) + length // 2) // tl.maximum(length, 1)
    rank = tl.maximum(1, tl.minimum(rank, count))
    if HISTOGRAM:
        # We only need a budget interval, not an exact sample order statistic.
        # Locate a coarse bin, then use its actual minimum sample score. This
        # includes the rank boundary without rounding the original scan scores.
        lo = tl.min(tl.where(valid, values, float("inf")), 0)
        hi = tl.max(tl.where(valid, values, float("-inf")), 0)
        width = tl.maximum(hi - lo, 1.0e-20)
        bucket = tl.minimum(tl.maximum(((values - lo) / width * 256).to(tl.int32), 0), 255)
        histogram = tl.histogram(bucket, 256, mask=valid)
        tail = tl.cumsum(histogram, 0, reverse=True)
        boundary = tl.max(tl.where(tail >= rank, tl.arange(0, 256), 0), 0)
        tau = tl.min(tl.where(valid & (bucket == boundary), values, float("inf")), 0)
    else:
        # Sorting provides a reference for the histogram threshold.
        ordered = tl.sort(tl.where(valid, values, float("-inf")), descending=True)
        tau = tl.sum(tl.where(i == rank - 1, ordered, 0.0), 0)
    tau = tl.where(length <= TARGET, float("-inf"), tau)
    bad_sample = ((count == 0) & (length > TARGET)) | (tl.sum((values != values).to(tl.int32), 0) > 0)
    bad_range = (start < 0) | (end < start) | (end > n)
    tau = tl.where(bad_sample | bad_range, float("inf"), tau)
    tl.store(Tau + row, tau)
    tl.store(Status + row, bad_sample.to(tl.int32) * 4 | bad_range.to(tl.int32) * 8)


@triton.jit
def _scan_compact_kernel(Hi, HS, Lo, LS, K, KS, Starts, Ends, Tau,
                         IDs, Counts, DebugScores, rows, n,
                         CAP: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr,
                         ACC: tl.constexpr, DEBUG: tl.constexpr,
                         RESERVE: tl.constexpr = 128,
                         ThresholdKeys=None, KEY_THRESHOLD: tl.constexpr = False,
                         Q_FIRST: tl.constexpr = False):
    query_tile = tl.program_id(0) if Q_FIRST else tl.program_id(1)
    key_tile = tl.program_id(1) if Q_FIRST else tl.program_id(0)
    qi = query_tile * BQ + tl.arange(0, BQ)
    ki = key_tile * BN + tl.arange(0, BN)
    score = _score_tile(Hi, HS, Lo, LS, K, KS, qi, ki, qi < rows, ki < n, ACC)
    start = tl.load(Starts + qi, qi < rows, 0)
    end = tl.load(Ends + qi, qi < rows, 0)
    tau = tl.load(Tau + qi, qi < rows, float("inf"))
    if KEY_THRESHOLD:
        # Finite FP32 numerical order, with both signed zeros canonicalized.
        # Token IDs disambiguate a threshold that falls inside a large tie.
        bits = tl.where(score == 0.0, 0, score.to(tl.uint32, bitcast=True))
        code = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
        keys = (code.to(tl.uint64) << 32) | (n - ki[None, :]).to(tl.uint32).to(tl.uint64)
        pivot = tl.load(ThresholdKeys + qi, qi < rows, 0).to(tl.uint64)
        passes = keys >= pivot[:, None]
    else:
        passes = score >= tau[:, None]
    keep = ((qi[:, None] < rows) & (ki[None, :] < n) &
            (ki[None, :] >= start[:, None]) & (ki[None, :] < end[:, None]) &
            passes)
    selected = keep.to(tl.int32)
    offsets = tl.cumsum(selected, 1) - 1
    count = tl.sum(selected, 1)
    # On Triton 3.4/SM100 a 64-element atomic result over 128 threads was
    # replicated before convert_layout: inactive replicas overwrote the live
    # return values in shared memory. Use at least one distinct logical element
    # per physical thread, then gather only the actual BQ reservation results.
    tl.static_assert(RESERVE >= BQ)
    reservation_lane = tl.arange(0, RESERVE)
    reservation_row = query_tile * BQ + reservation_lane
    reservation_count = tl.gather(count, tl.minimum(reservation_lane, BQ - 1), axis=0)
    reservation_valid = (reservation_lane < BQ) & (reservation_row < rows) & (reservation_count > 0)
    reservation_base = tl.atomic_add(Counts + reservation_row, reservation_count,
                                     reservation_valid, sem="relaxed")
    base = tl.gather(reservation_base, tl.arange(0, BQ), axis=0)
    slot = base[:, None] + offsets
    write = keep & (slot < CAP)
    address = qi[:, None].to(tl.int64) * CAP + slot.to(tl.int64)
    tl.store(IDs + address, ki[None, :], write)
    if DEBUG:
        tl.store(DebugScores + address, score, write)


def load_finish():
    global _FINISH_LOADED
    if not _FINISH_LOADED:
        from ...build import extension
        extension("threshold_finish")
        _FINISH_LOADED = True


def finish(selected, logits, starts, *, k=2048, cached=False):
    if isinstance(k, bool) or not isinstance(k, int) or not 0 < k <= min(8192, selected.budget.minimum):
        raise ValueError("require 0 < final K <= minimum budget and <= 8192")
    rows = selected.ids.shape[0]
    output = torch.empty((rows, k), dtype=torch.int32, device=logits.device)
    if rows:
        load_finish()
        torch.ops.relu_hisa_counted_finish.run(
            logits, selected.ids, selected.counts, selected.status, starts, output, cached)
    return output

@triton.jit
def _rerank_counted_kernel(Q, K, W, KS, IDs, Counts, Status, Starts, Ends, Out,
                           n, CAP: tl.constexpr, BN: tl.constexpr, SPLITS: tl.constexpr = 1):
    row = tl.program_id(0)
    count = tl.minimum(tl.load(Counts + row), CAP)
    status = tl.load(Status + row)
    if status == 0:
        h, d = tl.arange(0, 64), tl.arange(0, 128)
        qt = tl.load(Q + row * 64 * 128 + h[None, :] * 128 + d[:, None])
        w = tl.load(W + row * 64 + h)
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
            # Accumulate all heads in FP32 over 128-key tiles.
            dot = tl.dot(kt, qt, out_dtype=tl.float32, max_num_imprecise_acc=32)
            value = tl.sum(tl.maximum(dot * ks[:, None], 0.0) * w[None, :], 1)
            value = tl.where(valid & (ids >= start) & (ids < end), value, float("-inf"))
            tl.store(Out + row.to(tl.int64) * CAP + col, value, col < count)

