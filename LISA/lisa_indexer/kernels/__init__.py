import triton
import triton.language as tl


@triton.jit
def causal_ranges(Starts, Ends, ROWS: tl.constexpr, N: tl.constexpr,
                  PREFIX: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0) * B + tl.arange(0, B)
    start = tl.load(Starts + r, r < ROWS, 0)
    end = tl.load(Ends + r, r < ROWS, 0)
    tl.device_assert((r >= ROWS) | ((start == 0) & (end == PREFIX + r + 1) & (end <= N)),
                     "LISA requires single-request causal, request-relative ranges")


@triton.jit
def linear_status(Status, Counts, ROWS: tl.constexpr, CAP: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0) * B + tl.arange(0, B)
    status = tl.load(Status + r, r < ROWS, 0)
    count = tl.load(Counts + r, r < ROWS, 2048)
    tl.device_assert((r >= ROWS) | ((status == 0) & (count >= 2048) & (count <= CAP)),
                     "LISA HOT selector failed: inspect last_stats status/counts")


@triton.jit
def threshold_status(Status, Counts, Ends, ROWS: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0) * B + tl.arange(0, B)
    status = tl.load(Status + r, r < ROWS, 0)
    count = tl.load(Counts + r, r < ROWS, 0)
    end = tl.load(Ends + r, r < ROWS, 0)
    tl.device_assert((r >= ROWS) | ((status == 0) & (count >= tl.minimum(end, 2048))
                     & (count <= tl.minimum(end, 16384))), "LISA threshold recovery failed")
