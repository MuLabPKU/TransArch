"""Head reduction and FP8 prefill scoring.

The weighted head sum is quantized as one component or as a high component
plus a separately scaled residual. Q and K retain their input quantization;
dot products and scales accumulate in FP32.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ...api import validate_prepared as _validate_prepared, validate_ranges as _validate_ranges


@dataclass(frozen=True)
class ScanConfig:
    block_q: int = 64
    block_n: int = 128
    num_warps: int = 4
    accumulation: int = 32

    def __post_init__(self):
        # M=32 can lower to synchronous FP16 MMA; use M>=64 for native FP8.
        allowed = ((64, 128), (64, 128), (4, 8), (-1, 32, 64, 128))
        for value, choices in zip(vars(self).values(), allowed):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value not in choices
            ):
                raise ValueError("unsupported native FP8 scan configuration")
        # The 128x256/8-warp FP8x2 configuration failed memory checks.
        # Restrict dispatch to the supported tile shapes below.
        if (self.block_q, self.block_n, self.num_warps) not in (
            (64, 64, 4),
            (64, 128, 4),
            (128, 128, 8),
        ):
            raise ValueError("unsupported native FP8 tile family")


@dataclass(frozen=True)
class QuantizedG:
    high: torch.Tensor
    high_scale: torch.Tensor
    low: torch.Tensor
    low_scale: torch.Tensor
    components: int


def _check_components(components):
    if (
        isinstance(components, bool)
        or not isinstance(components, int)
        or components not in (1, 2)
    ):
        raise ValueError("components must be 1 or 2")


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
def _collapse_h32(Q, W, Hi, HS, Lo, LS, COMPONENTS: tl.constexpr):
    row = tl.program_id(0)
    h, d = tl.arange(0, 32), tl.arange(0, 128)
    q = tl.load(Q + row * 32 * 128 + h[:, None] * 128 + d[None, :]).to(tl.float32)
    w = tl.load(W + row * 32 + h).to(tl.float32)
    g = tl.sum(q * w[:, None], axis=0)
    hi, hs = _quantize_row(g)
    tl.store(Hi + row * 128 + d, hi)
    tl.store(HS + row, hs)
    if COMPONENTS == 2:
        low, ls = _quantize_row(g - hi.to(tl.float32) * hs)
        tl.store(Lo + row * 128 + d, low)
        tl.store(LS + row, ls)


@triton.jit
def _collapse_kernel(Q, W, Hi, HS, Lo, LS, COMPONENTS: tl.constexpr):
    row = tl.program_id(0)
    h, d = tl.arange(0, 64), tl.arange(0, 128)
    q = tl.load(Q + row * 64 * 128 + h[:, None] * 128 + d[None, :]).to(tl.float32)
    w = tl.load(W + row * 64 + h).to(tl.float32)
    g = tl.sum(q * w[:, None], axis=0)
    hi, hs = _quantize_row(g)
    tl.store(Hi + row * 128 + d, hi)
    tl.store(HS + row, hs)
    if COMPONENTS == 2:
        low, ls = _quantize_row(g - hi.to(tl.float32) * hs)
        tl.store(Lo + row * 128 + d, low)
        tl.store(LS + row, ls)


@triton.jit
def _dot(a, b, ACC: tl.constexpr):
    if ACC < 0:
        return tl.dot(a, b, out_dtype=tl.float32)
    return tl.dot(a, b, out_dtype=tl.float32, max_num_imprecise_acc=ACC)


@triton.jit
def _scan_kernel(
    Hi,
    HS,
    Lo,
    LS,
    K,
    KS,
    Starts,
    Ends,
    Out,
    rows,
    n,
    COMPONENTS: tl.constexpr,
    BQ: tl.constexpr,
    BN: tl.constexpr,
    ACC: tl.constexpr,
):
    qi = tl.program_id(1) * BQ + tl.arange(0, BQ)
    ki = tl.program_id(0) * BN + tl.arange(0, BN)
    d = tl.arange(0, 128)
    gh = tl.load(Hi + qi[:, None] * 128 + d[None, :], qi[:, None] < rows, 0.0)
    kt = tl.load(K + ki[None, :] * 128 + d[:, None], ki[None, :] < n, 0.0)
    hs = tl.load(HS + qi, qi < rows, 0)
    score = _dot(gh, kt, ACC) * hs[:, None]
    if COMPONENTS == 2:
        gl = tl.load(Lo + qi[:, None] * 128 + d[None, :], qi[:, None] < rows, 0.0)
        ls = tl.load(LS + qi, qi < rows, 0)
        score = score + _dot(gl, kt, ACC) * ls[:, None]
    ks = tl.load(KS + ki, ki < n, 0)
    score = score * ks[None, :]
    start = tl.load(Starts + qi, qi < rows, 0)
    end = tl.load(Ends + qi, qi < rows, 0)
    valid = (ki[None, :] >= start[:, None]) & (ki[None, :] < end[:, None])
    score = tl.where(valid, score, float("-inf"))
    # Write every real output, including invalid/empty ranges; no separate fill.
    tl.store(
        Out + qi[:, None].to(tl.int64) * n + ki[None, :],
        score,
        (qi[:, None] < rows) & (ki[None, :] < n),
    )


def collapse(prepared, *, components=1):
    """Reduce heads in FP32 and quantize the result in one kernel.

    Each component has a per-query power-of-two scale. Reduction order
    depends on the compiler layout.
    """
    _validate_prepared(prepared)
    _check_components(components)
    rows, device = prepared.q.shape[0], prepared.q.device
    high = torch.empty((rows, 128), dtype=torch.float8_e4m3fn, device=device)
    hs = torch.empty(rows, dtype=torch.float32, device=device)
    low = torch.empty_like(high) if components == 2 else high
    ls = torch.empty_like(hs) if components == 2 else hs
    if rows:
        with torch.cuda.device(device):
            kernel = _collapse_h32 if prepared.q.shape[1] == 32 else _collapse_kernel
            kernel[(rows,)](
                prepared.q,
                prepared.effective_weights,
                high,
                hs,
                low,
                ls,
                COMPONENTS=components,
                num_warps=4,
            )
    return QuantizedG(high, hs, low, ls, components)


def scan_prepared(
    g, prepared, starts, ends, *, config=ScanConfig(), out=None, return_kernel=False
):
    """Compute FP8 dot products, apply FP32 scales, and mask causal ranges.

    K scales apply after the dot product. Inputs and intermediate results
    must be finite, scales positive, and 0 <= starts <= ends <= N. Shape
    checks do not synchronize or inspect tensor contents.
    """
    _validate_prepared(prepared)
    _validate_ranges(prepared, starts, ends)
    if not isinstance(config, ScanConfig) or not isinstance(g, QuantizedG):
        raise TypeError("ScanConfig and QuantizedG required")
    _check_components(g.components)
    rows, n = prepared.q.shape[0], len(prepared.keys)
    for x, shape, dtype in (
        (g.high, (rows, 128), torch.float8_e4m3fn),
        (g.low, (rows, 128), torch.float8_e4m3fn),
        (g.high_scale, (rows,), torch.float32),
        (g.low_scale, (rows,), torch.float32),
    ):
        if (
            x.shape != shape
            or x.dtype != dtype
            or x.device != prepared.q.device
            or not x.is_contiguous()
        ):
            raise ValueError("invalid quantized-g metadata")
    if out is None:
        out = torch.empty((rows, n), device=prepared.q.device, dtype=torch.float32)
    elif (
        out.shape != (rows, n)
        or out.dtype != torch.float32
        or out.device != prepared.q.device
        or not out.is_contiguous()
    ):
        raise ValueError("contiguous FP32 output[rows,N] on input device required")
    compiled = None
    if rows and n:
        with torch.cuda.device(prepared.q.device):
            compiled = _scan_kernel[
                (triton.cdiv(n, config.block_n), triton.cdiv(rows, config.block_q))
            ](
                g.high,
                g.high_scale,
                g.low,
                g.low_scale,
                prepared.keys,
                prepared.key_scales,
                starts,
                ends,
                out,
                rows,
                n,
                COMPONENTS=g.components,
                BQ=config.block_q,
                BN=config.block_n,
                ACC=config.accumulation,
                num_warps=config.num_warps,
                enable_fp_fusion=False,
            )
    return (out, compiled) if return_kernel else out


def stage1(prepared, starts, ends, *, components=1, config=ScanConfig(), out=None):
    _validate_prepared(prepared)
    _validate_ranges(prepared, starts, ends)
    g = collapse(prepared, components=components)
    return scan_prepared(g, prepared, starts, ends, config=config, out=out)



@triton.jit
def pack(Hi,Lo,HS,LS,Q2,W4,rows,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B)
    row=x//128;col=x%128
    hi=tl.load(Hi+x,row<rows,0.);lo=tl.load(Lo+x,row<rows,0.)
    tl.store(Q2+row*256+col,hi,row<rows)
    tl.store(Q2+row*256+128+col,lo,row<rows)
    hs=tl.load(HS+row,(row<rows)&(col==0),0)
    ls=tl.load(LS+row,(row<rows)&(col==0),0)
    tl.store(W4+row*4,hs,(row<rows)&(col==0))
    tl.store(W4+row*4+1,ls,(row<rows)&(col==0))
    tl.store(W4+row*4+2,0.,(row<rows)&(col==0))
    tl.store(W4+row*4+3,0.,(row<rows)&(col==0))

