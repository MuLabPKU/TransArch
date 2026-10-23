"""Exact min/max pooling; scan only required query-boundary blocks."""
import torch
import triton
import triton.language as tl

@triton.jit
def _max_pair(a,b):return tl.maximum(a,b)

@triton.jit
def _min_pair(a,b):return tl.minimum(a,b)

@triton.jit(do_not_specialize=['N'])
def _block_extrema(K,Scale,Bounds,N,D:tl.constexpr,POOL:tl.constexpr,TN:tl.constexpr,TD:tl.constexpr):
    block=tl.program_id(0)
    ds=tl.program_id(1)*TD+tl.arange(0,TD)
    ts=block*POOL+tl.arange(0,TN)
    valid=(ts<N)&(tl.arange(0,TN)<POOL)
    key=tl.load(K+ts[:,None]*D+ds[None,:],valid[:,None]&(ds[None,:]<D),other=0.).to(tl.float32)
    scale=tl.load(Scale+ts,valid,other=0.).to(tl.float32)
    key=key*scale[:,None]
    mx=tl.max(tl.where(valid[:,None],key,-float('inf')),0)
    mn=tl.min(tl.where(valid[:,None],key,float('inf')),0)
    tl.store(Bounds+block*3*D+ds,mx,ds<D)
    tl.store(Bounds+block*3*D+D+ds,mn,ds<D)

@triton.jit(do_not_specialize=['N'])
def _prefix_compact(K,Scale,Prefix,Suffix,Rows,Count,N,D:tl.constexpr,POOL:tl.constexpr,
                    TN:tl.constexpr,TD:tl.constexpr,GROUPS:tl.constexpr):
    ds=tl.program_id(1)*TD+tl.arange(0,TD)
    count=tl.load(Count)
    for slot in range(tl.program_id(0),count,GROUPS):
        block=tl.load(Rows+slot)
        ts=block*POOL+tl.arange(0,TN)
        valid=(ts<N)&(tl.arange(0,TN)<POOL)
        key=tl.load(K+ts[:,None]*D+ds[None,:],valid[:,None]&(ds[None,:]<D),other=0.).to(tl.float32)
        scale=tl.load(Scale+ts,valid,other=0.)
        key=key*scale[:,None]
        mx=tl.where(valid[:,None],key,-float('inf'))
        mn=tl.where(valid[:,None],key,float('inf'))
        pmx=tl.associative_scan(mx,0,_max_pair)
        pmn=tl.associative_scan(mn,0,_min_pair)
        smx=tl.associative_scan(mx,0,_max_pair,reverse=True)
        smn=tl.associative_scan(mn,0,_min_pair,reverse=True)
        offsets=ts[:,None]*2*D+ds[None,:]
        mask=valid[:,None]&(ds[None,:]<D)
        tl.store(Prefix+offsets,pmx,mask)
        tl.store(Prefix+offsets+D,pmn,mask)
        tl.store(Suffix+offsets,smx,mask)
        tl.store(Suffix+offsets+D,smn,mask)

@triton.jit(do_not_specialize=['N','NB'])
def _internal_intervals(K,Scale,Starts,Ends,Bounds,N,D:tl.constexpr,NB,
                        POOL:tl.constexpr,TN:tl.constexpr,TD:tl.constexpr):
    row=tl.program_id(0)
    start,end=tl.load(Starts+row),tl.load(Ends+row)
    first,last=start//POOL,tl.maximum(end-1,0)//POOL
    used=((start<end)&(first==last)&(start%POOL!=0)
          &(end<tl.minimum((first+1)*POOL,N)))
    if used:
        ts=first*POOL+tl.arange(0,TN)
        valid=(ts>=start)&(ts<end)&(ts<N)
        scale=tl.load(Scale+ts,valid,other=0.)
        for group in range(tl.cdiv(D,TD)):
            ds=group*TD+tl.arange(0,TD)
            key=tl.load(K+ts[:,None]*D+ds[None,:],valid[:,None]&(ds[None,:]<D),other=0.).to(tl.float32)
            key=key*scale[:,None]
            mx=tl.max(tl.where(valid[:,None],key,-float('inf')),0)
            mn=tl.min(tl.where(valid[:,None],key,float('inf')),0)
            p=NB+row*2
            tl.store(Bounds+p*3*D+ds,mx,ds<D)
            tl.store(Bounds+p*3*D+D+ds,mn,ds<D)

@triton.jit(do_not_specialize=['M','NB'])
def _boundary_list(Starts,Ends,Flags,Rows,Count,M,NB,POOL:tl.constexpr,
                   BM:tl.constexpr,BB:tl.constexpr):
    r=tl.arange(0,BM)
    start=tl.load(Starts+r,r<M,other=0)
    end=tl.load(Ends+r,r<M,other=0)
    first,last=start//POOL,tl.maximum(end-1,0)//POOL
    valid=(r<M)&(start<end)
    need_first=valid&((start%POOL!=0)|(first==last))
    need_last=valid&(end%POOL!=0)&(first!=last)
    histogram=tl.histogram(first,BB,mask=need_first)+tl.histogram(last,BB,mask=need_last)
    b=tl.arange(0,BB)
    active=(histogram>0)&(b<NB)
    position=tl.cumsum(active.to(tl.int32),0)-1
    tl.store(Rows+position,b,active)
    tl.store(Count,tl.sum(active.to(tl.int32),0))
    tl.store(Flags+b,active.to(tl.int32),b<NB)

def prefix_extrema(key, scales, starts, ends, prefix, suffix, bounds,
                   n, d, pool, tn, td, **kwargs):
    """Run the accepted selective prefix/suffix pooling launch sequence."""
    starts = starts.reshape(-1).int().contiguous()
    ends = ends.reshape(-1).int().contiguous()
    m, nb = starts.numel(), triton.cdiv(n, pool)
    assert m <= 8192 and nb <= 2048
    flags = torch.empty(nb, device=key.device, dtype=torch.int32)
    rows = torch.empty_like(flags)
    count = torch.empty((), device=key.device, dtype=torch.int32)
    _boundary_list[(1,)](
        starts, ends, flags, rows, count, m, nb, pool,
        triton.next_power_of_2(m), triton.next_power_of_2(nb),
        num_warps=8 if m > 2048 else 4)
    _block_extrema[(nb, triton.cdiv(d, 16))](
        key, scales, bounds, n, d, pool, tn, 16, num_warps=8)
    groups = min(nb, 32)
    return _prefix_compact[(groups, triton.cdiv(d, td))](
        key, scales, prefix, suffix, rows, count, n, d, pool, tn, td,
        groups, **kwargs)


def boundary_extrema(key, scales, starts, ends, bounds,
                     n, d, nb, pool, tn, td, **kwargs):
    """Exactly reduce same-block intervals whose two ends are both partial."""
    return _internal_intervals[(starts.numel(),)](
        key, scales, starts, ends, bounds, n, d, nb, pool, tn, td, **kwargs)

