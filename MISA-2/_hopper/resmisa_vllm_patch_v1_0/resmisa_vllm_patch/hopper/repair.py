"""Exact uploaded FP32 reduction for short intervals and near ties."""
import torch
import triton
import triton.language as tl

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

@triton.jit(do_not_specialize=['NB'])
def _exact(Q,W,Bounds,Starts,Ends,Scores,Flags,
           H:tl.constexpr,D:tl.constexpr,BH:tl.constexpr,BD:tl.constexpr,NB,POOL:tl.constexpr):
    row=tl.program_id(0)
    if tl.load(Flags+row)!=0:
        hs,ds=tl.arange(0,BH),tl.arange(0,BD)
        q=tl.load(Q+row*H*D+hs[:,None]*D+ds[None,:],
                  (hs[:,None]<H)&(ds[None,:]<D),other=0.).to(tl.float32)
        pos,neg=tl.maximum(q,0.),tl.maximum(-q,0.)
        start,end=tl.load(Starts+row),tl.load(Ends+row)
        first,last=start//POOL,tl.maximum(end-1,0)//POOL
        best=tl.full((BH,),0.,tl.float32)
        for block in range(first,tl.cdiv(end,POOL)):
            p=block
            p=tl.where((block==last)&(end%POOL!=0),NB+row*2+1,p)
            p=tl.where((block==first)&((start%POOL!=0)|(first==last)),NB+row*2,p)
            mx=tl.load(Bounds+p*3*D+ds,ds<D,other=0.)
            mn=tl.load(Bounds+p*3*D+D+ds,ds<D,other=0.)
            upper=tl.sum(neg*mx[None,:]-pos*mn[None,:],1)
            best=tl.maximum(best,upper)
        w=tl.load(W+row*H+hs,hs<H,other=0.).to(tl.float32)
        tl.store(Scores+row*H+hs,best*tl.abs(w),hs<H)

@triton.jit
def _mark(Scores,Starts,Ends,Flags,H:tl.constexpr,POOL:tl.constexpr,
          EPS:tl.constexpr,ABS_EPS:tl.constexpr):
    row=tl.program_id(0)
    hs=tl.arange(0,H)
    value=tl.load(Scores+row*H+hs)
    ordered=tl.sort(value,descending=True)
    adjacent=tl.gather(ordered,tl.minimum(hs+1,H-1),axis=0)
    close=(hs<7)&((ordered-adjacent)<=ABS_EPS+EPS*tl.maximum(tl.abs(ordered),tl.abs(adjacent)))
    start,end=tl.load(Starts+row),tl.load(Ends+row)
    needed=((end-start)<=POOL)|(tl.sum(close.to(tl.int32),0)>0)
    needed=needed|(tl.sum((tl.abs(value)<float('inf')).to(tl.int32),0)<H)
    tl.store(Flags+row,needed.to(tl.int32))

def repair_small(q,w,bounds,starts,ends,scores,nb,pool,relative=1.e-5,absolute=1.e-6):
    m,h,d=q.shape
    flags=torch.empty(m,dtype=torch.int32,device=q.device)
    _mark[(m,)](scores,starts,ends,flags,h,pool,relative,absolute,num_warps=4)
    _exact[(m,)](q,w,bounds,starts,ends,scores,flags,h,d,
        triton.next_power_of_2(h),triton.next_power_of_2(d),nb,pool,
        num_warps=8,enable_fp_fusion=False)
    return flags

@triton.jit(do_not_specialize=['M'])
def _compact(Flags,Rows,Count,M,B:tl.constexpr):
    r=tl.arange(0,B)
    f=tl.load(Flags+r,r<M,other=0)
    position=tl.cumsum(f,0)-1
    tl.store(Rows+position,r,(r<M)&(f!=0))
    tl.store(Count,tl.sum(f,0))

@g.jit(do_not_specialize=['NB'])
def _parallel_exact(Q,Bounds,Starts,Ends,Rows,Count,Partial,
                    H:gl.constexpr,D:gl.constexpr,NB,POOL:gl.constexpr,
                    CAP:gl.constexpr,BH:gl.constexpr,BT:gl.constexpr,SPLITS:gl.constexpr):
    slot=gl.program_id(0)
    count=gl.load(Count)
    if (slot<count)&(count<=CAP):
        row=gl.load(Rows+slot)
        hg=gl.program_id(1)*BH
        split=gl.program_id(2)
        layout:gl.constexpr=gl.BlockedLayout([1,1,16],[4,1,8],[1,4,1],[2,1,0])
        q_layout:gl.constexpr=gl.SliceLayout(1,layout)
        k_layout:gl.constexpr=gl.SliceLayout(0,layout)
        best_layout:gl.constexpr=gl.SliceLayout(2,layout)
        hq=hg+gl.arange(0,BH,layout=gl.SliceLayout(1,q_layout))
        dq=gl.arange(0,D,layout=gl.SliceLayout(0,q_layout))
        bk=gl.arange(0,BT,layout=gl.SliceLayout(1,k_layout))
        dk=gl.arange(0,D,layout=gl.SliceLayout(0,k_layout))
        q=gl.load(Q+row*H*D+hq[:,None]*D+dq[None,:],hq[:,None]<H,other=0.).to(gl.float32)
        pos,neg=gl.maximum(q,0.),gl.maximum(-q,0.)
        start,end=gl.load(Starts+row),gl.load(Ends+row)
        first,last=start//POOL,gl.maximum(end-1,0)//POOL
        after=gl.cdiv(end,POOL)
        tiles=gl.cdiv(gl.maximum(after-first,0),BT)
        per_split=gl.cdiv(tiles,SPLITS)
        best=gl.full((BH,BT),0.,gl.float32,layout=best_layout)
        for tile in range(split*per_split,gl.minimum((split+1)*per_split,tiles)):
            block=first+tile*BT+bk
            p=block
            p=gl.where((block==last)&(end%POOL!=0),NB+row*2+1,p)
            p=gl.where((block==first)&((start%POOL!=0)|(first==last)),NB+row*2,p)
            mx=gl.load(Bounds+p[:,None]*3*D+dk[None,:],block[:,None]<after,other=0.)
            mn=gl.load(Bounds+p[:,None]*3*D+D+dk[None,:],block[:,None]<after,other=0.)
            upper=gl.sum(neg[:,None,:]*mx[None,:,:]-pos[:,None,:]*mn[None,:,:],2)
            best=gl.maximum(best,upper)
        head_best=gl.max(best,1)
        hs=hg+gl.arange(0,BH,layout=gl.SliceLayout(1,best_layout))
        gl.store(Partial+(split*CAP+slot)*H+hs,head_best,hs<H)

@triton.jit
def _merge(Partial,Rows,Count,W,Scores,H:tl.constexpr,CAP:tl.constexpr,SPLITS:tl.constexpr):
    slot=tl.program_id(0)
    count=tl.load(Count)
    if (slot<count)&(count<=CAP):
        row=tl.load(Rows+slot)
        h=tl.arange(0,H)
        s=tl.arange(0,SPLITS)
        value=tl.load(Partial+(s[:,None]*CAP+slot)*H+h[None,:])
        best=tl.max(value,0)
        w=tl.load(W+row*H+h).to(tl.float32)
        tl.store(Scores+row*H+h,best*tl.abs(w))

@triton.jit(do_not_specialize=['NB'])
def _overflow(Q,W,Bounds,Starts,Ends,Scores,Flags,Count,
              H:tl.constexpr,D:tl.constexpr,BH:tl.constexpr,BD:tl.constexpr,
              NB,POOL:tl.constexpr,CAP:tl.constexpr):
    if tl.load(Count)>CAP:
        _exact(Q,W,Bounds,Starts,Ends,Scores,Flags,H,D,BH,BD,NB,POOL)

def repair(q,w,bounds,starts,ends,scores,nb,pool,relative=1.e-5,absolute=1.e-6):
    m,h,d=q.shape
    if m<256:
        return repair_small(q,w,bounds,starts,ends,scores,nb,pool,relative,absolute)
    flags=torch.empty(m,dtype=torch.int32,device=q.device)
    rows=torch.empty_like(flags)
    count=torch.empty((),dtype=torch.int32,device=q.device)
    cap,splits,bh,bt=128,8,16,4
    partial=torch.empty((splits,cap,h),dtype=torch.float32,device=q.device)
    _mark[(m,)](scores,starts,ends,flags,h,pool,relative,absolute,num_warps=4)
    _compact[(1,)](flags,rows,count,m,triton.next_power_of_2(m),num_warps=8)
    _parallel_exact[(cap,triton.cdiv(h,bh),splits)](
        q,bounds,starts,ends,rows,count,partial,h,d,nb,pool,cap,bh,bt,splits,
        num_warps=4,enable_fp_fusion=False)
    _merge[(cap,)](partial,rows,count,w,scores,h,cap,splits,num_warps=4,enable_fp_fusion=False)
    _overflow[(m,)](q,w,bounds,starts,ends,scores,flags,count,h,d,
        triton.next_power_of_2(h),triton.next_power_of_2(d),nb,pool,cap,
        num_warps=8,enable_fp_fusion=False)
    return flags

