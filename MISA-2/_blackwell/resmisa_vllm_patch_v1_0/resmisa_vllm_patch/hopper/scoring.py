"""Shared BF16 high/low extrema; FP32 accumulation on tensor cores."""
import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['SIZE'])
def _pack(Bounds,Hi,Lo,SIZE,D:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    p,di=i//(2*D),i%(2*D)
    value=tl.load(Bounds+p*3*D+di,i<SIZE,other=0.)
    hi=value.to(tl.bfloat16)
    lo=(value-hi.to(tl.float32)).to(tl.bfloat16)
    tl.store(Hi+i,hi,i<SIZE)
    tl.store(Lo+i,lo,i<SIZE)

@triton.jit(do_not_specialize=['NB'])
def _score(Q,W,Hi,Lo,Starts,Ends,Out,
           H:tl.constexpr,D:tl.constexpr,NB,POOL:tl.constexpr,
           BH:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,STAGES:tl.constexpr):
    row=tl.program_id(0)
    heads=tl.program_id(1)*BH+tl.arange(0,BH)
    ds=tl.arange(0,BK)
    q=tl.load(Q+row*H*D+heads[:,None]*D+(ds[None,:]%D),
              (heads[:,None]<H)&(ds[None,:]<2*D),other=0.).to(tl.float32)
    a=tl.where(ds[None,:]<D,tl.maximum(-q,0.),-tl.maximum(q,0.)).to(tl.bfloat16)
    start,end=tl.load(Starts+row),tl.load(Ends+row)
    first,last=start//POOL,tl.maximum(end-1,0)//POOL
    after=tl.cdiv(end,POOL)
    best=tl.full((BH,),0.,tl.float32)
    for tile in tl.range(0,tl.cdiv(tl.maximum(after-first,0),BN),num_stages=STAGES):
        blocks=first+tile*BN+tl.arange(0,BN)
        p=blocks
        p=tl.where((blocks==last)&(end%POOL!=0),NB+row*2+1,p)
        p=tl.where((blocks==first)&((start%POOL!=0)|(first==last)),NB+row*2,p)
        offsets=p[None,:]*2*D+ds[:,None]
        mask=(blocks[None,:]<after)&(ds[:,None]<2*D)
        hi=tl.load(Hi+offsets,mask,other=0.)
        lo=tl.load(Lo+offsets,mask,other=0.)
        upper=tl.dot(a,hi)
        upper=tl.dot(a,lo,upper)
        best=tl.maximum(best,tl.max(upper,1))
    w=tl.load(W+row*H+heads,heads<H,other=0.).to(tl.float32)
    tl.store(Out+row*H+heads,best*tl.abs(w),heads<H)

def launch(q,w,bounds,starts,ends,out,nb,pool,configuration):
    bh,bn,nw,mode,stages=configuration
    assert mode==1
    m,h,d=q.shape
    hi=torch.empty((bounds.shape[0],2*d),dtype=torch.bfloat16,device=q.device)
    lo=torch.empty_like(hi)
    _pack[(triton.cdiv(hi.numel(),1024),)](bounds,hi,lo,hi.numel(),d,1024,enable_fp_fusion=False)
    return _score[(m,triton.cdiv(h,bh))](q,w,hi,lo,starts,ends,out,h,d,nb,pool,
        bh,bn,triton.next_power_of_2(2*d),stages,num_warps=nw,enable_fp_fusion=False)

