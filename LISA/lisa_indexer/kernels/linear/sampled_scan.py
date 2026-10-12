"""Sample a score threshold, compact candidates, and check the candidate count."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,TensorMemoryLayout,tcgen05_mma,fence_async_shared,mbarrier)
import triton
import triton.language as tl
import torch
from ...build import extension


@gluon.jit
def sample_scores(Q2, W4, K, KS, Starts, Ends, Origin, Inv, Threshold, Values, IDs, Counts,
                 rows, n, CAP:gl.constexpr, BQ:gl.constexpr, BN:gl.constexpr, GROUP_Q:gl.constexpr, SAMPLE:gl.constexpr):
    gl.static_assert(BQ == 64 or BQ == 128)
    gl.static_assert(BN == 64 or BN == 128)
    query_tile = gl.program_id(1) * GROUP_Q + gl.program_id(0) % GROUP_Q
    key_tile = gl.program_id(0) // GROUP_Q
    A:gl.constexpr=gl.BlockedLayout([1,16],[4,8],[4,1],[1,0])
    B:gl.constexpr=gl.BlockedLayout([16,1],[8,4],[1,4],[0,1])
    if BQ == 64:
        TM:gl.constexpr=gl.DistributedLinearLayout(
            [[0,1],[0,2],[0,4],[0,8],[0,16]] + ([[0,64]] if BN == 128 else []),
            [[1,0],[2,0],[4,0],[8,0],[0,32]],[[16,0],[32,0]],[],[BQ,BN])
    else:
        TM:gl.constexpr=gl.DistributedLinearLayout(
            [[0,1],[0,2],[0,4],[0,8],[0,16],[0,32]] + ([[0,64]] if BN == 128 else []),
            [[1,0],[2,0],[4,0],[8,0],[16,0]],[[32,0],[64,0]],[],[BQ,BN])
    qa=query_tile*BQ+gl.arange(0,BQ,layout=gl.SliceLayout(1,A))
    da=gl.arange(0,128,layout=gl.SliceLayout(0,A))
    db=gl.arange(0,128,layout=gl.SliceLayout(1,B))
    kb=12288+((key_tile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,B)))*(n-12288)//SAMPLE)
    hi=gl.load(Q2+qa[:,None]*256+da[None,:],qa[:,None]<rows,gl.full([BQ,128],0,gl.float32,A))
    lo=gl.load(Q2+qa[:,None]*256+128+da[None,:],qa[:,None]<rows,gl.full([BQ,128],0,gl.float32,A))
    key=gl.load(K+kb[None,:]*128+db[:,None],kb[None,:]<n,gl.full([128,BN],0,gl.float32,B))
    ah=gl.allocate_shared_memory(hi.dtype,[BQ,128],gl.NVMMASharedLayout(128,8,2),hi)
    al=gl.allocate_shared_memory(lo.dtype,[BQ,128],gl.NVMMASharedLayout(128,8,2),lo)
    b=gl.allocate_shared_memory(key.dtype,[128,BN],gl.NVMMASharedLayout(128,8,2,transposed=True),key)
    th=allocate_tensor_memory(gl.float32,[BQ,BN],TensorMemoryLayout([BQ,BN],col_stride=1))
    tl=allocate_tensor_memory(gl.float32,[BQ,BN],TensorMemoryLayout([BQ,BN],col_stride=1))
    bh=gl.allocate_shared_memory(gl.int64,[1],mbarrier.MBarrierLayout())
    bl=gl.allocate_shared_memory(gl.int64,[1],mbarrier.MBarrierLayout())
    mbarrier.init(bh,count=1);mbarrier.init(bl,count=1)
    fence_async_shared()
    tcgen05_mma(ah,b,th,use_acc=False,mbarriers=[bh],mbarrier_preds=[True])
    tcgen05_mma(al,b,tl,use_acc=False,mbarriers=[bl],mbarrier_preds=[True])
    mbarrier.wait(bh,phase=0);mbarrier.wait(bl,phase=0)
    mbarrier.invalidate(bh);mbarrier.invalidate(bl)
    high=th.load(TM);low=tl.load(TM)
    qm=query_tile*BQ+gl.arange(0,BQ,layout=gl.SliceLayout(1,TM))
    km=12288+((key_tile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,TM)))*(n-12288)//SAMPLE)
    zm=gl.full([BQ],0,gl.float32,gl.SliceLayout(1,TM))
    zk=gl.full([BN],0,gl.float32,gl.SliceLayout(0,TM))
    hs=gl.load(W4+4*qm,qm<rows,zm);ls=gl.load(W4+4*qm+1,qm<rows,zm)
    ks=gl.load(KS+km,km<n,zk)
    score=(high*hs[:,None]+low*ls[:,None])*ks[None,:]
    start=gl.load(Starts+qm,qm<rows,zm.to(gl.int32))
    end=gl.load(Ends+qm,qm<rows,zm.to(gl.int32))
    origin=gl.load(Origin+qm,qm<rows,zm)
    inv=gl.load(Inv+qm,qm<rows,zm)
    gate=gl.load(Threshold+qm,qm<rows,zm.to(gl.uint32))
    vth=-origin*inv
    affine=gl.fma(-score,inv[:,None],vth[:,None])
    bits=affine.to(gl.uint32,bitcast=True)
    ordered=gl.where((bits&0x80000000)!=0,~bits,bits^0x80000000).to(gl.uint32)
    code=ordered>>8
    passes=code < gate[:,None]
    si=key_tile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,TM))
    valid=(km[None,:]>=start[:,None]) & (km[None,:]<end[:,None])
    gl.store(IDs+qm[:,None]*SAMPLE+si[None,:],gl.where(valid,code,0xffffff),qm[:,None]<rows)


@triton.jit
def choose_gate(Scores,Threshold,Gate,GateBits,SAMPLE:tl.constexpr,RANK:tl.constexpr):
    row=tl.program_id(0);x=tl.arange(0,SAMPLE)
    code=tl.load(Scores+row*SAMPLE+x).to(tl.uint32)
    ordered=code<<8
    bits=tl.where((ordered&0x80000000)!=0,ordered^0x80000000,~ordered)
    value=bits.to(tl.float32,bitcast=True)
    valid=code<0xffffff
    bucket=tl.minimum(tl.maximum(value.to(tl.int32),0),255)
    hist=tl.histogram(bucket,256,mask=valid)
    prefix=tl.cumsum(hist,0)
    boundary=tl.min(tl.where(prefix>RANK,tl.arange(0,256),255),0)
    pivot=tl.max(tl.where(valid&(bucket==boundary),code,0),0)+1
    edge=(tl.load(Threshold+row)+1).to(tl.float32)
    bits=edge.to(tl.uint32,bitcast=True)
    original=tl.where((bits&0x80000000)!=0,~bits,bits^0x80000000)>>8
    gate=tl.minimum(pivot,original)
    tl.store(Gate+row,gate);tl.store(GateBits+row,gate<<8)


@triton.jit
def filter_seed(CV,CI,CC,BackupV,BackupI,BackupC,Gate,
                CAP:tl.constexpr,HEAD:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    count=tl.load(CC+row)
    tl.store(BackupC+row,count)
    gate=tl.load(Gate+row).to(tl.uint32)
    x=tl.arange(0,B)
    total=0
    for start in range(0,tl.cdiv(count,B)):
        slot=start*B+x
        pv=tl.load(CV+row.to(tl.int64)*CAP+slot,slot<count,0).to(tl.uint32)
        pi=tl.load(CI+row.to(tl.int64)*CAP+slot,slot<count,0).to(tl.uint32)
        tl.store(BackupV+row*HEAD+slot,pv,slot<count)
        tl.store(BackupI+row*HEAD+slot,pi,slot<count)
        code=((pi>>20)<<16)|pv
        keep=(slot<count)&(code<gate)
        offset=tl.cumsum(keep.to(tl.int32))-1+total
        tl.store(CV+row.to(tl.int64)*CAP+offset,pv,keep)
        tl.store(CI+row.to(tl.int64)*CAP+offset,pi,keep)
        total+=tl.sum(keep.to(tl.int32),0)
    tl.store(CC+row,total)


@triton.jit
def restore_failed(CV,CI,CC,BackupV,BackupI,BackupC,Starts,RecoveryStarts,Failed,
                   N:tl.constexpr,CAP:tl.constexpr,HEAD:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    count=tl.load(CC+row)
    fail=(count<2048)|(count>CAP)
    tl.store(Failed+row,fail.to(tl.int32))
    tl.store(RecoveryStarts+row,tl.where(fail,tl.load(Starts+row),N))
    if fail:
        old=tl.load(BackupC+row)
        x=tl.arange(0,B)
        for start in range(0,tl.cdiv(old,B)):
            slot=start*B+x
            pv=tl.load(BackupV+row*HEAD+slot,slot<old,0)
            pi=tl.load(BackupI+row*HEAD+slot,slot<old,0)
            tl.store(CV+row.to(tl.int64)*CAP+slot,pv,slot<old)
            tl.store(CI+row.to(tl.int64)*CAP+slot,pi,slot<old)
        tl.store(CC+row,old)


def scan(q2,k,ks,w4,starts,ends,origin,inv,th,cv,ci,cc,hist,config, *, workspace):
    if config not in (8, 9):
        raise ValueError("sampled scan requires configuration 8 or 9")
    q = len(q2)
    n = len(k)
    cap = cv.shape[1]
    sample = 2048
    buffers = workspace
    key=(q2.device,q,cap,sample)
    if key not in buffers:
        alloc=lambda shape,dtype:torch.empty(shape,device=q2.device,dtype=dtype)
        buffers[key]=dict(gate_bits=alloc((q,),torch.int32),scores=alloc((q,sample),torch.int32),gate=alloc((q,),torch.int32),
                     bv=alloc((q,12288),torch.uint16),bi=alloc((q,12288),torch.int32),
                     bc=alloc((q,),torch.int32),recovery=alloc((q,),torch.int32),
                     failed=alloc((q,),torch.int32))
    b=buffers[key]
    sample_scores[(triton.cdiv(sample,128),triton.cdiv(q,64))](
        q2,w4,k,ks,starts,ends,origin,inv,b['gate'],cv,b['scores'],cc,q,n,
        CAP=cap,BQ=64,BN=128,GROUP_Q=1,SAMPLE=sample,num_warps=4,enable_fp_fusion=False)
    rank=min(sample-1,max(1,8192*sample//n))
    choose_gate[(q,)](b['scores'],th,b['gate'],b['gate_bits'],SAMPLE=sample,RANK=rank,num_warps=4)
    filter_seed[(q,)](cv.view(torch.uint16),ci,cc,b['bv'],b['bi'],b['bc'],b['gate'],
                     CAP=cap,HEAD=12288,B=1024,num_warps=4)
    extension("linear_static").scan(q2,k,ks,w4,starts,ends,origin,inv,b['gate_bits'],cv,ci,cc,hist,config)
    restore_failed[(q,)](cv.view(torch.uint16),ci,cc,b['bv'],b['bi'],b['bc'],starts,
                        b['recovery'],b['failed'],N=n,CAP=cap,HEAD=12288,B=1024,num_warps=4)
    # No host status read. Valid rows have an empty suffix; failed rows are
    # recovered using the original conservative HOT seed and online gate.
    native_config=30 if n<=131072 else (31 if n<=262144 else 29)
    extension("linear_recovery").scan(q2,k,ks,w4,b['recovery'],ends,origin,inv,th,cv,ci,cc,hist,native_config)
