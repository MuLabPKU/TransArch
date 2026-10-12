"""SM100 scan and compaction in the Tensor Core result layout.

Each thread owns 32 adjacent keys for one row. Lanes 0..15 and 16..31 own
the lower and upper halves of those rows, so scores, predicates, counts,
and offsets use the same thread layout.
"""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,TensorMemoryLayout,tcgen05_mma,fence_async_shared,mbarrier)


@gluon.jit
def reserve_native_prefix(selected,pointer):
    # Sequential prefix outputs are written before all inputs are consumed.
    # Early-clobber constraints explicitly forbid output/input aliasing.
    return gl.inline_asm_elementwise(
        """{
.reg .b32 total, other, row_count, lane, base, adjustment, src;
.reg .pred leader, nonempty;
mov.u32 $0, 0;
add.u32 $1, $0, $32;
add.u32 $2, $1, $33;
add.u32 $3, $2, $34;
add.u32 $4, $3, $35;
add.u32 $5, $4, $36;
add.u32 $6, $5, $37;
add.u32 $7, $6, $38;
add.u32 $8, $7, $39;
add.u32 $9, $8, $40;
add.u32 $10, $9, $41;
add.u32 $11, $10, $42;
add.u32 $12, $11, $43;
add.u32 $13, $12, $44;
add.u32 $14, $13, $45;
add.u32 $15, $14, $46;
add.u32 $16, $15, $47;
add.u32 $17, $16, $48;
add.u32 $18, $17, $49;
add.u32 $19, $18, $50;
add.u32 $20, $19, $51;
add.u32 $21, $20, $52;
add.u32 $22, $21, $53;
add.u32 $23, $22, $54;
add.u32 $24, $23, $55;
add.u32 $25, $24, $56;
add.u32 $26, $25, $57;
add.u32 $27, $26, $58;
add.u32 $28, $27, $59;
add.u32 $29, $28, $60;
add.u32 $30, $29, $61;
add.u32 $31, $30, $62;
add.u32 total, $31, $63;
shfl.sync.bfly.b32 other, total, 16, 31, 0xffffffff;
add.u32 row_count, total, other;
mov.u32 lane, %laneid;
setp.lt.u32 leader, lane, 16;
setp.ne.u32 nonempty, row_count, 0;
and.pred nonempty, nonempty, leader;
mov.u32 base, 0;
@nonempty atom.relaxed.gpu.global.add.u32 base, [$64], row_count;
and.b32 src, lane, 15;
shfl.sync.idx.b32 base, base, src, 31, 0xffffffff;
selp.u32 adjustment, 0, other, leader;
add.u32 adjustment, adjustment, base;
add.u32 $0, $0, adjustment;
add.u32 $1, $1, adjustment;
add.u32 $2, $2, adjustment;
add.u32 $3, $3, adjustment;
add.u32 $4, $4, adjustment;
add.u32 $5, $5, adjustment;
add.u32 $6, $6, adjustment;
add.u32 $7, $7, adjustment;
add.u32 $8, $8, adjustment;
add.u32 $9, $9, adjustment;
add.u32 $10, $10, adjustment;
add.u32 $11, $11, adjustment;
add.u32 $12, $12, adjustment;
add.u32 $13, $13, adjustment;
add.u32 $14, $14, adjustment;
add.u32 $15, $15, adjustment;
add.u32 $16, $16, adjustment;
add.u32 $17, $17, adjustment;
add.u32 $18, $18, adjustment;
add.u32 $19, $19, adjustment;
add.u32 $20, $20, adjustment;
add.u32 $21, $21, adjustment;
add.u32 $22, $22, adjustment;
add.u32 $23, $23, adjustment;
add.u32 $24, $24, adjustment;
add.u32 $25, $25, adjustment;
add.u32 $26, $26, adjustment;
add.u32 $27, $27, adjustment;
add.u32 $28, $28, adjustment;
add.u32 $29, $29, adjustment;
add.u32 $30, $30, adjustment;
add.u32 $31, $31, adjustment;
}""",
        constraints='=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,=&r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l',
        args=[selected,pointer],dtype=gl.int32,is_pure=False,pack=32)


@gluon.jit
def compact_scan(Hi, HS, Lo, LS, K, KS, Starts, Ends, Tau, IDs, Counts, DebugScores,
                 rows, n, CAP:gl.constexpr, BN:gl.constexpr=64,
                 DEBUG:gl.constexpr=False, ThresholdKeys=None, KEY_THRESHOLD:gl.constexpr=False):
    gl.static_assert(BN==64)
    A:gl.constexpr=gl.BlockedLayout([1,16],[4,8],[4,1],[1,0])
    B:gl.constexpr=gl.BlockedLayout([16,1],[8,4],[1,4],[0,1])
    TM:gl.constexpr=gl.DistributedLinearLayout(
        [[0,1],[0,2],[0,4],[0,8],[0,16]],
        [[1,0],[2,0],[4,0],[8,0],[0,32]],[[16,0],[32,0]],[],[64,64])
    qa=gl.program_id(1)*64+gl.arange(0,64,layout=gl.SliceLayout(1,A))
    da=gl.arange(0,128,layout=gl.SliceLayout(0,A))
    db=gl.arange(0,128,layout=gl.SliceLayout(1,B))
    kb=gl.program_id(0)*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,B))
    hi=gl.load(Hi+qa[:,None]*128+da[None,:],qa[:,None]<rows,gl.full([64,128],0,gl.float32,A))
    lo=gl.load(Lo+qa[:,None]*128+da[None,:],qa[:,None]<rows,gl.full([64,128],0,gl.float32,A))
    key=gl.load(K+kb[None,:]*128+db[:,None],kb[None,:]<n,gl.full([128,BN],0,gl.float32,B))
    ah=gl.allocate_shared_memory(hi.dtype,[64,128],gl.NVMMASharedLayout(128,8,2),hi)
    al=gl.allocate_shared_memory(lo.dtype,[64,128],gl.NVMMASharedLayout(128,8,2),lo)
    b=gl.allocate_shared_memory(key.dtype,[128,BN],gl.NVMMASharedLayout(128,8,2,transposed=True),key)
    th=allocate_tensor_memory(gl.float32,[64,BN],TensorMemoryLayout([64,BN],col_stride=1))
    tl=allocate_tensor_memory(gl.float32,[64,BN],TensorMemoryLayout([64,BN],col_stride=1))
    bh=gl.allocate_shared_memory(gl.int64,[1],mbarrier.MBarrierLayout())
    bl=gl.allocate_shared_memory(gl.int64,[1],mbarrier.MBarrierLayout())
    mbarrier.init(bh,count=1);mbarrier.init(bl,count=1)
    fence_async_shared()
    tcgen05_mma(ah,b,th,use_acc=False,mbarriers=[bh],mbarrier_preds=[True])
    tcgen05_mma(al,b,tl,use_acc=False,mbarriers=[bl],mbarrier_preds=[True])
    mbarrier.wait(bh,phase=0);mbarrier.wait(bl,phase=0)
    mbarrier.invalidate(bh);mbarrier.invalidate(bl)
    high=th.load(TM);low=tl.load(TM)
    qm=gl.program_id(1)*64+gl.arange(0,64,layout=gl.SliceLayout(1,TM))
    km=gl.program_id(0)*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,TM))
    zm=gl.full([64],0,gl.float32,gl.SliceLayout(1,TM))
    zk=gl.full([BN],0,gl.float32,gl.SliceLayout(0,TM))
    hs=gl.load(HS+qm,qm<rows,zm);ls=gl.load(LS+qm,qm<rows,zm)
    ks=gl.load(KS+km,km<n,zk)
    score=(high*hs[:,None]+low*ls[:,None])*ks[None,:]
    start=gl.load(Starts+qm,qm<rows,zm.to(gl.int32))
    end=gl.load(Ends+qm,qm<rows,zm.to(gl.int32))
    tau=gl.load(Tau+qm,qm<rows,gl.full([64],float('inf'),gl.float32,gl.SliceLayout(1,TM)))
    if KEY_THRESHOLD:
        bits=gl.where(score==0.,0,score.to(gl.uint32,bitcast=True))
        code=gl.where((bits & 0x80000000)!=0,~bits,bits ^ 0x80000000).to(gl.uint32)
        keys=(code.to(gl.uint64)<<32) | (n-km[None,:]).to(gl.uint32).to(gl.uint64)
        pivot=gl.load(ThresholdKeys+qm,qm<rows,zm.to(gl.uint64)).to(gl.uint64)
        passes=keys>=pivot[:,None]
    else:
        passes=score>=tau[:,None]
    native_keep=((qm[:,None]<rows) & (km[None,:]<n) &
                 (km[None,:]>=start[:,None]) & (km[None,:]<end[:,None]) & passes)
    slot=reserve_native_prefix(native_keep.to(gl.int32),Counts+qm[:,None])
    addr=qm[:,None].to(gl.int64)*CAP+slot.to(gl.int64)
    write=native_keep & (slot<CAP)
    gl.store(IDs+addr,km[None,:],write)
    if DEBUG:
        gl.store(DebugScores+addr,score,write)


@gluon.jit
def cache_bad_group(Hi, HS, Lo, LS, K, KS, Starts, Ends, RowMap, Summary, Scores,
                    rows, n, STRIDE: gl.constexpr, BN: gl.constexpr = 64):
    cursor=gl.load(Summary+3)
    active=gl.minimum(64,gl.load(Summary)-cursor)
    gl.static_assert(BN==64)
    A:gl.constexpr=gl.BlockedLayout([1,16],[4,8],[4,1],[1,0])
    B:gl.constexpr=gl.BlockedLayout([16,1],[8,4],[1,4],[0,1])
    TM:gl.constexpr=gl.DistributedLinearLayout(
        [[0,1],[0,2],[0,4],[0,8],[0,16]],
        [[1,0],[2,0],[4,0],[8,0],[0,32]],[[16,0],[32,0]],[],[64,64])
    qa_slot=gl.arange(0,64,layout=gl.SliceLayout(1,A))
    qa=gl.load(RowMap+cursor+qa_slot,qa_slot<active,rows)
    da=gl.arange(0,128,layout=gl.SliceLayout(0,A))
    db=gl.arange(0,128,layout=gl.SliceLayout(1,B))
    kb=gl.program_id(0)*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,B))
    hi=gl.load(Hi+qa[:,None]*128+da[None,:],qa[:,None]<rows,gl.full([64,128],0,gl.float32,A))
    lo=gl.load(Lo+qa[:,None]*128+da[None,:],qa[:,None]<rows,gl.full([64,128],0,gl.float32,A))
    key=gl.load(K+kb[None,:]*128+db[:,None],kb[None,:]<n,gl.full([128,BN],0,gl.float32,B))
    ah=gl.allocate_shared_memory(hi.dtype,[64,128],gl.NVMMASharedLayout(128,8,2),hi)
    al=gl.allocate_shared_memory(lo.dtype,[64,128],gl.NVMMASharedLayout(128,8,2),lo)
    b=gl.allocate_shared_memory(key.dtype,[128,BN],gl.NVMMASharedLayout(128,8,2,transposed=True),key)
    th=allocate_tensor_memory(gl.float32,[64,BN],TensorMemoryLayout([64,BN],col_stride=1))
    tl=allocate_tensor_memory(gl.float32,[64,BN],TensorMemoryLayout([64,BN],col_stride=1))
    bh=gl.allocate_shared_memory(gl.int64,[1],mbarrier.MBarrierLayout())
    bl=gl.allocate_shared_memory(gl.int64,[1],mbarrier.MBarrierLayout())
    mbarrier.init(bh,count=1);mbarrier.init(bl,count=1)
    fence_async_shared()
    tcgen05_mma(ah,b,th,use_acc=False,mbarriers=[bh],mbarrier_preds=[True])
    tcgen05_mma(al,b,tl,use_acc=False,mbarriers=[bl],mbarrier_preds=[True])
    mbarrier.wait(bh,phase=0);mbarrier.wait(bl,phase=0)
    mbarrier.invalidate(bh);mbarrier.invalidate(bl)
    high=th.load(TM);low=tl.load(TM)
    qm_slot=gl.arange(0,64,layout=gl.SliceLayout(1,TM))
    qm=gl.load(RowMap+cursor+qm_slot,qm_slot<active,rows)
    km=gl.program_id(0)*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,TM))
    zm=gl.full([64],0,gl.float32,gl.SliceLayout(1,TM))
    zk=gl.full([BN],0,gl.float32,gl.SliceLayout(0,TM))
    hs=gl.load(HS+qm,qm<rows,zm);ls=gl.load(LS+qm,qm<rows,zm)
    ks=gl.load(KS+km,km<n,zk)
    score=(high*hs[:,None]+low*ls[:,None])*ks[None,:]
    start=gl.load(Starts+qm,qm<rows,zm.to(gl.int32))
    end=gl.load(Ends+qm,qm<rows,zm.to(gl.int32))
    valid=(km[None,:]>=start[:,None]) & (km[None,:]<end[:,None])
    score=gl.where(valid,score,float('-inf'))
    gl.store(Scores+qm_slot[:,None].to(gl.int64)*STRIDE+km[None,:],score,
             (qm_slot[:,None]<active) & (km[None,:]<n))


class ScanAdapter:
    def __init__(self,original):self.original=original

    def __getitem__(self,grid):
        def launch(*args,**kw):
            if (kw['BQ'],kw['BN'],kw['ACC'],kw['num_warps'])!=(64,64,32,4):
                return self.original[grid](*args,**kw)
            rows,n=args[-2:]
            return compact_scan[(triton.cdiv(n,64),triton.cdiv(rows,64))](
                *args,CAP=kw['CAP'],BN=64,DEBUG=kw['DEBUG'],ThresholdKeys=kw['ThresholdKeys'],
                KEY_THRESHOLD=kw['KEY_THRESHOLD'],num_warps=4,enable_fp_fusion=False,maxnreg=96)
        return launch


def install(module):module._scan_compact_kernel=ScanAdapter(module._scan_compact_kernel)


@gluon.jit
def _add(a,b):
    return a+b


@gluon.jit
def compact_scan_tuned(Hi, HS, Lo, LS, K, KS, Starts, Ends, Tau, IDs, Counts, DebugScores,
                 rows, n, CAP:gl.constexpr, BQ:gl.constexpr, BN:gl.constexpr, GROUP_Q:gl.constexpr,
                 DEBUG:gl.constexpr=False, ThresholdKeys=None, KEY_THRESHOLD:gl.constexpr=False):
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
    kb=key_tile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,B))
    hi=gl.load(Hi+qa[:,None]*128+da[None,:],qa[:,None]<rows,gl.full([BQ,128],0,gl.float32,A))
    lo=gl.load(Lo+qa[:,None]*128+da[None,:],qa[:,None]<rows,gl.full([BQ,128],0,gl.float32,A))
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
    km=key_tile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,TM))
    zm=gl.full([BQ],0,gl.float32,gl.SliceLayout(1,TM))
    zk=gl.full([BN],0,gl.float32,gl.SliceLayout(0,TM))
    hs=gl.load(HS+qm,qm<rows,zm);ls=gl.load(LS+qm,qm<rows,zm)
    ks=gl.load(KS+km,km<n,zk)
    score=(high*hs[:,None]+low*ls[:,None])*ks[None,:]
    start=gl.load(Starts+qm,qm<rows,zm.to(gl.int32))
    end=gl.load(Ends+qm,qm<rows,zm.to(gl.int32))
    tau=gl.load(Tau+qm,qm<rows,gl.full([BQ],float('inf'),gl.float32,gl.SliceLayout(1,TM)))
    if KEY_THRESHOLD:
        bits=gl.where(score==0.,0,score.to(gl.uint32,bitcast=True))
        code=gl.where((bits & 0x80000000)!=0,~bits,bits ^ 0x80000000).to(gl.uint32)
        keys=(code.to(gl.uint64)<<32) | (n-km[None,:]).to(gl.uint32).to(gl.uint64)
        pivot=gl.load(ThresholdKeys+qm,qm<rows,zm.to(gl.uint64)).to(gl.uint64)
        passes=keys>=pivot[:,None]
    else:
        passes=score>=tau[:,None]
    native_keep=((qm[:,None]<rows) & (km[None,:]<n) &
                 (km[None,:]>=start[:,None]) & (km[None,:]<end[:,None]) & passes)
    if BQ == 64:
        slot=reserve_native_prefix(native_keep.to(gl.int32),Counts+qm[:,None])
    else:
        selected = native_keep.to(gl.int32)
        prefix = gl.associative_scan(selected, 1, _add) - selected
        count = gl.sum(selected, 1)
        begin = gl.atomic_add(Counts+qm, count, mask=(qm<rows)&(count>0), sem='relaxed')
        slot = prefix + begin[:,None]
    addr=qm[:,None].to(gl.int64)*CAP+slot.to(gl.int64)
    write=native_keep & (slot<CAP)
    gl.store(IDs+addr,km[None,:],write)
    if DEBUG:
        gl.store(DebugScores+addr,score,write)

