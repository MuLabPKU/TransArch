"""GPU WHILE recovery with a 64-row score cache."""
import ctypes
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from . import score as p
from .score import QuantizedG
from .scan import cache_bad_group

_BRIDGE = None

@triton.jit
def _pack_status(Counts, Starts, Ends, Status, RowMap, Summary, rows, n,
                 MINIMUM: tl.constexpr, CAP: tl.constexpr, B: tl.constexpr):
    row = tl.arange(0, B)
    valid = row < rows
    count = tl.load(Counts + row, valid, 0)
    start = tl.load(Starts + row, valid, 0)
    end = tl.load(Ends + row, valid, 0)
    status = tl.load(Status + row, valid, 0)
    status |= (count < tl.minimum(MINIMUM, tl.maximum(end-start, 0))).to(tl.int32)
    status |= (count > CAP).to(tl.int32) * 2
    status |= ((start < 0) | (end < start) | (end > n)).to(tl.int32) * 8
    bad = valid & (status != 0)
    slot = tl.cumsum(bad.to(tl.int32), 0) - 1
    tl.store(RowMap + slot, row, bad)
    tl.store(Status + row, status, valid)
    tl.store(Summary, tl.sum(bad.to(tl.int32), 0))
    tl.store(Summary + 1, tl.sum((valid & ((status & 8) != 0)).to(tl.int32), 0))
    tl.store(Summary + 2, tl.sum((valid & ((status & 16) != 0)).to(tl.int32), 0))


@triton.jit
def _initial_keys(Tau, Keys, rows, B: tl.constexpr):
    row=tl.program_id(0)*B+tl.arange(0,B)
    tau=tl.load(Tau+row,row<rows,0.)
    bits=tl.where(tau==0.,0,tau.to(tl.uint32,bitcast=True))
    code=tl.where((bits & 0x80000000)!=0,~bits,bits ^ 0x80000000).to(tl.uint32)
    tl.store(Keys+row,code.to(tl.uint64)<<32,row<rows)


@triton.jit
def _count_cache(Scores, RowMap, Summary, Keys, Counts, Starts, Ends, Active,
                 n, STRIDE: tl.constexpr, B: tl.constexpr):
    slot=tl.program_id(1)
    if tl.load(Active+slot) != 0:
        row=tl.load(RowMap+tl.load(Summary+3)+slot)
        ki=tl.program_id(0)*B+tl.arange(0,B)
        start=tl.load(Starts+row);end=tl.load(Ends+row)
        valid=(ki<n) & (ki>=start) & (ki<end)
        score=tl.load(Scores+slot.to(tl.int64)*STRIDE+ki,valid,0.)
        bits=tl.where(score==0.,0,score.to(tl.uint32,bitcast=True))
        code=tl.where((bits & 0x80000000)!=0,~bits,bits ^ 0x80000000).to(tl.uint32)
        key=(code.to(tl.uint64)<<32) | (n-ki).to(tl.uint32).to(tl.uint64)
        pivot=tl.load(Keys+row)
        count=tl.sum((valid & (key>=pivot)).to(tl.int32),0)
        tl.atomic_add(Counts+row,count,sem='relaxed')
        invalid=tl.sum((valid & ((score!=score) | (tl.abs(score)==float('inf')))).to(tl.int32),0)
        tl.atomic_add(Summary+2,invalid,invalid!=0,sem='relaxed')


@triton.jit
def _compact_cache(Scores, RowMap, Summary, Keys, IDs, Counts, Starts, Ends,
                    Debug, n, STRIDE: tl.constexpr, CAP: tl.constexpr,
                    DEBUG: tl.constexpr, B: tl.constexpr):
    slot=tl.program_id(1)
    packed=tl.load(Summary+3)+slot
    if packed < tl.load(Summary):
        row=tl.load(RowMap+packed)
        ki=tl.program_id(0)*B+tl.arange(0,B)
        start=tl.load(Starts+row);end=tl.load(Ends+row)
        score=tl.load(Scores+slot.to(tl.int64)*STRIDE+ki,ki<n,float('-inf'))
        bits=tl.where(score==0.,0,score.to(tl.uint32,bitcast=True))
        code=tl.where((bits & 0x80000000)!=0,~bits,bits ^ 0x80000000).to(tl.uint32)
        key=(code.to(tl.uint64)<<32) | (n-ki).to(tl.uint32).to(tl.uint64)
        pivot=tl.load(Keys+row)
        keep=(ki<n) & (ki>=start) & (ki<end) & (key>=pivot)
        count=tl.sum(keep.to(tl.int32),0)
        prefix=tl.cumsum(keep.to(tl.int32),0)-1
        base=tl.atomic_add(Counts+row,count,sem='relaxed')
        offset=base+prefix
        address=row.to(tl.int64)*CAP+offset.to(tl.int64)
        tl.store(IDs+address,ki,keep & (offset<CAP))
        if DEBUG:
            tl.store(Debug+address,score,keep & (offset<CAP))


class Bridge:
    def __init__(self):
        from ...build import extension
        path = extension("threshold_recovery")
        self.lib=ctypes.CDLL(str(path))
        i=ctypes.c_int;z=ctypes.c_int64;v=ctypes.c_void_p
        self.lib.hisa_last_error.restype=ctypes.c_char_p
        specs={
            'hisa_graph_create':([z,i],v),'hisa_graph_stream':([v],z),
            'hisa_graph_begin':([v],i),'hisa_graph_end':([v],i),
            'hisa_graph_launch':([v,z],i),'hisa_graph_destroy':([v],i),
            'hisa_inner_begin':([v],i),'hisa_inner_end':([v],i),
            'hisa_group_begin_launch':([v,z,z,z,z,z,z,z],i),
            'hisa_group_update_launch':([z,z,z,z,z,z,z,z,z,z,z,i,i,i,i,z],i),
            'hisa_group_check_launch':([v,z,z,z,z,z,z,z,z,i,i,z],i),
            'hisa_group_finish_launch':([z,z,z,z,z,z,z],i),
            'hisa_advance_launch':([v,z,z,z,z,z,z],i),
        }
        for name,(args,result) in specs.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=result

    def check(self,value):
        if value:
            raise RuntimeError(self.lib.hisa_last_error().decode())


@dataclass
class DeviceSelection:
    fast: p.Selection
    gpu_stats: torch.Tensor
    status_checks: int = 0

    @property
    def indices(self):
        return self.fast.indices


def bridge():
    global _BRIDGE
    if _BRIDGE is None:_BRIDGE=Bridge()
    return _BRIDGE


class RecoveryGraph:
    def __init__(self, ws, prepared, n, debug):
        b=bridge();lib=b.lib
        self.bridge=b;self.refs=(prepared.keys,prepared.key_scales);self.handle=None
        args=(ws.g.high,ws.g.high_scale,ws.g.low,ws.g.low_scale,prepared.keys,prepared.key_scales,
              ws.starts,ws.ends,ws.rowmap,ws.summary,ws.scores,ws.rows,n)
        scan_kw=dict(STRIDE=ws.max_n,BN=64,num_warps=4,enable_fp_fusion=False,maxnreg=96)
        compact_args=(ws.scores,ws.rowmap,ws.summary,ws.threshold_keys,ws.ids,ws.counts,
                      ws.starts,ws.ends,ws.debug if debug else ws.ids,n)
        compact_kw=dict(STRIDE=ws.max_n,CAP=ws.budget.capacity,DEBUG=debug,B=1024,num_warps=4)
        count_args=(ws.scores,ws.rowmap,ws.summary,ws.threshold_keys,ws.counts,ws.starts,ws.ends,ws.active,n)
        count_kw=dict(STRIDE=ws.max_n,B=2048,num_warps=4)
        # Compile without running kernels or reading uninitialized device state.
        compiled_scan=cache_bad_group.warmup(*args,grid=(max(1,triton.cdiv(n,64)),),**scan_kw)
        compiled_compact=_compact_cache.warmup(*compact_args,grid=(max(1,triton.cdiv(n,1024)),64),**compact_kw)
        compiled_count=_count_cache.warmup(*count_args,grid=(max(1,triton.cdiv(n,2048)),64),**count_kw)
        # CUDA module loading is a cold-path operation and must occur before
        # entering raw stream capture, even when Triton compilation is cached.
        compiled_scan._init_handles();compiled_compact._init_handles();compiled_count._init_handles()
        self.handle=lib.hisa_graph_create(ws.summary.data_ptr(),ws.rows)
        if not self.handle:raise RuntimeError(lib.hisa_last_error().decode())
        raw_stream=lib.hisa_graph_stream(self.handle)
        stream=torch.cuda.ExternalStream(raw_stream,device=ws.device)
        with torch.cuda.stream(stream):
            b.check(lib.hisa_graph_begin(self.handle))
            cache_bad_group[(max(1,triton.cdiv(n,64)),)](*args,**scan_kw)
            b.check(lib.hisa_group_begin_launch(self.handle,ws.rowmap.data_ptr(),ws.summary.data_ptr(),
                                                ws.lower.data_ptr(),ws.upper.data_ptr(),ws.active.data_ptr(),
                                                ws.iterations.data_ptr(),raw_stream))
            b.check(lib.hisa_inner_begin(self.handle))
            b.check(lib.hisa_group_update_launch(
                ws.rowmap.data_ptr(),ws.summary.data_ptr(),
                ws.starts.data_ptr(),ws.ends.data_ptr(),ws.thresholds.data_ptr(),ws.threshold_keys.data_ptr(),
                ws.counts.data_ptr(),ws.lower.data_ptr(),ws.upper.data_ptr(),ws.active.data_ptr(),ws.iterations.data_ptr(),
                n,ws.budget.minimum,ws.budget.target,ws.budget.capacity,raw_stream))
            _count_cache[(max(1,triton.cdiv(n,2048)),64)](*count_args,**count_kw)
            b.check(lib.hisa_group_check_launch(self.handle,ws.rowmap.data_ptr(),ws.summary.data_ptr(),
                                                ws.starts.data_ptr(),ws.ends.data_ptr(),ws.counts.data_ptr(),
                                                ws.active.data_ptr(),ws.iterations.data_ptr(),ws.expected.data_ptr(),
                                                ws.budget.minimum,ws.budget.capacity,raw_stream))
            b.check(lib.hisa_inner_end(self.handle))
            b.check(lib.hisa_group_finish_launch(ws.rowmap.data_ptr(),ws.summary.data_ptr(),ws.counts.data_ptr(),
                                                 ws.status.data_ptr(),ws.active.data_ptr(),ws.expected.data_ptr(),raw_stream))
            _compact_cache[(max(1,triton.cdiv(n,1024)),64)](*compact_args,**compact_kw)
            b.check(lib.hisa_advance_launch(self.handle,ws.rowmap.data_ptr(),ws.summary.data_ptr(),
                                            ws.counts.data_ptr(),ws.status.data_ptr(),ws.expected.data_ptr(),raw_stream))
            b.check(lib.hisa_graph_end(self.handle))

    def launch(self):
        self.bridge.check(self.bridge.lib.hisa_graph_launch(self.handle,torch.cuda.current_stream().cuda_stream))

    def __del__(self):
        if self.handle:
            self.bridge.lib.hisa_graph_destroy(self.handle)


class Workspace:
    def __init__(self, rows, device, budget, max_n, debug=False):
        self.rows=rows;self.device=device;self.budget=budget;self.max_n=max_n
        self.graphs={}
        def alloc(shape,dtype):return torch.empty(shape,dtype=dtype,device=device)
        self.ids=alloc((rows,budget.capacity),torch.int32)
        self.counts=alloc((rows,),torch.int32);self.status=alloc((rows,),torch.int32)
        self.thresholds=alloc((rows,),torch.float32);self.threshold_keys=alloc((rows,),torch.uint64)
        self.pilot=alloc((rows,budget.samples),torch.float32)
        self.starts=alloc((rows,),torch.int32);self.ends=alloc((rows,),torch.int32)
        self.rowmap=alloc((rows,),torch.int32);self.summary=alloc((16,),torch.int32)
        self.expected=alloc((64,),torch.int32)
        self.lower=alloc((64,),torch.uint64);self.upper=alloc((64,),torch.uint64)
        self.active=alloc((64,),torch.int32);self.iterations=alloc((64,),torch.int32)
        self.scores=alloc((64,max_n),torch.float32)
        self.debug=alloc((rows,budget.capacity),torch.float32) if debug else None
        self.g=QuantizedG(alloc((rows,128),torch.float8_e4m3fn),alloc((rows,),torch.float32),
                          alloc((rows,128),torch.float8_e4m3fn),alloc((rows,),torch.float32),2)

    def graph(self,prepared,n,debug):
        key=(n,prepared.keys.data_ptr(),prepared.key_scales.data_ptr(),debug)
        if key not in self.graphs:
            # Bound graph ownership across prefill lengths. The graph records
            # stable K storage, but its launch dimensions depend on N.
            if self.graphs:
                torch.cuda.current_stream(self.device).synchronize()
                self.graphs.clear()
            self.graphs[key] = RecoveryGraph(self,prepared,n,debug)
        return self.graphs[key]
