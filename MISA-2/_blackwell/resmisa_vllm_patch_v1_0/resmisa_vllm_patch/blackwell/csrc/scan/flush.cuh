// V5 cold path: preserve the native ring reservation, packed ordering and
// exact-once epoch discipline; keep its instructions outside the scan loop.
template <uint32_t Q, uint32_t SLOTS, uint32_t KV>
static __device__ __noinline__ void flush_native_ring(
    LaneCounts<Q> counts, uint32_t* ring,
    CandidateValue* cv, int32_t* ci, int32_t* cc, uint32_t cap,
    uint32_t first_q, uint32_t rows, uint32_t window_base,
    uint32_t token_lane,int32_t* hist) {
  const uint32_t lane = ptx::get_lane_idx();
  constexpr unsigned FULL = 0xffffffffu;
  uint32_t row_total = 0;
  #pragma unroll
  for (uint32_t i=0; i<Q; ++i) {
    uint32_t total=__reduce_add_sync(FULL,counts.get(i));
    if (lane==i) row_total=total;
  }
  int row_base=0;
  if (lane<Q && first_q+lane<rows && row_total)
    row_base=atomicAdd(cc+first_q+lane,static_cast<int>(row_total));
  uint32_t inclusive_pairs[Q/2];
  #pragma unroll
  for (uint32_t p=0; p<Q/2; ++p) {
    uint32_t v=counts.get(p*2) | (counts.get(p*2+1)<<16);
    #pragma unroll
    for (int delta=1; delta<32; delta<<=1) {
      uint32_t other=__shfl_up_sync(FULL,v,delta);
      if (lane>=static_cast<uint32_t>(delta)) v+=other;
    }
    inclusive_pairs[p]=v;
  }
  #pragma unroll
  for (uint32_t i=0; i<Q; ++i) {
    uint32_t count=counts.get(i);
    uint32_t offset=((inclusive_pairs[i/2]>>((i&1u)*16))&65535u)-count;
    int base=__shfl_sync(FULL,row_base,i);
    #pragma unroll 1
    for (uint32_t slot=0; slot<count; ++slot) {
      const uint32_t record=ring[(i*SLOTS+slot)*32u+lane];
      uint32_t bits=record&0xffffff00u;
      bits=(bits&0x80000000u)?(bits^0x80000000u):~bits;
      const float value=__uint_as_float(bits);
      const uint32_t bucket=value<0.f?0u:(value<255.f?static_cast<uint32_t>(value):255u);
      if(first_q+i<rows) {
        atomicAdd(hist+i*256u+bucket,1);
        if(static_cast<uint64_t>(base)+offset+slot<cap) {
          const uint32_t token=window_base+(record&255u)*KV+token_lane;
          const uint64_t out=static_cast<uint64_t>(first_q+i)*cap+base+offset+slot;
          store_candidate_record(cv+out,ci+out,(record&0xffffff00u)>>8,token);
        }
      }
    }
  }
  __syncwarp();
}

template <uint32_t Q, uint32_t WARPS>
static __device__ __noinline__ void tighten_flush_subset(
    const int32_t* hist,uint32_t* gates,uint32_t topk,
    uint32_t first_q,uint32_t rows) {
  const uint32_t lane=ptx::get_lane_idx();
  const uint32_t warp=threadIdx.x/32u;
  for(uint32_t row=warp;row<Q;row+=WARPS) {
    int carry=0,found=-1;
    #pragma unroll 1
    for(uint32_t base=0;base<256 && found<0;base+=32) {
      const int value=hist[row*256u+base+lane];
      const int total=__reduce_add_sync(0xffffffffu,value);
      if(carry+total<static_cast<int>(topk)) {carry+=total;continue;}
      int prefix=value;
      #pragma unroll
      for(int d=1;d<32;d<<=1) {
        const int prev=__shfl_up_sync(0xffffffffu,prefix,d);
        if(lane>=static_cast<uint32_t>(d)) prefix+=prev;
      }
      const uint32_t mask=__ballot_sync(0xffffffffu,carry+prefix>=static_cast<int>(topk));
      if(mask) found=base+__ffs(mask)-1;
      else carry+=__shfl_sync(0xffffffffu,prefix,31);
    }
    // The clamped last bucket cannot establish a finite conservative edge.
    if(lane==0 && first_q+row<rows && found>=0 && found<255) {
      const uint32_t edge=__float_as_uint(static_cast<float>(found+1));
      gates[row]=min(gates[row],edge);
    }
  }
}
