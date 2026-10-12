// Cold-path flush preserves ring reservation, packed ordering and
// exact-once epoch discipline; keep its instructions outside the scan loop.
template <class T>
CUTLASS_DEVICE T warp_uniform_shared_load(const volatile T* ptr) {
  T value=0;
  if (ptx::get_lane_idx()==0) value=*ptr;
  return __shfl_sync(0xffffffffu,value,0);
}

template <uint32_t Q, uint32_t SLOTS, uint32_t KV>
static __device__ __noinline__ void flush_native_ring(
    LaneCounts<Q> counts, uint32_t* ring, uint32_t* seq,
    CandidateValue* cv, int32_t* ci, int32_t* cc, uint32_t cap,
    uint32_t first_q, uint32_t rows, uint32_t window_base,
    uint32_t token_lane) {
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
    int copies=first_q+i<rows
      ? min(static_cast<int>(count),max(static_cast<int>(cap)-base-static_cast<int>(offset),0)) : 0;
    #pragma unroll 1
    for (int slot=0; slot<copies; ++slot) {
      uint32_t pos=(i*SLOTS+slot)*32u+lane;
      uint32_t record=ring[pos];
      ring[pos]=kRingSentinel;
      uint32_t token=window_base+(record&255u)*KV+token_lane;
      uint64_t out=static_cast<uint64_t>(first_q+i)*cap+base+offset+slot;
      store_candidate_record(cv+out,ci+out,(record&0xffffff00u)>>8,token);
    }
    #pragma unroll 1
    for (int slot=copies; slot<static_cast<int>(count); ++slot)
      ring[(i*SLOTS+slot)*32u+lane]=kRingSentinel;
  }
  __syncwarp();
  __threadfence_block();
  if (lane<Q) seq[lane]+=1u;
  __threadfence_block();
}
