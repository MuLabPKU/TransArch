#pragma once

#include <cuda_fp16.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace dsa_litetopk {

using namespace deep_gemm;

inline constexpr uint32_t kEmitChunkBlocks = 256;
inline constexpr uint32_t kEmitLaneSlots = 18;
inline constexpr uint32_t kGateStride = 64;
inline constexpr uint32_t kWarpQueueCap = 64;
inline constexpr uint32_t kMathRegisters = 240;
inline constexpr uint32_t kSpecializedRegisters = 24;
inline constexpr uint32_t kTmemRows = 4;
inline constexpr uint32_t kUmmaStages = 2;
inline constexpr uint32_t kSparseRefreshNs = 512;
inline constexpr uint32_t kSparseRefreshIdleNs = 2048;
// A ring slot can only equal this pattern if a +NaN score passed the gate,
// which both gate comparisons reject; the refresher treats it as "empty".
inline constexpr uint32_t kRingSentinel = 0xffffffffu;
// Gate updates only need to keep pace with the 65,536-token flush windows,
// so the refresher can poll orders of magnitude slower than the dynamic
// histogram daemon and stay invisible to the math warps' issue ports.
inline constexpr uint32_t kRingActiveNs = 2048;
inline constexpr uint32_t kRingIdleNs = 2048;
inline constexpr uint32_t kPagedCandidateInlineRecords = 8192;
inline constexpr uint32_t kPagedCandidatePageRecords = 4096;
inline constexpr uint32_t kPagedCandidateOverflowFlag = 1u << 31;
inline constexpr uint32_t kPooledSegmentMicroRecords = 512;
inline constexpr uint32_t kPooledSegmentPageRecords = 4096;
inline constexpr uint32_t kPooledSegmentSpillPages = 2;
inline constexpr uint32_t kPooledSegmentInvalidStart = 0xffffffffu;

inline constexpr int32_t kPagedCandidateEmpty = -1;
inline constexpr int32_t kPagedCandidateBusy = -2;
inline constexpr int32_t kPagedCandidateFailed = -3;
inline constexpr uint32_t kPagedCandidateBadArguments = 1u << 0;
inline constexpr uint32_t kPagedCandidatePoolExhausted = 1u << 1;
inline constexpr uint32_t kPagedCandidateLogicalOverflow = 1u << 2;
inline constexpr uint32_t kPooledSegmentLaneOverflow = 1u << 3;

// Candidate ABI: a six-byte global record. cand_val stores the
// low 16 bits of an ascending, sign-aware FP32 high24 score code; cand_idx
// stores its high eight score bits above the exact 20-bit KV index. The local
// ring uses one uint32_t containing high24 score bits and an eight-bit
// block-in-window.
using CandidateValue = uint16_t;
constexpr uint32_t kCandidateIndexBits = 20;
constexpr uint32_t kCandidateIndexMask = (1u << kCandidateIndexBits) - 1u;

CUTLASS_DEVICE uint32_t candidate_pack_index(const uint32_t payload,
                                             const uint32_t kv_index) {
  return (kv_index & kCandidateIndexMask) |
         ((payload >> 16) << kCandidateIndexBits);
}

CUTLASS_DEVICE uint32_t candidate_ordered_fp32_code(const float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

CUTLASS_DEVICE uint32_t candidate_fp24_code(const float value) {
  return candidate_ordered_fp32_code(value) >> 8;
}

// Full-width counterpart of the dense selector's coarse ordering. Smaller
// unsigned keys correspond to larger FP32 values.
CUTLASS_DEVICE uint32_t candidate_descending_fp32_code(const float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? bits : (~bits & 0x7fffffffu);
}

// High eight bits of the sign-aware FP16 key used by the dense selector.
// Keeping 256 coarse buckets makes the online histogram small enough to live
// beside the score pipeline while retaining much more resolution than the
// sign/exponent byte of a raw FP32 key.
CUTLASS_DEVICE uint32_t candidate_fixed_half_bucket(const float value) {
  const uint32_t fp32_bits = __float_as_uint(value);
  const __half half_value = __float2half(value);
  uint16_t bits = __half_as_ushort(half_value) & 0x7fffu;
  bits |= static_cast<uint16_t>((fp32_bits >> 16) & 0x8000u);
  bits = (bits & 0x8000u) ? bits : static_cast<uint16_t>(~bits & 0x7fffu);
  return static_cast<uint32_t>(bits >> 8);
}

struct OnlineFixedCandidate {
  uint32_t bucket;
  uint32_t payload;
};

CUTLASS_DEVICE OnlineFixedCandidate candidate_online_fixed(const float value) {
  const uint32_t bucket = candidate_fixed_half_bucket(value);
  const uint32_t raw_desc24 = candidate_descending_fp32_code(value) >> 8;
  // Six-byte payload: the coarse bucket is explicit, while
  // the remaining 16 bits retain the raw-score order inside that bucket.
  return {bucket, (bucket << 16) | (raw_desc24 & 0xffffu)};
}

CUTLASS_DEVICE uint32_t candidate_load_score_code(const CandidateValue value,
                                                  const int32_t packed_idx) {
  return (static_cast<uint32_t>(packed_idx) >> kCandidateIndexBits) << 16 |
         static_cast<uint32_t>(value);
}

CUTLASS_DEVICE float candidate_decode_score(const CandidateValue value,
                                            const int32_t packed_idx) {
  const uint32_t code = candidate_load_score_code(value, packed_idx);
  const uint32_t ordered = code << 8;
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
}

CUTLASS_DEVICE int32_t candidate_decode_index(const int32_t packed_idx) {
  return static_cast<int32_t>(static_cast<uint32_t>(packed_idx) &
                              kCandidateIndexMask);
}

CUTLASS_DEVICE void store_candidate(CandidateValue* value_dst,
                                    int32_t* index_dst, const float bq,
                                    const uint32_t kv_index) {
  const uint32_t payload = candidate_fp24_code(bq);
  __stcs(value_dst, static_cast<CandidateValue>(payload));
  __stcs(index_dst,
         static_cast<int32_t>(candidate_pack_index(payload, kv_index)));
}

CUTLASS_DEVICE void store_candidate_payload(CandidateValue* value_dst,
                                            int32_t* index_dst,
                                            const uint32_t payload,
                                            const uint32_t kv_index) {
  __stcs(value_dst, static_cast<CandidateValue>(payload));
  __stcs(index_dst,
         static_cast<int32_t>(candidate_pack_index(payload, kv_index)));
}

CUTLASS_DEVICE void store_candidate_record(CandidateValue* value_dst,
                                           int32_t* index_dst,
                                           const uint32_t payload,
                                           const uint32_t kv_index) {
  store_candidate_payload(value_dst, index_dst, payload, kv_index);
}

CUTLASS_DEVICE void paged_candidate_set_status(int32_t* pool_status,
                                               const uint32_t status) {
  if (pool_status != nullptr) {
    atomicOr(reinterpret_cast<unsigned int*>(pool_status),
             static_cast<unsigned int>(status));
  }
}

static __device__ __noinline__ int32_t paged_candidate_ensure_page(
    const uint32_t row_q, const uint32_t page_ordinal, const uint32_t cand_cap,
    int32_t* page_table, uint32_t* pool_head, int32_t* pool_status,
    const uint32_t pool_pages) {
  const uint32_t pages_per_row = math::ceil_div(
      cand_cap - kPagedCandidateInlineRecords, kPagedCandidatePageRecords);
  int32_t* const entry = page_table + row_q * pages_per_row + page_ordinal;
  int32_t page = atomicCAS(entry, kPagedCandidateEmpty, kPagedCandidateBusy);
  if (page == kPagedCandidateEmpty) {
    const uint32_t allocated = atomicAdd(pool_head, 1u);
    if (allocated >= pool_pages) {
      paged_candidate_set_status(pool_status, kPagedCandidatePoolExhausted);
      atomicExch(entry, kPagedCandidateFailed);
      return kPagedCandidateFailed;
    }
    page = static_cast<int32_t>(allocated);
    atomicExch(entry, page);
    return page;
  }
  while (page == kPagedCandidateBusy) {
    __nanosleep(64);
    page = atomicCAS(entry, kPagedCandidateBusy, kPagedCandidateBusy);
  }
  return page;
}

struct PagedCandidateReservation {
  int32_t page_ordinal0;
  int32_t page0;
  int32_t page1;
};

// The ordinary reservation stays wholly inside the fixed 8K prefix. Keep all
// page-table work out of that path; this cold helper handles the one
// reservation that crosses the prefix and the still rarer page boundary.
static __device__ __noinline__ PagedCandidateReservation
paged_candidate_prepare_reservation(const uint32_t row_q, const int row_base,
                                    const uint32_t row_total,
                                    const uint32_t cand_cap,
                                    int32_t* page_table, uint32_t* pool_head,
                                    int32_t* pool_status,
                                    const uint32_t pool_pages) {
  PagedCandidateReservation reservation = {-1, -1, -1};
  const uint32_t reservation_end = static_cast<uint32_t>(row_base) + row_total;
  if (reservation_end > cand_cap) {
    paged_candidate_set_status(pool_status, kPagedCandidateLogicalOverflow);
  }
  const uint32_t overflow_begin =
      max(static_cast<uint32_t>(row_base), kPagedCandidateInlineRecords);
  const uint32_t overflow_end = static_cast<uint32_t>(
      reservation_end < cand_cap ? reservation_end : cand_cap);
  if (overflow_begin >= overflow_end) {
    return reservation;
  }

  reservation.page_ordinal0 =
      static_cast<int32_t>((overflow_begin - kPagedCandidateInlineRecords) /
                           kPagedCandidatePageRecords);
  const int32_t last_page_ordinal =
      static_cast<int32_t>((overflow_end - 1 - kPagedCandidateInlineRecords) /
                           kPagedCandidatePageRecords);
  reservation.page0 = paged_candidate_ensure_page(
      row_q, static_cast<uint32_t>(reservation.page_ordinal0), cand_cap,
      page_table, pool_head, pool_status, pool_pages);
  if (last_page_ordinal != reservation.page_ordinal0) {
    reservation.page1 = paged_candidate_ensure_page(
        row_q, static_cast<uint32_t>(last_page_ordinal), cand_cap, page_table,
        pool_head, pool_status, pool_pages);
  }
  return reservation;
}

static __device__ __noinline__ void paged_candidate_store_prepared_span(
    const uint32_t* emit_lane_records, const int copy_count,
    const uint32_t logical_out_base, const uint32_t emit_window_kv_base,
    const uint32_t probe_group, const uint64_t probe_magic,
    const uint32_t probe_add_max, CandidateValue* row_cand_val,
    int32_t* row_cand_idx, CandidateValue* overflow_val, int32_t* overflow_idx,
    const int32_t page_ordinal0, const int32_t page0, const int32_t page1) {
  const uint32_t math_thread_idx = threadIdx.x;
#pragma unroll 1
  for (int slot = 0; slot < copy_count; ++slot) {
    const uint32_t record = emit_lane_records[slot * 32];
    const uint32_t payload = (record & 0xffffff00u) >> 8;
    uint32_t kv_index =
        emit_window_kv_base + ((record & 0xffu) * 256u) + math_thread_idx;
    if (probe_group != 0) {
      const uint32_t sup = static_cast<uint32_t>(
          (static_cast<uint64_t>(kv_index) * probe_magic) >> 42);
      kv_index += min((sup + 1) * 64u, probe_add_max);
    }

    const uint32_t logical_out = logical_out_base + slot;
    if (logical_out < kPagedCandidateInlineRecords) {
      store_candidate_record(&row_cand_val[logical_out],
                             &row_cand_idx[logical_out], payload, kv_index);
      continue;
    }

    const uint32_t overflow_out = logical_out - kPagedCandidateInlineRecords;
    const int32_t page_ordinal =
        static_cast<int32_t>(overflow_out / kPagedCandidatePageRecords);
    const int32_t page =
        page_ordinal == page_ordinal0
            ? page0
            : (page_ordinal == page_ordinal0 + 1 ? page1
                                                 : kPagedCandidateFailed);
    if (page >= 0) {
      const uint64_t out_pos =
          static_cast<uint64_t>(page) * kPagedCandidatePageRecords +
          overflow_out % kPagedCandidatePageRecords;
      store_candidate_record(&overflow_val[out_pos], &overflow_idx[out_pos],
                             payload, kv_index);
    }
  }
}

// A lane exceeding its local-ring quota is already a cold path. Keeping its
// address translation, probe remap, and allocation here avoids cloning them
// into the scan kernel's instruction working set.
static __device__ __noinline__ void paged_candidate_store_lane_overflow(
    const uint32_t row_q, const int logical_out, const uint32_t cand_cap,
    const uint32_t payload, uint32_t kv_index, const uint32_t probe_group,
    const uint64_t probe_magic, const uint32_t probe_add_max,
    CandidateValue* cand_val, int32_t* cand_idx, CandidateValue* overflow_val,
    int32_t* overflow_idx, int32_t* page_table, uint32_t* pool_head,
    int32_t* pool_status, const uint32_t pool_pages) {
  if (logical_out >= static_cast<int>(cand_cap)) {
    paged_candidate_set_status(pool_status, kPagedCandidateLogicalOverflow);
    return;
  }
  if (probe_group != 0) {
    const uint32_t sup = static_cast<uint32_t>(
        (static_cast<uint64_t>(kv_index) * probe_magic) >> 42);
    kv_index += min((sup + 1) * 64u, probe_add_max);
  }

  const uint32_t out = static_cast<uint32_t>(logical_out);
  if (out < kPagedCandidateInlineRecords) {
    const uint64_t out_pos =
        static_cast<uint64_t>(row_q) * kPagedCandidateInlineRecords + out;
    store_candidate_record(&cand_val[out_pos], &cand_idx[out_pos], payload,
                           kv_index);
    return;
  }

  const uint32_t overflow_out = out - kPagedCandidateInlineRecords;
  const uint32_t page_ordinal = overflow_out / kPagedCandidatePageRecords;
  const int32_t page =
      paged_candidate_ensure_page(row_q, page_ordinal, cand_cap, page_table,
                                  pool_head, pool_status, pool_pages);
  if (page >= 0) {
    const uint64_t out_pos =
        static_cast<uint64_t>(page) * kPagedCandidatePageRecords +
        overflow_out % kPagedCandidatePageRecords;
    store_candidate_record(&overflow_val[out_pos], &overflow_idx[out_pos],
                           payload, kv_index);
  }
}

static __device__ __noinline__ uint32_t pooled_segment_allocate_page(
    uint32_t* pool_head, int32_t* pool_status, const uint32_t pool_pages) {
  const uint32_t page = atomicAdd(pool_head, 1u);
  if (page >= pool_pages) {
    paged_candidate_set_status(pool_status, kPagedCandidatePoolExhausted);
    return kPooledSegmentInvalidStart;
  }
  return page;
}

// The ordinary writer keeps 18 records per lane in its shared ring.  A
// skewed lane may exceed that quota even though the row remains sparse.  The
// overflow path is deliberately outlined: it appends into two dedicated 4K
// pages for this (warp,row,window), leaving the common ring path unchanged.
// A 256-block window has at most 32*(256-18)=7616 overflow records, so two
// pages are an exact, distribution-independent bound.
static __device__ __noinline__ void pooled_segment_store_lane_spill(
    const uint32_t row_q, const uint32_t payload, uint32_t kv_index,
    const uint32_t probe_group, const uint64_t probe_magic,
    const uint32_t probe_add_max, uint32_t* spill_count,
    int32_t* spill_page_ids, CandidateValue* segment_val, int32_t* segment_idx,
    uint32_t* pool_head, int32_t* pool_status, const uint32_t pool_pages,
    const uint32_t micro_records) {
  const uint32_t slot = atomicAdd(spill_count, 1u);
  const uint32_t ordinal = slot / kPooledSegmentPageRecords;
  if (ordinal >= kPooledSegmentSpillPages) {
    paged_candidate_set_status(pool_status, kPooledSegmentLaneOverflow);
    return;
  }

  int32_t* const entry = spill_page_ids + ordinal;
  int32_t page = atomicCAS(entry, kPagedCandidateEmpty, kPagedCandidateBusy);
  if (page == kPagedCandidateEmpty) {
    const uint32_t allocated = atomicAdd(pool_head, 1u);
    if (allocated >= pool_pages) {
      paged_candidate_set_status(pool_status, kPagedCandidatePoolExhausted);
      atomicExch(entry, kPagedCandidateFailed);
      return;
    }
    page = static_cast<int32_t>(allocated);
    atomicExch(entry, page);
  } else {
    while (page == kPagedCandidateBusy) {
      __nanosleep(64);
      page = atomicCAS(entry, kPagedCandidateBusy, kPagedCandidateBusy);
    }
    if (page < 0) return;
  }

  if (probe_group != 0) {
    const uint32_t sup = static_cast<uint32_t>(
        (static_cast<uint64_t>(kv_index) * probe_magic) >> 42);
    kv_index += min((sup + 1) * 64u, probe_add_max);
  }
  const uint64_t physical =
      static_cast<uint64_t>(micro_records) +
      static_cast<uint64_t>(page) * kPooledSegmentPageRecords +
      slot % kPooledSegmentPageRecords;
  store_candidate_record(&segment_val[physical], &segment_idx[physical],
                         payload, kv_index);
}

CUTLASS_DEVICE uint64_t pooled_segment_pack_descriptor(
    const uint32_t physical_start, const uint32_t count) {
  return static_cast<uint64_t>(physical_start) |
         (static_cast<uint64_t>(count) << 32);
}

template <uint32_t Q> struct LaneCounts {
  uint32_t words[(Q + 3) / 4] = {};
  CUTLASS_DEVICE uint32_t get(uint32_t i) const {
    return (words[i / 4] >> ((i % 4) * 8)) & 255u;
  }
  CUTLASS_DEVICE void add(uint32_t i) { words[i / 4] += 1u << ((i % 4) * 8); }
  CUTLASS_DEVICE void clear() {
    #pragma unroll
    for (uint32_t i=0; i<(Q+3)/4; ++i) words[i]=0;
  }
};
#include "flush.cuh"

template <uint32_t kBlockQ>
struct RawDumpGate {
  float g[kBlockQ];
};

// Keep sentinel refill, epoch publication, and gate reload outside the
// scoring loop. This work runs once per window. Return the refreshed gate
// by value so the caller can keep o_reg in registers.
template <uint32_t kBlockQ>
static __device__ __noinline__ RawDumpGate<kBlockQ> raw_dump_ring_flush_tail(
    uint32_t* ring_warp_base,   // emit_smem_records + warp block
    uint32_t* ring_seq_warp,    // ring_seq + warp_idx * kBlockQ
    const uint32_t* gate_bits,  // CTA gate mailbox
    const uint32_t lane_idx, const LaneCounts<kBlockQ>& emit_lane_counts) {
// The bulk wait_group.read has completed, so refilling sentinels cannot
// race the dump. Same epoch discipline as the drain path: refill exactly
// the [0, count) slots this window wrote, then publish the flush epoch so
// the daemon replays the column from zero.
#pragma unroll
  for (uint32_t i = 0; i < kBlockQ; ++i) {
    const uint32_t count = emit_lane_counts.get(i);
    for (uint32_t slot = 0; slot < count; ++slot) {
      ring_warp_base[(i * kEmitLaneSlots + slot) * 32u + lane_idx] =
          kRingSentinel;
    }
  }
  __syncwarp();
  __threadfence_block();
  if (lane_idx < kBlockQ) {
    ring_seq_warp[lane_idx] += 1u;
  }
  __threadfence_block();
  // Window-granular gate uptake: the bulk copy is count-insensitive,
  // so per-block freshness buys nothing under raw dump; reading the
  // mailbox once per window keeps the daemon's tightening flowing to
  // later windows for free.
  RawDumpGate<kBlockQ> out;
  if constexpr (kBlockQ == 4) {
    const float4 gate =
        ptx::ld_shared(reinterpret_cast<const float4*>(gate_bits));
    out.g[0] = gate.x;
    out.g[1] = gate.y;
    out.g[2 < kBlockQ ? 2 : 0] = gate.z;
    out.g[3 < kBlockQ ? 3 : 0] = gate.w;
  } else {
#pragma unroll
    for (uint32_t gi = 0; gi < kBlockQ; ++gi)
      out.g[gi] = __uint_as_float(ptx::ld_shared(gate_bits + gi));
  }
  return out;
}

// Raw-dump emit: at each window boundary one elected lane bulk-copies the
// warp's whole staged ring block (BLOCK_Q rows x kEmitLaneSlots x 32 lanes
// x 4B) to a deterministic global region — no output reservation atomic, no
// per-record drain. The generic-proxy ring stores must be fenced into the
// async proxy before the bulk read, and the wait releases the smem block
// for the next window's appends.
CUTLASS_DEVICE void raw_dump_emit_block(const uint32_t* smem_block,
                                        uint32_t* gmem_region,
                                        const uint32_t bytes) {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_block));
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(
          gmem_region),
      "r"(smem_addr), "r"(bytes)
      : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_Q,
          uint32_t BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs, uint32_t kNumSpecializedThreads,
          uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128,
          bool kOnlineFixedBuckets = false, bool kStaticHotGate = false,
          bool kStaticHotNoHist = false, bool kStaticHotExactGate = false,
          bool kPagedCandidates = false, bool kPooledSegments = false,
          bool kRingRefresh = false, bool kRawDumpEmit = false,
          uint32_t kEmitLaneSlots = 18, uint32_t kTuneGateStride = 8,
          uint32_t kTuneMathRegisters = 240, uint32_t kTuneEmitBlocks = 256,
#if defined(LISA_SCAN_RECOVERY)
          uint32_t kTuneRingNs = 2048, bool kAggregateSpills = false, uint32_t kFlushMode = 0, uint32_t kMinBlocks = 1>
#else
          uint32_t kTuneRingNs = 2048, bool kAggregateSpills = false, uint32_t kFlushMode = 0, uint32_t kMinBlocks = 1, uint32_t kTuneBits = 0>
#endif
CUTLASS_GLOBAL __launch_bounds__(
    kNumSpecializedThreads + kNumMathThreads,
    kMinBlocks) void sm100_dsa_litetopk(const uint32_t seq_len,
                               const uint32_t seq_len_kv,
                               uint32_t* cu_seq_len_k_start,
                               uint32_t* cu_seq_len_k_end,
                               const float* __restrict__ origin,  // [seq_len]
                               const float* __restrict__ inv_delta,  // [seq_len]
                               int32_t* __restrict__ th_bucket,  // [seq_len]
                               int32_t* __restrict__ bcount,     // [seq_len,
                                                                 // num_buckets]
                               const uint32_t num_buckets, const uint32_t topk,
                               const uint32_t refresh_every,
                               const uint32_t num_kv_splits,
                               const uint32_t
                                   probe_group,  // compacted-space group size
                                                 // (pstp-1)*64; 0 = no probe
                                                 // compaction (identity map)
                               const uint64_t
                                   probe_magic,  // ceil(2^42/probe_group):
                                                 // exact div via mul-shift
                               const uint32_t
                                   probe_add_max,  // npage*64 cap for the map
                               CandidateValue* __restrict__ cand_val,  // [seq_len,
                                                                       // cand_cap]
                               int32_t* __restrict__ cand_idx,  // [seq_len,
                                                                // cand_cap]
                               int32_t* __restrict__ cand_cnt,  // [seq_len]
                               const uint32_t cand_cap,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_q,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_kv,
                               // fp8: per-token fp32 dequant scales. fp4: the
                               // UE8M0 SF-KV int32 stream (same 1-D shape).
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_kv_scales,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_weights,
                               // fp4 only: per-(token, head) packed UE8M0 SF-Q
                               // int32 stream; fp8 callers pass any map
                               // (unread).
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_sf_q,
                               CandidateValue* __restrict__ overflow_val =
                                   nullptr,
                               int32_t* __restrict__ overflow_idx = nullptr,
                               int32_t* __restrict__ page_table = nullptr,
                               uint32_t* __restrict__ pool_head = nullptr,
                               int32_t* __restrict__ pool_status = nullptr,
                               const uint32_t pool_pages = 0,
                               CandidateValue* __restrict__ segment_val =
                                   nullptr,
                               int32_t* __restrict__ segment_idx = nullptr,
                               // [seq_len, segment_desc_windows, 8]. Descriptor
                               // low32=physical_start, high32=count. The
                               // physical record space starts with
                               // ceil(seq_len/4)*8*512 deterministic
                               // micro-slabs, followed by segment_pool_pages
                               // 4K-record pages.
                               uint64_t* __restrict__ segment_desc = nullptr,
                               uint32_t* __restrict__ segment_pool_head =
                                   nullptr,
                               int32_t* __restrict__ segment_status = nullptr,
                               const uint32_t segment_pool_pages = 0,
                               const uint32_t segment_desc_windows = 0,
                               // [seq_len, segment_desc_windows, 8, 2]. Each
                               // nonempty entry owns one complete 4K pool page.
                               uint64_t* __restrict__ segment_spill_desc =
                                   nullptr,
                               int32_t* __restrict__ segment_row_spill =
                                   nullptr,
                               // Raw-dump emit: [num_q_blocks, 8,
                               // dump_max_windows] regions of
                               // BLOCK_Q*kEmitLaneSlots*32 uint32 ring words,
                               // and one packed per-lane count word per region
                               // ([.., 32] uint32). Validity is bounded by the
                               // counts; slots past a lane's count are stale.
                               uint32_t* __restrict__ dump_slab = nullptr,
                               uint32_t* __restrict__ dump_counts = nullptr,
                               const uint32_t dump_max_windows = 0) {
  static_assert(!(kPagedCandidates && kPooledSegments),
                "paged rows and pooled segments are separate A/B writers");
  static_assert(!kRawDumpEmit || (kStaticHotNoHist && !kStaticHotExactGate &&
                                  !kPagedCandidates && !kPooledSegments &&
                                  !kOnlineFixedBuckets),
                "raw-dump emit covers only the bucket-gate no-hist writer "
                "(plain or ring-refresh)");
  static_assert(!kPooledSegments || !kOnlineFixedBuckets,
                "pooled descriptors use fixed 256-block windows");
  static_assert(!kPooledSegments || kStaticHotNoHist,
                "pooled spill metadata reuses the no-hist shared-memory tail");
  if constexpr (kPagedCandidates) {
    const bool valid =
        cand_val != nullptr && cand_idx != nullptr && overflow_val != nullptr &&
        overflow_idx != nullptr && page_table != nullptr &&
        pool_head != nullptr && pool_status != nullptr && pool_pages != 0 &&
        cand_cap >= kPagedCandidateInlineRecords && num_kv_splits == 1;
    if (!valid) {
      if (threadIdx.x == 0) {
        paged_candidate_set_status(pool_status, kPagedCandidateBadArguments);
      }
      return;
    }
  }
  if constexpr (kPooledSegments) {
    const uint32_t required_windows =
        math::ceil_div(math::ceil_div(seq_len_kv, BLOCK_KV), kEmitChunkBlocks);
    const uint64_t num_q_blocks_u64 = math::ceil_div(seq_len, BLOCK_Q);
    const uint64_t micro_records =
        num_q_blocks_u64 * 8u * kPooledSegmentMicroRecords;
    const uint64_t physical_records =
        micro_records +
        static_cast<uint64_t>(segment_pool_pages) * kPooledSegmentPageRecords;
    const bool valid =
        segment_val != nullptr && segment_idx != nullptr &&
        segment_desc != nullptr && segment_spill_desc != nullptr &&
        segment_row_spill != nullptr && segment_pool_head != nullptr &&
        segment_status != nullptr && segment_pool_pages != 0 &&
        segment_desc_windows >= required_windows &&
        physical_records <= kPooledSegmentInvalidStart && num_kv_splits == 1;
    if (!valid) {
      if (threadIdx.x == 0) {
        paged_candidate_set_status(segment_status, kPagedCandidateBadArguments);
      }
      return;
    }
  }
  const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
  constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
  constexpr uint32_t kNumUmmaStages = kUmmaStages;
  constexpr uint32_t kNumUmmaBuffers = kNumMathWarpGroups * kNumUmmaStages;
  // fp4 replaces the per-warpgroup UMMA double buffers with a shared
  // 3-stage TMEM accumulator pool (the SF columns take the freed space);
  constexpr uint32_t kNumAccumBufs = kNumUmmaBuffers;
  // The ring refresher reloads more often: a flooding row keeps emitting
  // at the stale gate for a full stride, so the flood integral scales with
  // the reload cadence. The reload is one ld.shared.v4 per stride.
  // Raw-dump emit is insensitive to per-window record counts (the bulk
  // copy is fixed-size), so the flood integral no longer justifies the
  // tight cadence there and the default stride applies.
  constexpr uint32_t kActiveGateStride =
      kOnlineFixedBuckets
          ? 4u
          : ((kRingRefresh && !kRawDumpEmit) ? kTuneGateStride : kGateStride);
  constexpr uint32_t kActiveEmitChunkBlocks =
      kOnlineFixedBuckets ? 32u : kTuneEmitBlocks;
  static_assert(
      !(kOnlineFixedBuckets && kStaticHotGate),
      "online fixed buckets and static HOT gate cannot be combined");
  static_assert(!kStaticHotNoHist || kStaticHotGate,
                "static HOT no-hist requires the static HOT gate");
  static_assert(!kStaticHotExactGate || kStaticHotNoHist,
                "exact FP24 gate requires the static HOT no-hist path");
  static_assert(
      !kRingRefresh ||
          (kStaticHotNoHist && !kStaticHotExactGate && !kPagedCandidates &&
           !kPooledSegments && !kOnlineFixedBuckets),
      "ring refresher v1 covers only the plain bucket-gate no-hist writer");
  DG_STATIC_ASSERT(
#if defined(LISA_SCAN_RECOVERY)
      (BLOCK_Q == 4 || BLOCK_Q == 8 || BLOCK_Q == 16) && (kNumMathWarps == 4 || kNumMathWarps == 8),
#elif defined(LISA_SCAN_STATIC)
      (BLOCK_Q == 4 || BLOCK_Q == 8 || BLOCK_Q == 16 || BLOCK_Q == 32) && (kNumMathWarps == 4 || kNumMathWarps == 8 || kNumMathWarps == 16),
#else
      (BLOCK_Q == 4 || BLOCK_Q == 8 || BLOCK_Q == 16 || BLOCK_Q == 32) && (kNumMathWarps == 4 || kNumMathWarps == 8),
#endif
      "LiteTopK requires BLOCK_Q in {4 (H=32), 2 (H=64)} and 8 math warps");
  DG_STATIC_ASSERT(
#if defined(LISA_SCAN_STATIC)
      (BLOCK_KV == 128 || BLOCK_KV == 256 || BLOCK_KV == 512),
#else
      (BLOCK_KV == 128 || BLOCK_KV == 256),
#endif
      "the local ring infers the KV tile coordinate from its owner lane");
  static_assert(32u * kEmitLaneSlots <= kPagedCandidatePageRecords,
                "one warp-row reservation must intersect at most two pages");
  static_assert(!kPooledSegments || BLOCK_Q * 32u * kEmitLaneSlots <= kPooledSegmentPageRecords,
                "one four-row warp flush must fit in one pooled page");
  static_assert(
      32u * (kEmitChunkBlocks - kEmitLaneSlots) <=
          kPooledSegmentSpillPages * kPooledSegmentPageRecords,
      "two pooled spill pages must cover a worst-case warp-row window");

  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0,
                   "Invalid threads");

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
  }

  // fp4 packs two e2m1 elements per byte; SF regions are UE8M0 bytes
  // packed 4-per-int32 and padded to the 128-element UTCCP granule.
  static constexpr uint32_t kKvElemBytesNum = 2u;  // per 2 elems
  static constexpr uint32_t kNumUTCCPAlignedElems = 128;
  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * kHeadDim * kKvElemBytesNum / 2;
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      math::constexpr_align(BLOCK_Q * 4u * uint32_t(sizeof(float)), 512u);
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE =
      BLOCK_KV * kHeadDim * kKvElemBytesNum / 2;
  static constexpr uint32_t kNumSFQ =
      math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
  static constexpr uint32_t kNumSFKV =
      math::constexpr_align(BLOCK_KV, kNumUTCCPAlignedElems);
  // fp8: one fp32 dequant scale per token. fp4: one int32 (4x UE8M0) per
  // token, padded to the UTCCP granule.
  static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE =
      BLOCK_KV * sizeof(float);
  static constexpr uint32_t SMEM_SF_Q_SIZE = 0u;
  static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE =
      math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

  extern __shared__ __align__(512) uint8_t smem_buffer[];
  DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0,
                   "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0,
                   "Unaligned TMA swizzling");

#if defined(LISA_SCAN_RECOVERY)
  constexpr uint32_t kAccumStride = 32;
#else
  constexpr uint32_t kAccumStride = BLOCK_Q * kNumHeads > 32 ? BLOCK_Q * kNumHeads : 32;
#endif
  constexpr uint32_t kNumAccumTmemCols = kAccumStride * kNumAccumBufs;
  constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
  constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
  constexpr uint32_t kNumTmemCols = kNumAccumTmemCols;
  DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

  auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
                                            SMEM_Q_SIZE_PER_STAGE * i);
  });
  auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<float*>(smem_buffer +
                                    SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                                    SMEM_WEIGHT_SIZE_PER_STAGE * i);
  });
  auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<__nv_fp8_e4m3*>(
        smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                       SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
                       SMEM_KV_SIZE_PER_STAGE * i));
  });
  auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<float*>(smem_buffer +
                                    SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                                    SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
                                    SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
                                    ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
  });

  auto smem_sf_q = reinterpret_cast<uint32_t*>(
      reinterpret_cast<uint8_t*>(smem_kv_scales[kNumKVStages]));
  auto barrier_ptr = reinterpret_cast<Barrier*>(
      reinterpret_cast<uint8_t*>(smem_sf_q) + SMEM_SF_Q_SIZE);
  auto full_q_barriers =
      utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
  auto empty_q_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
  auto full_kv_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
  auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i);
  });
  auto full_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i);
  });
  auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr +
           (kNumQStages * 2 + kNumKVStages * 2 + kNumAccumBufs + i);
  });

  auto sf_ready_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr +
           (kNumQStages * 2 + kNumKVStages * 2 + kNumAccumBufs * 2 + i);
  });
  auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
      barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumAccumBufs * 2 +
      kNumKVStages);
  auto scan_done_flag = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 1);
  auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 4);
  // Chunked emit leaves the warp-queue count words unused. Reuse four
  // of them as a CTA-local Gate4 mailbox. Values are positive float edge
  // bit-patterns, so unsigned atomicMin is exactly a monotonic gate tighten.
  auto sparse_gate_bits = reinterpret_cast<uint32_t*>(warpq_count);
  auto emit_smem_records =
      reinterpret_cast<uint32_t*>(warpq_count + kNumMathWarps * BLOCK_Q);
  auto smem_hist = reinterpret_cast<int32_t*>(
      emit_smem_records + kNumMathWarps * BLOCK_Q * kEmitLaneSlots * 32u);
  // The pooled writer is a no-hist specialization, so this tail is free.
  // Each warp owns four counters and two page ids per row.
  auto pooled_spill_count = reinterpret_cast<uint32_t*>(smem_hist);
  auto pooled_spill_page_ids =
      reinterpret_cast<int32_t*>(pooled_spill_count + kNumMathWarps * BLOCK_Q);
  // Ring-refresher scratch sits after the refresher's histogram (which
  // reuses the dynamic-path histogram region): per-(math warp, row) flush
  // sequence words, the spare warps' shadow copies, and per-lane
  // consumed-slot cursors.
  auto ring_seq = reinterpret_cast<uint32_t*>(smem_hist + BLOCK_Q * 256);
  auto ring_shadow = ring_seq + kNumMathWarps * BLOCK_Q;
  auto ring_pending =
      reinterpret_cast<int32_t*>(ring_shadow + kNumMathWarps * BLOCK_Q);
  auto ring_stale = ring_pending + BLOCK_Q;
  auto ring_last = ring_stale + BLOCK_Q;
  auto ring_progress = reinterpret_cast<uint8_t*>(ring_last + BLOCK_Q);

  DG_STATIC_ASSERT(
      kNumSpecializedThreads % 128 == 0 and kNumSpecializedThreads >= 64,
      "Invalid threads");
  if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      full_q_barriers[i]->init(1);
      empty_q_barriers[i]->init(kNumMathThreads + 32);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      full_kv_barriers[i]->init(1);
#if defined(LISA_SCAN_STATIC) || defined(LISA_SCAN_FINAL)
      empty_kv_barriers[i]->init((kTuneBits & 16) ? kNumMathWarps : kNumMathThreads);
#else
      empty_kv_barriers[i]->init(kNumMathThreads);
#endif
    }
    if constexpr (!kStaticHotNoHist || kRingRefresh) *scan_done_flag = 0;
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 1) {
    if (cute::elect_one_sync()) {
#pragma unroll
      for (uint32_t i = 0; i < kNumAccumBufs; ++i) {
        full_umma_barriers[i]->init(1);
#if defined(LISA_SCAN_STATIC) || defined(LISA_SCAN_FINAL)
        empty_umma_barriers[i]->init((kTuneBits & 16) ? 4 : 128);
#else
        empty_umma_barriers[i]->init(128);
#endif
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    if constexpr (kMinBlocks > 1) cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  // One CTA owns each row. The hot-only path emits no sample seeds and scans
  // the complete KV range, so its exact histogram starts at zero in SMEM.
  // kStaticHotNoHist skips the histogram clear and updates;
  // a post-scan kernel rebuilds metadata from the compact candidate slab.
  if constexpr (!kStaticHotNoHist) {
    for (uint32_t idx = threadIdx.x; idx < BLOCK_Q * num_buckets;
         idx += blockDim.x) {
      smem_hist[idx] = 0;
    }
  }
  if constexpr (kRingRefresh) {
    for (uint32_t idx = threadIdx.x;
         idx < kNumMathWarps * BLOCK_Q * kEmitLaneSlots * 32u;
         idx += blockDim.x) {
      emit_smem_records[idx] = kRingSentinel;
    }
    // Preload the seed sample's full bucket histogram as the daemon's
    // refresh base. The sample records are genuine row records in the
    // same (origin, inv) bucket space and the exact-once scan starts after
    // the sampled prefix, so ring harvests never recount them; base plus
    // harvested stays a subset of the row's true records and every
    // publish remains a provable upper bound.
    for (uint32_t idx = threadIdx.x; idx < BLOCK_Q * num_buckets;
         idx += blockDim.x) {
      const uint32_t r = idx / num_buckets;
      const uint32_t row_q = static_cast<uint32_t>(blockIdx.x) * BLOCK_Q + r;
      smem_hist[idx] =
          row_q < seq_len
              ? __ldcg(bcount + static_cast<uint64_t>(row_q) * num_buckets +
                       (idx - r * num_buckets))
              : 0;
    }
    for (uint32_t idx = threadIdx.x; idx < kNumMathWarps * BLOCK_Q;
         idx += blockDim.x) {
      ring_seq[idx] = 0u;
      ring_shadow[idx] = 0u;
    }
    for (uint32_t idx = threadIdx.x; idx < BLOCK_Q; idx += blockDim.x) {
      ring_pending[idx] = 0;
      ring_last[idx] = 0;
      ring_stale[idx] = blockIdx.x * BLOCK_Q + idx < seq_len ? 0 : 2;
    }
    for (uint32_t idx = threadIdx.x; idx < kNumMathWarps * BLOCK_Q * 32u;
         idx += blockDim.x) {
      ring_progress[idx] = 0u;
    }
  }
  if (threadIdx.x < BLOCK_Q) {
    const uint32_t row_q =
        static_cast<uint32_t>(blockIdx.x) * BLOCK_Q + threadIdx.x;
    const uint32_t row = min(row_q, seq_len - 1);
    const int gate = __ldcg(th_bucket + row);
    if constexpr (kStaticHotExactGate) {
      // Exact compaction stores the ordered-FP32 pivot edge in this
      // int32 slot. Preserve its bits so the unsigned strict comparison
      // admits only later records from a better high24 class.
      ptx::st_shared(sparse_gate_bits + threadIdx.x,
                     static_cast<uint32_t>(gate));
    } else {
      ptx::st_shared(sparse_gate_bits + threadIdx.x,
                     __float_as_uint(static_cast<float>(gate + 1)));
    }
  }
  __syncthreads();

  constexpr uint32_t kNumSpecializedRegisters = kSpecializedRegisters;
  constexpr uint32_t kNumMathRegisters = kTuneMathRegisters;

  // KV-split scheduling: blockIdx.x = q-block (one per CTA), blockIdx.y =
  // contiguous KV sub-window. Split boundaries are BLOCK_KV-aligned.
  const uint32_t block_q_idx = blockIdx.x;
  const uint32_t kv_split = blockIdx.y;
  uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
  const auto load_schedule =
      [&](const uint32_t block_q_idx) -> cute::tuple<uint32_t, uint32_t> {
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();

#pragma unroll
    for (uint32_t i = 0; i < BLOCK_Q; ++i) {
      const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
      seq_k_start[i] = cu_seq_len_k_start[q_idx];
      seq_k_end[i] = cu_seq_len_k_end[q_idx];
      if (block_q_idx * BLOCK_Q + i >= seq_len) {
        // Padded row of a ragged final q-block: empty, aggregation-neutral.
        seq_k_start[i] = seq_len_kv;
        seq_k_end[i] = 0;
      }
      start = min(start, min(seq_k_start[i], seq_len_kv));
      end = max(end, min(seq_k_end[i], seq_len_kv));
    }
    const uint32_t total_blocks = math::ceil_div(seq_len_kv, BLOCK_KV);
    const uint32_t blocks_per_split =
        math::ceil_div(total_blocks, num_kv_splits);
    const uint32_t split_lo = kv_split * blocks_per_split * BLOCK_KV;
    const uint32_t split_hi =
        min((kv_split + 1) * blocks_per_split * BLOCK_KV, seq_len_kv);
    start = start / 4 * 4;  // TMA alignment for SF KV
    if (start < split_lo) start = split_lo;
    if (end > split_hi) end = split_hi;
    const uint32_t nkv =
        (end > start) ? math::ceil_div(end - start, BLOCK_KV) : 0;
    return {start, nkv};
  };

  const auto get_kv_pipeline =
      [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
    return {kv_block_idx % kNumKVStages, (kv_block_idx / kNumKVStages) & 1};
  };

  constexpr uint32_t UMMA_M = 128;
  constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
  constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;

  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    if (cute::elect_one_sync()) {
      if (block_q_idx < num_q_blocks) {
        // Q + weights once for this q-block. Inner extents are BYTES
        // for the uint8 maps, so fp4's packed rows halve them.
        constexpr uint32_t kKvInnerBytes = kHeadDim * kKvElemBytesNum / 2;
        tma::copy<kKvInnerBytes, BLOCK_Q * kNumHeads, kKvInnerBytes>(
            &tensor_map_q, full_q_barriers[0], smem_q[0], 0,
            block_q_idx * BLOCK_Q * kNumHeads);
        tma::copy<4, BLOCK_Q, 0>(&tensor_map_weights,
                                         full_q_barriers[0], smem_weights[0], 0,
                                         block_q_idx * BLOCK_Q);
        full_q_barriers[0]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE +
                                                 BLOCK_Q * 4u * sizeof(float));

        CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
        for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
             ++kv_block_idx) {
          CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
          empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);
#if defined(LISA_SCAN_STATIC)

          if constexpr (BLOCK_KV == 512) {
#pragma unroll
            for (uint32_t part=0; part<2; ++part) {
              tma::copy<kKvInnerBytes,256,kKvInnerBytes>(
                  &tensor_map_kv,full_kv_barriers[kv_stage_idx],
                  smem_kv[kv_stage_idx]+part*256*kHeadDim,0,
                  kv_start+kv_block_idx*BLOCK_KV+part*256);
              tma::copy<256,1,0>(
                  &tensor_map_kv_scales,full_kv_barriers[kv_stage_idx],
                  smem_kv_scales[kv_stage_idx]+part*256,
                  kv_start+kv_block_idx*BLOCK_KV+part*256,0);
            }
          } else {
#else

#endif
          tma::copy<kKvInnerBytes, BLOCK_KV, kKvInnerBytes>(
              &tensor_map_kv, full_kv_barriers[kv_stage_idx],
              smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
          tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales,
                                    full_kv_barriers[kv_stage_idx],
                                    smem_kv_scales[kv_stage_idx],
#if defined(LISA_SCAN_STATIC)
                                    kv_start + kv_block_idx * BLOCK_KV, 0);
          }
#else
                                    kv_start + kv_block_idx * BLOCK_KV, 0);
#endif
          full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(
              SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
        }
      }
    }
  } else if (warp_idx == kSpecWarpStart + 1) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    const uint32_t allocated_tmem = ptx::ld_shared(tmem_ptr_in_smem);

    auto instr_desc = cute::UMMA::make_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, UMMA_M, UMMA_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

    if (block_q_idx < num_q_blocks) {
      CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
      full_q_barriers[0]->wait(0);
#if defined(LISA_SCAN_RECOVERY)
#else
      const uint32_t cached_tmem = (kTuneBits & 2) ? ptx::ld_shared(tmem_ptr_in_smem) : 0;
#endif

      for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
           ++kv_block_idx) {
        const uint32_t kvg = kv_block_idx;
        CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);
        full_kv_barriers[kv_stage_idx]->wait(kv_phase);

        DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads, "Invalid block size");
        DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
        // Round-robin over kNumUmmaStages accumulators. A stage is
        // reused every kNumUmmaStages tiles, so its phase toggles at
        // that rate, not every tile.
        const uint32_t umma_stage = kvg % kNumUmmaStages;
        const uint32_t umma_phase = (kvg / kNumUmmaStages) & 1;
#pragma unroll
        for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
          const uint32_t buf = i * kNumUmmaStages + umma_stage;
          empty_umma_barriers[buf]->wait(umma_phase ^ 1);
          ptx::tcgen05_after_thread_sync();
#pragma unroll
          for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                     kHeadDim, kHeadDim>(
                smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
            auto b_desc =
                mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim,
                                           kHeadDim>(smem_q[0], 0, k * UMMA_K);
            cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, allocated_tmem + buf * kAccumStride, k,
                                           runtime_instr_desc);
          }
          cutlass::arch::umma_arrive(
              reinterpret_cast<uint64_t*>(full_umma_barriers[buf]));
        }
      }
      empty_q_barriers[0]->arrive();
    }
  } else if (warp_idx == kSpecWarpStart + 2 or warp_idx == kSpecWarpStart + 3) {
    // Spare-warp threshold-refresh daemon. Keeping refresh off the math
    // warps avoids placing histogram scan latency on their critical path.
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    if (block_q_idx < num_q_blocks) {
      const uint32_t spare_id = warp_idx - (kSpecWarpStart + 2);  // 0 or 1
      const auto refresh_row = [&](const uint32_t row,
                                   const bool publish_boundary_counts) {
        if (row >= seq_len) return false;
        const uint32_t local_row = row - block_q_idx * BLOCK_Q;
        const int32_t* srow =
            smem_hist + (row - block_q_idx * BLOCK_Q) * num_buckets;
        const int current_gate =
            min(max(static_cast<int>(__uint_as_float(
                        ptx::ld_shared(sparse_gate_bits + local_row))) -
                        1,
                    0),
                static_cast<int>(num_buckets) - 1);
        // Only buckets strictly below the published gate can tighten
        // it. If they contain fewer than topk entries, keep the
        // current gate and avoid scanning the dead upper tail.
        const uint32_t search_buckets = static_cast<uint32_t>(current_gate);
        int found = current_gate;
        int carry = 0;
        int found_lt = 0;
        int found_eq = 0;
        bool done = false;
        for (uint32_t base = 0; base < search_buckets && !done; base += 32) {
          uint32_t b = base + lane_idx;
          int v = 0;
          if (b < search_buckets) {
            // In the single-split fast path smem already contains
            // the initial global histogram plus every scan hit.
            v = srow[b];
          }
          // Histogram entries are nonnegative. Most 32-bucket
          // groups remain strictly below K, so reject them with
          // one REDUX and reserve the five dependent prefix
          // shuffles for the single group that crosses K.
          const int group_sum = __reduce_add_sync(0xffffffffu, v);
          if (carry + group_sum < static_cast<int>(topk)) {
            carry += group_sum;
            continue;
          }
          int prefix = v;
#pragma unroll
          for (int off = 1; off < 32; off <<= 1) {
            int nsh = __shfl_up_sync(0xffffffffu, prefix, off);
            if (static_cast<int>(lane_idx) >= off) prefix += nsh;
          }
          int incl = carry + prefix;
          bool hit = (b < search_buckets) && (incl >= static_cast<int>(topk)) &&
                     (incl - v < static_cast<int>(topk));
          unsigned hm = __ballot_sync(0xffffffffu, hit);
          if (hm) {
            const int hit_lane = __ffs(hm) - 1;
            found = static_cast<int>(base) + hit_lane;
            found_lt = __shfl_sync(0xffffffffu, incl - v, hit_lane);
            found_eq = __shfl_sync(0xffffffffu, v, hit_lane);
            done = true;
          } else {
            carry += __shfl_sync(0xffffffffu, prefix, 31);
          }
        }
        if (!done) {
          // No lower bucket reached K, so the current gate remains
          // the boundary. `carry` is exactly count(bucket<gate).
          found_lt = carry;
          found_eq = srow[found];
        }
        if (lane_idx == 0) {
          // One CTA owns this row. Publish only a genuine tightening;
          // a stale math-warp mailbox read remains conservatively loose.
          const uint32_t edge = __float_as_uint(static_cast<float>(found + 1));
          if (done) {
            th_bucket[row] = found;
            if constexpr (!kStaticHotGate) {
              atomicMin(sparse_gate_bits + local_row, edge);
            }
          }
          if (publish_boundary_counts && num_buckets >= 3) {
            // Reuse the dead histogram row as selector metadata.
            int32_t* meta = bcount + static_cast<uint64_t>(row) * num_buckets;
            meta[0] = ~found;
            meta[1] = found_lt;
            meta[2] = found_eq;
          }
        }
        return done;
      };
      // The external HOT sample provides a
      // fixed scan-time gate. Avoid every intermediate prefix scan and
      // publish the tight full-scan boundary certificate only once.
      if constexpr (kRingRefresh) {
        // Ring-readback refresher: histogram already-emitted smem
        // ring records (word-atomic reads) and tighten the CTA gate
        // mailbox with zero additions to the math-warp hit path.
        // Counting a SUBSET of emitted records keeps every published
        // edge a provable lower bound on the row's current K-th
        // best, and the per-(math warp, row) flush seqlock discards
        // any read that raced a drain so no record counts twice.
        //
        // The ring daemon stays active and sleeps for 2048 ns on quiet passes.
        uint32_t pass = 0;
        while (warp_uniform_shared_load(scan_done_flag) == 0) {
          bool all_stale = true;
          #pragma unroll
          for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
            all_stale &= warp_uniform_shared_load(ring_stale+r) >= 2;
          if (all_stale) break;
          const uint32_t local_row = spare_id + 2u * (pass % (BLOCK_Q / 2));
          ++pass;
          uint32_t fresh_total = 0;
          if (block_q_idx * BLOCK_Q + local_row < seq_len &&
              warp_uniform_shared_load(ring_stale+local_row) < 2) {
            for (uint32_t w = 0; w < kNumMathWarps; ++w) {
              const uint32_t pair = w * BLOCK_Q + local_row;
              const uint32_t s0 = warp_uniform_shared_load(ring_seq+pair);
              if (s0 != warp_uniform_shared_load(ring_shadow+pair)) {
                // New flush epoch: the drained column is
                // sentinel again and refills from zero.
                ring_progress[pair * 32u + lane_idx] = 0u;
                __syncwarp();
                if (lane_idx == 0) ring_shadow[pair] = s0;
                __syncwarp();
              }
              // Deep harvest: drain the lane's whole staged
              // column per visit (up to kEmitLaneSlots) instead
              // of one record per pass. The flood-time daemon
              // bottleneck is harvest bandwidth, not publish
              // cadence; reads and smem atomics only, so the
              // math warps' issue ports stay untouched.
              uint32_t progress = ring_progress[pair * 32u + lane_idx];
              uint32_t took = 0;
              while (progress < kEmitLaneSlots) {
                const uint32_t record = *reinterpret_cast<volatile uint32_t*>(
                    emit_smem_records +
                    (pair * kEmitLaneSlots + progress) * 32u + lane_idx);
                if (record == kRingSentinel) break;
                if (*reinterpret_cast<volatile uint32_t*>(ring_seq + pair) !=
                    s0) {
                  // Raced a flush drain: discard this
                  // read; the epoch reset above replays
                  // the refilled column next visit.
                  break;
                }
                // Invert the truncated fp24 code back to
                // the affine bucket value. Truncation only
                // rounds toward better scores and never
                // crosses an integer boundary (codes and
                // integer edges share the 0x100-aligned
                // grid), so unshifted bins support a
                // pad-free published edge. The comparison
                // ladder (not fmin/fmax) routes NaN and
                // out-of-range scores to the top bin, which
                // the publish guard below never counts.
                uint32_t fbits = record & 0xffffff00u;
                fbits = (fbits & 0x80000000u) ? (fbits ^ 0x80000000u) : ~fbits;
                const float value = __uint_as_float(fbits);
                uint32_t bucket;
                if (value < 0.0f) {
                  bucket = 0u;
                } else if (value < static_cast<float>(num_buckets - 1)) {
                  bucket = static_cast<uint32_t>(value);
                } else {
                  bucket = num_buckets - 1;
                }
                atomicAdd(smem_hist + local_row * num_buckets + bucket, 1);
                ++progress;
                ++took;
              }
              if (took != 0u) {
                ring_progress[pair * 32u + lane_idx] =
                    static_cast<uint8_t>(progress);
              }
              const uint32_t fresh = __reduce_add_sync(0xffffffffu, took);
              if (lane_idx == 0 && fresh != 0u) {
                ring_pending[local_row] += static_cast<int32_t>(fresh);
              }
              fresh_total += fresh;
            }
            __syncwarp();
            // The seed base already carries K records, so each
            // topk/16 batch of fresh evidence can move the
            // boundary.
            const int32_t pend = warp_uniform_shared_load(ring_pending+local_row);
            if (pend - warp_uniform_shared_load(ring_last+local_row) >=
                static_cast<int32_t>(topk) / 16) {
              if (lane_idx == 0) ring_last[local_row] = pend;
              __syncwarp();
              // Smallest bucket prefix holding at least K ring
              // records. The truncated codes and the ordered
              // codes of integers <= 256 both sit on the
              // 0x100-aligned grid, so a counted record's
              // true code is < ord(found + 1) exactly and the
              // bare bucket upper boundary is a provably safe
              // published edge — no truncation pad needed.
              int carry = 0;
              int found = -1;
              for (uint32_t bbase = 0; bbase < num_buckets && found < 0;
                   bbase += 32) {
                const uint32_t b = bbase + lane_idx;
                const int v = b < num_buckets
                                  ? smem_hist[local_row * num_buckets + b]
                                  : 0;
                const int group_sum = __reduce_add_sync(0xffffffffu, v);
                if (carry + group_sum < static_cast<int>(topk)) {
                  carry += group_sum;
                  continue;
                }
                int prefix = v;
#pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                  const int nsh = __shfl_up_sync(0xffffffffu, prefix, off);
                  if (static_cast<int>(lane_idx) >= off) prefix += nsh;
                }
                const bool hit = carry + prefix >= static_cast<int>(topk);
                const unsigned hm = __ballot_sync(0xffffffffu, hit);
                if (hm) {
                  found = static_cast<int>(bbase) + __ffs(hm) - 1;
                } else {
                  carry += __shfl_sync(0xffffffffu, prefix, 31);
                }
              }
              if (lane_idx == 0) {
                bool published = false;
                // The top bin collects NaN and clamped
                // out-of-range records, so an edge may only
                // be published from a strictly lower bin.
                if (found >= 0 && found < static_cast<int>(num_buckets) - 1) {
                  const uint32_t edge =
                      __float_as_uint(static_cast<float>(found + 1));
                  published =
                      edge < atomicMin(sparse_gate_bits + local_row, edge);
                }
                // Early warm-start scans that fail to
                // tighten only lack fresh evidence, not
                // convergence; they must not retire the
                // daemon.
                ring_stale[local_row] =
                    published ? 0
                              : (pend >= static_cast<int32_t>(topk)
                                     ? ring_stale[local_row] + 1
                                     : ring_stale[local_row]);
              }
              __syncwarp();
            }
          }
          // Quiet pacing. The CTA retires promptly once the math
          // warps finish; bursts skip pacing and chain straight
          // into the next pass.
          if (fresh_total < 64u) {
            uint32_t slept = 0;
            while (slept < kTuneRingNs && warp_uniform_shared_load(scan_done_flag) == 0) {
              __nanosleep(kTuneRingNs);
              slept += kTuneRingNs;
            }
          }
        }
      } else if constexpr (kStaticHotNoHist) {
        // Scratch path: no in-scan histogram has to be finalized, so
        // both spare refresh warps can retire immediately.
      } else if constexpr (kStaticHotGate) {
        while (*scan_done_flag == 0) __nanosleep(kSparseRefreshIdleNs);
      } else {
        while (*scan_done_flag == 0) {
          bool tightened = false;
          for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
            tightened |= refresh_row(block_q_idx * BLOCK_Q + r, false);
          __nanosleep(tightened ? kSparseRefreshNs : kSparseRefreshIdleNs);
        }
      }
      if constexpr (!kStaticHotNoHist) {
        for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
          refresh_row(block_q_idx * BLOCK_Q + r, true);
      }
    }
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    const auto math_thread_idx = warp_idx * 32 + lane_idx;

    auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr,
                        float* accum) {
      constexpr int N = decltype(num_elems_c)::value;
#if defined(LISA_SCAN_RECOVERY)
      DG_STATIC_ASSERT(N == 32 or N == 64 or N == 128,
#else
      DG_STATIC_ASSERT(N == 8 or N == 16 or N == 32 or N == 64 or N == 128,
#endif
                       "Unsupported TMEM load size");
#if defined(LISA_SCAN_RECOVERY)
      using Loader = cute::conditional_t<
#else
      using Loader = cute::conditional_t<N == 8, cute::SM100_TMEM_LOAD_32dp32b8x,
          cute::conditional_t<N == 16, cute::SM100_TMEM_LOAD_32dp32b16x, cute::conditional_t<
#endif
          N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
          cute::conditional_t<N == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
#if defined(LISA_SCAN_RECOVERY)
                              cute::SM100_TMEM_LOAD_32dp32b128x>>;
#else
                              cute::SM100_TMEM_LOAD_32dp32b128x>>>>;
#endif
      [&]<size_t... Is>(cute::index_sequence<Is...>) {
        Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
      }(cute::make_index_sequence<N>{});
      cutlass::arch::fence_view_async_tmem_load();
    };

    // Bucket comparisons use the affine score space. Raw float bit
    // patterns are nonuniform across exponent ranges and cannot preserve
    // this threshold contract.
    float weights[BLOCK_Q][kNumHeads];
    float o_reg[BLOCK_Q], inv_reg[BLOCK_Q], vth_reg[BLOCK_Q];
    uint32_t kstart_reg[BLOCK_Q],
        kspan_reg[BLOCK_Q];  // unsigned range-check trick
    const unsigned FULL = 0xffffffffu;

    if (block_q_idx < num_q_blocks) {
      CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
      full_q_barriers[0]->wait(0);
#if defined(LISA_SCAN_RECOVERY)
#else
      const uint32_t cached_tmem = (kTuneBits & 2) ? ptx::ld_shared(tmem_ptr_in_smem) : 0;
#endif

// Weights into registers (once per CTA -- the 2.5-generation win).
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j)
          weights[i][j] = ptx::ld_shared(smem_weights[0] + i * 4 + j);
      }
      // Queue fill counts are warp-uniform: every lane tracks them
      // redundantly in registers, so the hot emit path needs no shared
      // bookkeeping or shuffle broadcast.
      // Four private 8-bit counts. They reset at every fixed-size chunk,
      // bounding the address dispersion of direct global record stores.
      LaneCounts<BLOCK_Q> emit_lane_counts;
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        const uint32_t rq = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
        if constexpr (kOnlineFixedBuckets) {
          // No sample-derived affine transform in the online A/B
          // path.  The raw score is converted to fixed buckets in
          // the epilogue below; keep these registers compile-time
          // independent of the nullable origin/inv pointers.
          o_reg[i] = 0.0f;
          inv_reg[i] = 1.0f;
          vth_reg[i] = 0.0f;
        } else {
          o_reg[i] = origin[rq];
          inv_reg[i] = inv_delta[rq];
          vth_reg[i] = -o_reg[i] * inv_reg[i];
        }
        o_reg[i] = 0.0f;  // gate closed until the first consume
        kstart_reg[i] = seq_k_start[i];
        kspan_reg[i] =
            seq_k_end[i] > seq_k_start[i] ? seq_k_end[i] - seq_k_start[i] : 0;
      }
      // FP8x2 scales remain raw; affine conversion follows the signed score.
      uint32_t rs_max = 0, re_min = 0xffffffffu;
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        rs_max = max(rs_max, kstart_reg[i]);
        re_min = min(re_min, kstart_reg[i] + kspan_reg[i]);
      }

      if constexpr (BLOCK_Q == 4) {
        const float4 initial_gate =
            ptx::ld_shared(reinterpret_cast<const float4*>(sparse_gate_bits));
        o_reg[0] = initial_gate.x;
        o_reg[1] = initial_gate.y;
        o_reg[2 < BLOCK_Q ? 2 : 0] = initial_gate.z;
        o_reg[3 < BLOCK_Q ? 3 : 0] = initial_gate.w;
#if defined(LISA_SCAN_RECOVERY)
#else
      } else if constexpr (kTuneBits & 4) {
#pragma unroll
        for (uint32_t gi = 0; gi < BLOCK_Q; gi += 4) {
          const float4 gate = ptx::ld_shared(reinterpret_cast<const float4*>(sparse_gate_bits + gi));
          o_reg[gi] = gate.x; o_reg[gi+1] = gate.y;
          o_reg[gi+2] = gate.z; o_reg[gi+3] = gate.w;
        }
#endif
      } else {
#pragma unroll
        for (uint32_t gi = 0; gi < BLOCK_Q; ++gi)
          o_reg[gi] = __uint_as_float(ptx::ld_shared(sparse_gate_bits + gi));
      }

      // The first 512 records of every (q-block, math-warp) are a
      // deterministic micro-slab, so the common light warp never
      // contends on pool_head. Once abandoned, a 4K pooled page remains
      // current across later windows until its bump no longer fits.
      uint32_t pooled_segment_base = kPooledSegmentInvalidStart;
      uint32_t pooled_segment_bump = 0;
      uint32_t pooled_segment_capacity = kPooledSegmentMicroRecords;
      uint32_t* const my_spill_count = pooled_spill_count + warp_idx * BLOCK_Q;
      int32_t* const my_spill_page_ids =
          pooled_spill_page_ids + warp_idx * BLOCK_Q * kPooledSegmentSpillPages;
      if constexpr (kPooledSegments) {
        pooled_segment_base = (block_q_idx * kNumMathWarps + warp_idx) *
                              kPooledSegmentMicroRecords;
        if (lane_idx < BLOCK_Q) {
          my_spill_count[lane_idx] = 0u;
        }
        if (lane_idx < BLOCK_Q * kPooledSegmentSpillPages) {
          my_spill_page_ids[lane_idx] = kPagedCandidateEmpty;
        }
        __syncwarp();
      }

      for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
           ++kv_block_idx) {
        const uint32_t kvg = kv_block_idx;
        CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);

        // Reload the gate once per window to keep mailbox reads out
        // of the per-block scoring loop.
        if constexpr ((!kStaticHotGate || kRingRefresh) && !kRawDumpEmit) {
          if (kv_block_idx != 0 && (kv_block_idx % kActiveGateStride) == 0) {
            if constexpr (BLOCK_Q == 4) {
              const float4 gate = ptx::ld_shared(
                  reinterpret_cast<const float4*>(sparse_gate_bits));
              o_reg[0] = gate.x;
              o_reg[1] = gate.y;
              o_reg[2 < BLOCK_Q ? 2 : 0] = gate.z;
              o_reg[3 < BLOCK_Q ? 3 : 0] = gate.w;
#if defined(LISA_SCAN_RECOVERY)
#else
            } else if constexpr (kTuneBits & 4) {
#pragma unroll
              for (uint32_t gi = 0; gi < BLOCK_Q; gi += 4) {
                const float4 gate = ptx::ld_shared(reinterpret_cast<const float4*>(sparse_gate_bits + gi));
                o_reg[gi] = gate.x; o_reg[gi+1] = gate.y;
                o_reg[gi+2] = gate.z; o_reg[gi+3] = gate.w;
              }
#endif
            } else {
#pragma unroll
              for (uint32_t gi = 0; gi < BLOCK_Q; ++gi)
#if defined(LISA_SCAN_RECOVERY)
                o_reg[gi] =
                    __uint_as_float(ptx::ld_shared(sparse_gate_bits + gi));
#else
                o_reg[gi] = __uint_as_float(ptx::ld_shared(sparse_gate_bits + gi));
#endif
            }
          }
        }
#if defined(LISA_SCAN_SYNC) || defined(LISA_SCAN_STATIC) || defined(LISA_SCAN_FINAL)
        float scale_kv;
        if constexpr (!(kTuneBits & 8)) {
          full_kv_barriers[kv_stage_idx]->wait(kv_phase);
          scale_kv = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);
        }
#else
        full_kv_barriers[kv_stage_idx]->wait(kv_phase);

        const float scale_kv =
            ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);
#endif

        const uint32_t umma_buf =
            warpgroup_idx * kNumUmmaStages + (kvg % kNumUmmaStages);
        const uint32_t umma_wait_phase = (kvg / kNumUmmaStages) & 1;
#if defined(LISA_SCAN_RECOVERY)
        const auto tmem_start = ptx::ld_shared(tmem_ptr_in_smem) + umma_buf * kAccumStride;
#else
        const auto tmem_start = ((kTuneBits & 2) ? cached_tmem : ptx::ld_shared(tmem_ptr_in_smem)) + umma_buf * kAccumStride;
#endif
        full_umma_barriers[umma_buf]->wait(umma_wait_phase);
        ptx::tcgen05_after_thread_sync();
#if defined(LISA_SCAN_SYNC)
        if constexpr (kTuneBits & 8) {
          // MMA producer acquired this KV stage before publishing the
          // completed accumulator. Read the scales before releasing KV.
          scale_kv = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);
        }
        empty_kv_barriers[kv_stage_idx]->arrive();
#elif defined(LISA_SCAN_STATIC) || defined(LISA_SCAN_FINAL)
        if constexpr (kTuneBits & 8) {
          // MMA producer acquired this KV stage before publishing the
          // completed accumulator. Read the scales before releasing KV.
          scale_kv = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);
        }
        if constexpr (kTuneBits & 16) {
          __syncwarp();
          if (lane_idx == 0) empty_kv_barriers[kv_stage_idx]->arrive();
        } else {
          empty_kv_barriers[kv_stage_idx]->arrive();
        }
#else

        empty_kv_barriers[kv_stage_idx]->arrive();
#endif

        const auto kv_offset =
            kv_start + kv_block_idx * BLOCK_KV + math_thread_idx;
        DG_STATIC_ASSERT(kNumHeads == 2, "FP8x2 needs exactly high and residual");

        uint32_t pass_bits = 0;
#if defined(LISA_SCAN_RECOVERY)
#else
        constexpr uint32_t kLoadCols = (kTuneBits & 1) ? BLOCK_Q * kNumHeads : kAccumStride;
#endif
        float v_row[BLOCK_Q];
        uint32_t fixed_payload_row[BLOCK_Q];
        uint32_t fixed_bucket_row[BLOCK_Q];
        constexpr uint32_t kTmemRowsEff =
            (128u / kNumHeads) < BLOCK_Q ? (128u / kNumHeads) : BLOCK_Q;
        DG_STATIC_ASSERT(BLOCK_Q % kTmemRowsEff == 0,
                         "BLOCK_Q must divide evenly into TMEM load groups");
#define LITETOPK_SCORE_GATE(i, RC)                                          \
  bool g;                                                                   \
  if constexpr (kOnlineFixedBuckets) {                                      \
    const float raw_score = scale_kv * (sum.x + sum.y);                     \
    const OnlineFixedCandidate fixed = candidate_online_fixed(raw_score);   \
    fixed_payload_row[i] = fixed.payload;                                   \
    fixed_bucket_row[i] = fixed.bucket;                                     \
    g = fixed.bucket < static_cast<uint32_t>(o_reg[i]);                     \
  } else {                                                                  \
    const float raw = __fmul_rn(__fadd_rn(sum.x, sum.y), scale_kv);     \
    const float bq = fmaf(-raw, inv_reg[i], vth_reg[i]);             \
    if constexpr (kStaticHotExactGate) {                                    \
      const uint32_t ordered = candidate_ordered_fp32_code(bq);             \
      v_row[i] = __uint_as_float(ordered);                                  \
      g = ordered < __float_as_uint(o_reg[i]);                              \
    } else {                                                                \
      v_row[i] = bq;                                                        \
      g = __float_as_int(bq) < __float_as_int(o_reg[i]);                    \
    }                                                                       \
  }                                                                         \
  if constexpr (RC) g = g and ((kv_offset - kstart_reg[i]) < kspan_reg[i]); \
  pass_bits |= g ? (1u << i) : 0u;
        const uint32_t kv_base = kv_start + kv_block_idx * BLOCK_KV;
        const bool interior =
            (kv_base >= rs_max) && (kv_base + BLOCK_KV <= re_min);
#if defined(LISA_SCAN_RECOVERY)
#define LITETOPK_SCORE_ROWS(RANGE_CHECK)                                       \
  _Pragma("unroll") for (uint32_t pr = 0; pr < BLOCK_Q / kTmemRowsEff; ++pr) { \
    float accum2[32];                                    \
    tmem_load(cute::Int<32>{},                           \
              tmem_start + pr * kTmemRowsEff * kNumHeads, accum2);             \
    if (pr == BLOCK_Q / kTmemRowsEff - 1) {                                    \
      ptx::tcgen05_before_thread_sync();                                       \
      empty_umma_barriers[umma_buf]->arrive();                                 \
    }                                                                          \
    _Pragma("unroll") for (uint32_t k = 0; k < kTmemRowsEff; ++k) {            \
      const uint32_t i = pr * kTmemRowsEff + k;                                \
      const float* accum = accum2 + k * kNumHeads;                             \
      const auto sum = make_float2(                                        \
          __fmul_rn(accum[0], weights[i][0]),                                \
          __fmul_rn(accum[1], weights[i][1]));                               \
      LITETOPK_SCORE_GATE(i, RANGE_CHECK)                                      \
    }                                                                          \
  }
#elif defined(LISA_SCAN_STATIC) || defined(LISA_SCAN_FINAL)
#define LITETOPK_SCORE_ROWS(RANGE_CHECK)                                       \
  _Pragma("unroll") for (uint32_t pr = 0; pr < BLOCK_Q / kTmemRowsEff; ++pr) { \
    float accum2[kLoadCols];                                    \
    tmem_load(cute::Int<kLoadCols>{},                           \
              tmem_start + pr * kTmemRowsEff * kNumHeads, accum2);             \
    if (pr == BLOCK_Q / kTmemRowsEff - 1) {                                    \
      ptx::tcgen05_before_thread_sync();                                       \
      if constexpr (kTuneBits & 16) { __syncwarp(); if (lane_idx == 0) empty_umma_barriers[umma_buf]->arrive(); } else { empty_umma_barriers[umma_buf]->arrive(); }                                 \
    }                                                                          \
    _Pragma("unroll") for (uint32_t k = 0; k < kTmemRowsEff; ++k) {            \
      const uint32_t i = pr * kTmemRowsEff + k;                                \
      const float* accum = accum2 + k * kNumHeads;                             \
      const auto sum = make_float2(                                        \
          __fmul_rn(accum[0], weights[i][0]),                                \
          __fmul_rn(accum[1], weights[i][1]));                               \
      LITETOPK_SCORE_GATE(i, RANGE_CHECK)                                      \
    }                                                                          \
  }
#else
#define LITETOPK_SCORE_ROWS(RANGE_CHECK)                                       \
  _Pragma("unroll") for (uint32_t pr = 0; pr < BLOCK_Q / kTmemRowsEff; ++pr) { \
    float accum2[kLoadCols];                                    \
    tmem_load(cute::Int<kLoadCols>{},                           \
              tmem_start + pr * kTmemRowsEff * kNumHeads, accum2);             \
    if (pr == BLOCK_Q / kTmemRowsEff - 1) {                                    \
      ptx::tcgen05_before_thread_sync();                                       \
      empty_umma_barriers[umma_buf]->arrive();                                 \
    }                                                                          \
    _Pragma("unroll") for (uint32_t k = 0; k < kTmemRowsEff; ++k) {            \
      const uint32_t i = pr * kTmemRowsEff + k;                                \
      const float* accum = accum2 + k * kNumHeads;                             \
      const auto sum = make_float2(                                        \
          __fmul_rn(accum[0], weights[i][0]),                                \
          __fmul_rn(accum[1], weights[i][1]));                               \
      LITETOPK_SCORE_GATE(i, RANGE_CHECK)                                      \
    }                                                                          \
  }
#endif
        if (interior) {
          LITETOPK_SCORE_ROWS(false)
        } else {
          LITETOPK_SCORE_ROWS(true)
        }
#undef LITETOPK_SCORE_ROWS
#undef LITETOPK_SCORE_GATE

        if (pass_bits != 0) {
#pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            if ((pass_bits >> i) & 1u) {
              uint32_t candidate_score_bits;
              uint32_t candidate_bucket;
              if constexpr (kOnlineFixedBuckets) {
                candidate_score_bits = fixed_payload_row[i] << 8;
                candidate_bucket = fixed_bucket_row[i];
              } else {
                if constexpr (kStaticHotExactGate) {
                  // v_row carries the already-computed
                  // ordered32 key in this specialization.
                  // The ring's low byte belongs exclusively
                  // to local_block; clear the discarded FP32
                  // low bits before packing that coordinate.
                  candidate_score_bits =
                      __float_as_uint(v_row[i]) & 0xffffff00u;
                  candidate_bucket = 0;
                } else {
                  const float x = v_row[i];
                  candidate_score_bits = candidate_fp24_code(x) << 8;
                  const int candidate_braw = static_cast<int>(x);
                  candidate_bucket =
                      static_cast<uint32_t>(max(candidate_braw, 0));
                }
              }
              const uint32_t count = emit_lane_counts.get(i);
              if (count < kEmitLaneSlots) {
                const uint32_t pos =
                    ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + count) * 32u +
                    lane_idx;
                const uint32_t local_block =
                    kv_block_idx % kActiveEmitChunkBlocks;
                const uint32_t record = candidate_score_bits | local_block;
                emit_smem_records[pos] = record;
                emit_lane_counts.add(i);
              } else {
                // A skewed lane can overflow its small local
                // quota without losing candidates. This slow
                // path writes directly to the final buffer.
                if constexpr (kPooledSegments) {
                  const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                  const uint32_t micro_records =
                      num_q_blocks * kNumMathWarps * kPooledSegmentMicroRecords;
                  pooled_segment_store_lane_spill(
                      row_q, candidate_score_bits >> 8, kv_offset, probe_group,
                      probe_magic, probe_add_max, my_spill_count + i,
                      my_spill_page_ids + i * kPooledSegmentSpillPages,
                      segment_val, segment_idx, segment_pool_head,
                      segment_status, segment_pool_pages, micro_records);
                } else {
                  const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                  int out;
                  if constexpr (kAggregateSpills) {
                    const uint32_t spill_mask = __activemask();
                    const uint32_t leader = __ffs(spill_mask) - 1;
                    int base = 0;
                    if (lane_idx == leader)
                      base = atomicAdd(cand_cnt + row_q, __popc(spill_mask));
                    base = __shfl_sync(spill_mask, base, leader);
                    out = base + __popc(spill_mask & ((1u << lane_idx) - 1));
                  } else {
                    out = atomicAdd(cand_cnt + row_q, 1);
                  }
                  if constexpr (kPagedCandidates) {
                    paged_candidate_store_lane_overflow(
                        row_q, out, cand_cap, candidate_score_bits >> 8,
                        kv_offset, probe_group, probe_magic, probe_add_max,
                        cand_val, cand_idx, overflow_val, overflow_idx,
                        page_table, pool_head, pool_status, pool_pages);
                  } else {
                    if (out < static_cast<int>(cand_cap)) {
                      uint32_t kvo = kv_offset;
                      if (probe_group != 0) {
                        const uint32_t sup = static_cast<uint32_t>(
                            (static_cast<uint64_t>(kvo) * probe_magic) >> 42);
                        kvo += min((sup + 1) * 64u, probe_add_max);
                      }
                      const uint64_t out_base =
                          static_cast<uint64_t>(row_q) * cand_cap;
                      store_candidate_record(&cand_val[out_base + out],
                                             &cand_idx[out_base + out],
                                             candidate_score_bits >> 8, kvo);
                    }
                  }
                }
              }

              // A passing positive bq is below the float edge
              // (gate + 1), hence already < num_buckets.
              if constexpr (!kStaticHotNoHist) {
                atomicAdd(smem_hist + i * num_buckets +
                              static_cast<int>(candidate_bucket),
                          1);
              }
            }
          }
        }

        if (((kv_block_idx + 1) % kActiveEmitChunkBlocks) == 0 ||
            kv_block_idx + 1 == num_kv_blocks) {
          if constexpr (kFlushMode == 1) {
            static_assert(kRingRefresh && kStaticHotNoHist && !kPagedCandidates && !kPooledSegments);
            flush_native_ring<BLOCK_Q,kEmitLaneSlots,BLOCK_KV>(emit_lane_counts,
                emit_smem_records+warp_idx*BLOCK_Q*kEmitLaneSlots*32u,
                ring_seq+warp_idx*BLOCK_Q,cand_val,cand_idx,cand_cnt,cand_cap,
                block_q_idx*BLOCK_Q,seq_len,
                kv_start+(kv_block_idx/kActiveEmitChunkBlocks)*kActiveEmitChunkBlocks*BLOCK_KV,
                math_thread_idx);
          } else if constexpr (kRawDumpEmit) {
            // One deterministic region per (q-block, warp, window):
            // no output reservation atomic, no per-record drain.
            // Each lane publishes its packed 4x8-bit counts; one
            // elected lane bulk-copies the warp's whole staged ring
            // block. The wait_group.read releases the smem block
            // for the next window's appends.
            const uint32_t window_idx = kv_block_idx / kActiveEmitChunkBlocks;
            const uint64_t region_idx =
                (static_cast<uint64_t>(block_q_idx) * kNumMathWarps +
                 warp_idx) *
                    dump_max_windows +
                window_idx;
            constexpr uint32_t kDumpRegionWords =
                BLOCK_Q * kEmitLaneSlots * 32u;
            __stcg(dump_counts + region_idx * 32u + lane_idx, emit_lane_counts.words[0]);
            __syncwarp();
            if (cute::elect_one_sync()) {
              const uint32_t* smem_block =
                  emit_smem_records + warp_idx * kDumpRegionWords;
              DG_TRAP_ONLY_DEVICE_ASSERT(
                  (__cvta_generic_to_shared(smem_block) & 15u) == 0);
              raw_dump_emit_block(smem_block,
                                  dump_slab + region_idx * kDumpRegionWords,
                                  kDumpRegionWords * 4u);
            }
            __syncwarp();
            if constexpr (kRingRefresh) {
              const RawDumpGate<BLOCK_Q> gate =
                  raw_dump_ring_flush_tail<BLOCK_Q>(
                      emit_smem_records + warp_idx * kDumpRegionWords,
                      ring_seq + warp_idx * BLOCK_Q, sparse_gate_bits, lane_idx,
                      emit_lane_counts);
#pragma unroll
              for (uint32_t gi = 0; gi < BLOCK_Q; ++gi) o_reg[gi] = gate.g[gi];
            }
          } else {
            // Reserve one contiguous final-buffer segment per
            // (source warp, row, chunk), then have each lane copy its
            // private records into that segment. This keeps all
            // collectives and the returning global atomic off the
            // ordinary-hit path while deleting the rectangular global
            // chunk workspace and the post-scan compactor.
            // Keep the four row reservations parallel and early so
            // their L2 round trips overlap the prefix work. Replace
            // four independent 32-lane scans (20 shuffles) with two
            // scans whose 16-bit fields carry two rows each (10
            // shuffles). A row prefix is at most 32*18=576, so fields
            // cannot carry into one another.
            int my_row_base = 0;
            uint32_t my_row_total = 0;
            int my_page_ordinal0 = -1;
            int my_page0 = -1;
            int my_page1 = -1;
            uint32_t my_segment_row_offset = 0;
            uint32_t segment_window_total = 0;
            uint32_t segment_window_base = kPooledSegmentInvalidStart;
#pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++i) {
              const uint32_t count = emit_lane_counts.get(i);
              const uint32_t total = __reduce_add_sync(FULL, count);
              if (lane_idx == i) my_row_total = total;
            }
            if constexpr (kPooledSegments) {
              // All divergent spill helpers have returned before
              // their owner lanes publish page descriptors.
              __syncwarp();
              uint32_t inclusive_rows = my_row_total;
#pragma unroll
              for (int delta = 1; delta < static_cast<int>(BLOCK_Q);
                   delta <<= 1) {
                const uint32_t other =
                    __shfl_up_sync(FULL, inclusive_rows, delta);
                if (lane_idx < BLOCK_Q &&
                    lane_idx >= static_cast<uint32_t>(delta)) {
                  inclusive_rows += other;
                }
              }
              if (lane_idx < BLOCK_Q) {
                my_segment_row_offset = inclusive_rows - my_row_total;
              }
              segment_window_total =
                  __shfl_sync(FULL, inclusive_rows, BLOCK_Q - 1);
              if (segment_window_total != 0) {
                if (pooled_segment_base == kPooledSegmentInvalidStart ||
                    pooled_segment_bump + segment_window_total >
                        pooled_segment_capacity) {
                  uint32_t page = kPooledSegmentInvalidStart;
                  if (lane_idx == 0) {
                    page = pooled_segment_allocate_page(
                        segment_pool_head, segment_status, segment_pool_pages);
                  }
                  page = __shfl_sync(FULL, page, 0);
                  pooled_segment_bump = 0;
                  pooled_segment_capacity = kPooledSegmentPageRecords;
                  if (page == kPooledSegmentInvalidStart) {
                    pooled_segment_base = kPooledSegmentInvalidStart;
                  } else {
                    const uint32_t micro_records = num_q_blocks *
                                                   kNumMathWarps *
                                                   kPooledSegmentMicroRecords;
                    pooled_segment_base =
                        micro_records + page * kPooledSegmentPageRecords;
                  }
                }
                if (pooled_segment_base != kPooledSegmentInvalidStart) {
                  segment_window_base =
                      pooled_segment_base + pooled_segment_bump;
                  pooled_segment_bump += segment_window_total;
                }
              }

              if (lane_idx < BLOCK_Q) {
                const uint32_t row_q = block_q_idx * BLOCK_Q + lane_idx;
                if (row_q < seq_len) {
                  const uint32_t count =
                      segment_window_base != kPooledSegmentInvalidStart
                          ? my_row_total
                          : 0u;
                  const uint32_t physical_start =
                      count != 0 ? segment_window_base + my_segment_row_offset
                                 : kPooledSegmentInvalidStart;
                  const uint32_t window_idx =
                      kv_block_idx / kActiveEmitChunkBlocks;
                  const uint64_t descriptor_idx =
                      (static_cast<uint64_t>(row_q) * segment_desc_windows +
                       window_idx) *
                          kNumMathWarps +
                      warp_idx;
                  segment_desc[descriptor_idx] =
                      pooled_segment_pack_descriptor(physical_start, count);

                  const uint32_t spill_n = my_spill_count[lane_idx];
                  if (spill_n != 0u) {
                    atomicExch(segment_row_spill + row_q, 1);
                    if (spill_n >
                        kPooledSegmentSpillPages * kPooledSegmentPageRecords) {
                      paged_candidate_set_status(segment_status,
                                                 kPooledSegmentLaneOverflow);
                    }
                    const uint32_t micro_records = num_q_blocks *
                                                   kNumMathWarps *
                                                   kPooledSegmentMicroRecords;
#pragma unroll
                    for (uint32_t ordinal = 0;
                         ordinal < kPooledSegmentSpillPages; ++ordinal) {
                      const uint32_t ordinal_begin =
                          ordinal * kPooledSegmentPageRecords;
                      const uint32_t spill_count =
                          spill_n > ordinal_begin
                              ? min(spill_n - ordinal_begin,
                                    kPooledSegmentPageRecords)
                              : 0u;
                      if (spill_count == 0u) continue;
                      const int32_t page =
                          my_spill_page_ids[lane_idx *
                                                kPooledSegmentSpillPages +
                                            ordinal];
                      uint32_t spill_start = kPooledSegmentInvalidStart;
                      uint32_t published_count = 0u;
                      if (page >= 0) {
                        spill_start =
                            micro_records + static_cast<uint32_t>(page) *
                                                kPooledSegmentPageRecords;
                        published_count = spill_count;
                      } else {
                        paged_candidate_set_status(
                            segment_status, kPagedCandidatePoolExhausted);
                      }
                      const uint64_t spill_descriptor_idx =
                          (((static_cast<uint64_t>(row_q) *
                                 segment_desc_windows +
                             window_idx) *
                                kNumMathWarps +
                            warp_idx) *
                               kPooledSegmentSpillPages +
                           ordinal);
                      segment_spill_desc[spill_descriptor_idx] =
                          pooled_segment_pack_descriptor(spill_start,
                                                         published_count);
                    }
                  }

                  my_spill_count[lane_idx] = 0u;
#pragma unroll
                  for (uint32_t ordinal = 0; ordinal < kPooledSegmentSpillPages;
                       ++ordinal) {
                    my_spill_page_ids[lane_idx * kPooledSegmentSpillPages +
                                      ordinal] = kPagedCandidateEmpty;
                  }
                }
              }
              __syncwarp();
            } else {
              if (lane_idx < BLOCK_Q) {
                const uint32_t row_q = block_q_idx * BLOCK_Q + lane_idx;
                if (row_q < seq_len && my_row_total != 0) {
                  my_row_base = atomicAdd(cand_cnt + row_q,
                                          static_cast<int>(my_row_total));
                  if constexpr (kPagedCandidates) {
                    const uint32_t reservation_end =
                        static_cast<uint32_t>(my_row_base) + my_row_total;
                    const bool row_inline =
                        reservation_end <= kPagedCandidateInlineRecords;
                    if (!row_inline) {
                      const PagedCandidateReservation reservation =
                          paged_candidate_prepare_reservation(
                              row_q, my_row_base, my_row_total, cand_cap,
                              page_table, pool_head, pool_status, pool_pages);
                      my_page_ordinal0 = reservation.page_ordinal0;
                      my_page0 = reservation.page0;
                      my_page1 = reservation.page1;
                      my_row_base =
                          static_cast<int>(static_cast<uint32_t>(my_row_base) |
                                           kPagedCandidateOverflowFlag);
                    }
                  }
                }
              }
            }

            uint32_t inclusive_pairs[BLOCK_Q / 2];
            #pragma unroll
            for (uint32_t pair = 0; pair < BLOCK_Q / 2; ++pair) {
              uint32_t v = emit_lane_counts.get(pair*2) |
                           (emit_lane_counts.get(pair*2+1) << 16);
              #pragma unroll
              for (int delta=1; delta<32; delta<<=1) {
                const uint32_t other=__shfl_up_sync(FULL,v,delta);
                if (lane_idx >= static_cast<uint32_t>(delta)) v += other;
              }
              inclusive_pairs[pair]=v;
            }

            const uint32_t emit_window_kv_base =
                kv_start + (kv_block_idx / kActiveEmitChunkBlocks) *
                               kActiveEmitChunkBlocks * BLOCK_KV;
#pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++i) {
              const uint32_t count = emit_lane_counts.get(i);
              const uint32_t inclusive =
                  (inclusive_pairs[i / 2] >> ((i & 1u) * 16)) & 0xffffu;
              const uint32_t offset = inclusive - count;
              const uint32_t row_q = block_q_idx * BLOCK_Q + i;
              int out_base = 0;
              bool row_overflow = false;
              if constexpr (!kPooledSegments) {
                out_base = __shfl_sync(FULL, my_row_base, i);
                if constexpr (kPagedCandidates) {
                  row_overflow = (static_cast<uint32_t>(out_base) &
                                  kPagedCandidateOverflowFlag) != 0;
                  if (row_overflow) {
                    out_base =
                        static_cast<int>(static_cast<uint32_t>(out_base) &
                                         ~kPagedCandidateOverflowFlag);
                  }
                }
              }
              int copy_count = 0;
              if (row_q < seq_len) {
                if constexpr (kPooledSegments) {
                  copy_count = segment_window_base != kPooledSegmentInvalidStart
                                   ? static_cast<int>(count)
                                   : 0;
                } else {
                  copy_count =
                      kPagedCandidates && !row_overflow
                          ? static_cast<int>(count)
                          : min(static_cast<int>(count),
                                max(static_cast<int>(cand_cap) - out_base -
                                        static_cast<int>(offset),
                                    0));
                }
              }
              if constexpr (kPooledSegments) {
                const uint32_t row_segment_offset =
                    __shfl_sync(FULL, my_segment_row_offset, i);
                const uint32_t physical_out_base =
                    segment_window_base + row_segment_offset + offset;
#pragma unroll 1
                for (int slot = 0; slot < copy_count; ++slot) {
                  const uint32_t local_pos =
                      ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + slot) * 32u +
                      lane_idx;
                  const uint32_t record = emit_smem_records[local_pos];
                  const uint32_t record_payload = (record & 0xffffff00u) >> 8;
                  uint32_t kvo = emit_window_kv_base +
                                 ((record & 0xffu) * BLOCK_KV) +
                                 math_thread_idx;
                  if (probe_group != 0) {
                    const uint32_t sup = static_cast<uint32_t>(
                        (static_cast<uint64_t>(kvo) * probe_magic) >> 42);
                    kvo += min((sup + 1) * 64u, probe_add_max);
                  }
                  store_candidate_record(&segment_val[physical_out_base + slot],
                                         &segment_idx[physical_out_base + slot],
                                         record_payload, kvo);
                }
              } else if constexpr (kPagedCandidates) {
                if (!row_overflow) {
                  const uint64_t inline_out_base =
                      static_cast<uint64_t>(row_q) *
                          kPagedCandidateInlineRecords +
                      out_base + offset;
#pragma unroll 1
                  for (int slot = 0; slot < copy_count; ++slot) {
                    const uint32_t local_pos =
                        ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + slot) *
                            32u +
                        lane_idx;
                    const uint32_t record = emit_smem_records[local_pos];
                    const uint32_t record_payload = (record & 0xffffff00u) >> 8;
                    uint32_t kvo = emit_window_kv_base +
                                   ((record & 0xffu) * BLOCK_KV) +
                                   math_thread_idx;
                    if (probe_group != 0) {
                      const uint32_t sup = static_cast<uint32_t>(
                          (static_cast<uint64_t>(kvo) * probe_magic) >> 42);
                      kvo += min((sup + 1) * 64u, probe_add_max);
                    }
                    store_candidate_record(&cand_val[inline_out_base + slot],
                                           &cand_idx[inline_out_base + slot],
                                           record_payload, kvo);
                  }
                } else {
                  const int row_page_ordinal0 =
                      __shfl_sync(FULL, my_page_ordinal0, i);
                  const int row_page0 = __shfl_sync(FULL, my_page0, i);
                  const int row_page1 = __shfl_sync(FULL, my_page1, i);
                  const uint32_t local_pos =
                      ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots) * 32u +
                      lane_idx;
                  const uint64_t row_out_base = static_cast<uint64_t>(row_q) *
                                                kPagedCandidateInlineRecords;
                  paged_candidate_store_prepared_span(
                      &emit_smem_records[local_pos], copy_count,
                      static_cast<uint32_t>(out_base +
                                            static_cast<int>(offset)),
                      emit_window_kv_base, probe_group, probe_magic,
                      probe_add_max, &cand_val[row_out_base],
                      &cand_idx[row_out_base], overflow_val, overflow_idx,
                      row_page_ordinal0, row_page0, row_page1);
                }
              } else {
                if constexpr (kFlushMode == 2) {
                #pragma unroll 1
                for (int slot = 0; slot < copy_count; ++slot) {
                  const uint32_t local_pos =
                      ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + slot) * 32u +
                      lane_idx;
                  const uint32_t record = emit_smem_records[local_pos];
                  if constexpr (kRingRefresh) {
                    emit_smem_records[local_pos] = kRingSentinel;
                  }
                  const int out = out_base + static_cast<int>(offset) + slot;
                  const uint32_t record_payload = (record & 0xffffff00u) >> 8;
                  uint32_t kvo = emit_window_kv_base +
                                 ((record & 0xffu) * BLOCK_KV) +
                                 math_thread_idx;
                  if (probe_group != 0) {
                    const uint32_t sup = static_cast<uint32_t>(
                        (static_cast<uint64_t>(kvo) * probe_magic) >> 42);
                    kvo += min((sup + 1) * 64u, probe_add_max);
                  }
                  const uint64_t out_pos =
                      static_cast<uint64_t>(row_q) * cand_cap + out;
                  store_candidate_record(&cand_val[out_pos], &cand_idx[out_pos],
                                         record_payload, kvo);
                }
                } else {
                for (int slot = 0; slot < copy_count; ++slot) {
                  const uint32_t local_pos =
                      ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + slot) * 32u +
                      lane_idx;
                  const uint32_t record = emit_smem_records[local_pos];
                  if constexpr (kRingRefresh) {
                    emit_smem_records[local_pos] = kRingSentinel;
                  }
                  const int out = out_base + static_cast<int>(offset) + slot;
                  const uint32_t record_payload = (record & 0xffffff00u) >> 8;
                  uint32_t kvo = emit_window_kv_base +
                                 ((record & 0xffu) * BLOCK_KV) +
                                 math_thread_idx;
                  if (probe_group != 0) {
                    const uint32_t sup = static_cast<uint32_t>(
                        (static_cast<uint64_t>(kvo) * probe_magic) >> 42);
                    kvo += min((sup + 1) * 64u, probe_add_max);
                  }
                  const uint64_t out_pos =
                      static_cast<uint64_t>(row_q) * cand_cap + out;
                  store_candidate_record(&cand_val[out_pos], &cand_idx[out_pos],
                                         record_payload, kvo);
                }
                }
                if constexpr (kRingRefresh) {
                  // A capped row copies fewer slots than it
                  // holds; the leftovers must not leak into
                  // the next flush epoch.
                  for (int slot = copy_count; slot < static_cast<int>(count);
                       ++slot) {
                    emit_smem_records
                        [((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + slot) *
                             32u +
                         lane_idx] = kRingSentinel;
                  }
                }
              }
            }
            if constexpr (kRingRefresh) {
              // Publish the flush epoch: every drain store above
              // is visible before the bump, and the bump is
              // visible before any next-window append.
              __syncwarp();
              __threadfence_block();
              if (lane_idx < BLOCK_Q) {
                ring_seq[warp_idx * BLOCK_Q + lane_idx] += 1u;
              }
              __threadfence_block();
            }
          }
          emit_lane_counts.clear();
        }
      }

      // Every actual chunk was published at its final KV block.
      empty_q_barriers[0]->arrive();
    }

    // Signal the refresh daemon, then free tensor memory.
#if defined(LISA_SCAN_FINAL)
    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (warp_idx == 0 && lane_idx < BLOCK_Q && block_q_idx*BLOCK_Q+lane_idx < seq_len) {
      // Any published edge is a conservative exact-K certificate. A stale
      // read only retains extra candidates; no tighter edge is invented.
      th_bucket[block_q_idx*BLOCK_Q+lane_idx] =
          static_cast<int>(__uint_as_float(ptx::ld_shared(sparse_gate_bits+lane_idx)))-1;
    }
#else
    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
#endif
    if constexpr (!kStaticHotNoHist || kRingRefresh) {
      if (threadIdx.x == 0) {
        __threadfence_block();
        *scan_done_flag = 1;
      }
    }
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(ptx::ld_shared(tmem_ptr_in_smem), kNumTmemCols);
  }
}


} // namespace dsa_litetopk
