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

inline constexpr uint32_t kSpecializedRegisters = 24;
inline constexpr uint32_t kUmmaStages = 2;

// Production candidate ABI: a six-byte global record. cand_val stores the
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

template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_Q,
          uint32_t BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSpecializedThreads,
          uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128,
          uint32_t kEmitLaneSlots = 18, 
          uint32_t kTuneMathRegisters = 240, uint32_t kTuneEmitBlocks = 256,
          uint32_t kMinBlocks = 1>
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
                               const uint32_t num_kv_splits,
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
                               // Per-token FP32 dequantization scales.
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_kv_scales,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_weights) {

  const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
  constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
  constexpr uint32_t kNumUmmaStages = kUmmaStages;
  constexpr uint32_t kNumUmmaBuffers = kNumMathWarpGroups * kNumUmmaStages;
  constexpr uint32_t kNumAccumBufs = kNumUmmaBuffers;
  constexpr uint32_t kActiveEmitChunkBlocks =
      kTuneEmitBlocks;

  DG_STATIC_ASSERT(
      (BLOCK_Q == 4 || BLOCK_Q == 8 || BLOCK_Q == 16) && (kNumMathWarps == 4 || kNumMathWarps == 8),
      "LiteTopK requires BLOCK_Q in {4 (H=32), 2 (H=64)} and 8 math warps");
  DG_STATIC_ASSERT(
      (BLOCK_KV == 128 || BLOCK_KV == 256),
      "the local ring infers the KV tile coordinate from its owner lane");

  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0,
                   "Invalid threads");

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
  }

  // Two FP8 elements occupy two bytes.
  static constexpr uint32_t kKvElemBytesNum = 2u;  // per 2 elems
  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * kHeadDim * kKvElemBytesNum / 2;
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      math::constexpr_align(BLOCK_Q * kNumHeads * uint32_t(sizeof(float)), 512u);
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE =
      BLOCK_KV * kHeadDim * kKvElemBytesNum / 2;
  // One FP32 dequantization scale per token.
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

  constexpr uint32_t kAccumStride = math::constexpr_align(BLOCK_Q * kNumHeads, 32u);
  constexpr uint32_t kNumAccumTmemCols = kAccumStride * kNumAccumBufs;
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

  auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
      barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumAccumBufs * 2 +
      kNumKVStages);
  auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 4);
  // Chunked emit does not use the legacy warp-queue count words. Reuse four
  // of them as a CTA-local Gate4 mailbox. Values are positive float edge
  // bit-patterns, so unsigned atomicMin is exactly a monotonic gate tighten.
  auto sparse_gate_bits = reinterpret_cast<uint32_t*>(warpq_count);
  auto emit_smem_records =
      reinterpret_cast<uint32_t*>(warpq_count + kNumMathWarps * BLOCK_Q);
  auto smem_hist = reinterpret_cast<int32_t*>(
      emit_smem_records + kNumMathWarps * BLOCK_Q * kEmitLaneSlots * 32u);

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
      empty_kv_barriers[i]->init(kNumMathThreads);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 1) {
    if (cute::elect_one_sync()) {
#pragma unroll
      for (uint32_t i = 0; i < kNumAccumBufs; ++i) {
        full_umma_barriers[i]->init(1);
        empty_umma_barriers[i]->init(128);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    if constexpr (kMinBlocks > 1) cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  
  {
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

  }
  if (threadIdx.x < BLOCK_Q) {
    const uint32_t row_q =
        static_cast<uint32_t>(blockIdx.x) * BLOCK_Q + threadIdx.x;
    const uint32_t row = min(row_q, seq_len - 1);
    const int gate = __ldcg(th_bucket + row);
    {
      ptx::st_shared(sparse_gate_bits + threadIdx.x,
                     __float_as_uint(static_cast<float>(gate + 1)));
    }
  }
  __syncthreads();

  constexpr uint32_t kNumSpecializedRegisters = kSpecializedRegisters;
  constexpr uint32_t kNumMathRegisters = kTuneMathRegisters;

  // V1 KV-split scheduling: blockIdx.x = q-block (one per CTA), blockIdx.y =
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
    _Pragma("unroll") for (uint32_t i = 0; i < BLOCK_Q; ++i) {
      seq_k_start[i] = max(seq_k_start[i], split_lo);
      seq_k_end[i] = min(seq_k_end[i], split_hi);
    }
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
        // Q + weights once for this q-block; uint8 map extents are bytes.
        constexpr uint32_t kKvInnerBytes = kHeadDim * kKvElemBytesNum / 2;
        tma::copy<kKvInnerBytes, BLOCK_Q * kNumHeads, kKvInnerBytes>(
            &tensor_map_q, full_q_barriers[0], smem_q[0], 0,
            block_q_idx * BLOCK_Q * kNumHeads);
        tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights,
                                         full_q_barriers[0], smem_weights[0], 0,
                                         block_q_idx * BLOCK_Q);
        full_q_barriers[0]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE +
                                                 BLOCK_Q * kNumHeads * sizeof(float));

        CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
        for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
             ++kv_block_idx) {
          CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
          empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

          tma::copy<kKvInnerBytes, BLOCK_KV, kKvInnerBytes>(
              &tensor_map_kv, full_kv_barriers[kv_stage_idx],
              smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
          tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales,
                                    full_kv_barriers[kv_stage_idx],
                                    smem_kv_scales[kv_stage_idx],
                                    kv_start + kv_block_idx * BLOCK_KV, 0);
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
    // These spare warps do not participate in the math-warp flush path.
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    if (block_q_idx < num_q_blocks) {
      {
        // Math warps update the seen-candidate subset at flush epochs.
        // All CTA-wide initialization has already completed above.
        return;
      }
      
    }
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    const auto math_thread_idx = warp_idx * 32 + lane_idx;

    auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr,
                        float* accum) {
      constexpr int N = decltype(num_elems_c)::value;
      DG_STATIC_ASSERT(N == 32 or N == 64 or N == 128,
                       "Unsupported TMEM load size");
      using Loader = cute::conditional_t<
          N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
          cute::conditional_t<N == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
                              cute::SM100_TMEM_LOAD_32dp32b128x>>;
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

// Weights into registers (once per CTA -- the 2.5-generation win).
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j)
          weights[i][j] = ptx::ld_shared(smem_weights[0] + i * kNumHeads + j);
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
        {
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
      } else {
#pragma unroll
        for (uint32_t gi = 0; gi < BLOCK_Q; ++gi)
          o_reg[gi] = __uint_as_float(ptx::ld_shared(sparse_gate_bits + gi));
      }

      for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
           ++kv_block_idx) {
        const uint32_t kvg = kv_block_idx;
        CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);

        {
          if (kv_block_idx != 0 && (kv_block_idx % kActiveEmitChunkBlocks) == 0) {
            if constexpr (BLOCK_Q == 4) {
              const float4 gate = ptx::ld_shared(
                  reinterpret_cast<const float4*>(sparse_gate_bits));
              o_reg[0] = gate.x;
              o_reg[1] = gate.y;
              o_reg[2 < BLOCK_Q ? 2 : 0] = gate.z;
              o_reg[3 < BLOCK_Q ? 3 : 0] = gate.w;
            } else {
#pragma unroll
              for (uint32_t gi = 0; gi < BLOCK_Q; ++gi)
                o_reg[gi] =
                    __uint_as_float(ptx::ld_shared(sparse_gate_bits + gi));
            }
          }
        }
        full_kv_barriers[kv_stage_idx]->wait(kv_phase);
        const float scale_kv =
            ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);

        const uint32_t umma_buf =
            warpgroup_idx * kNumUmmaStages + (kvg % kNumUmmaStages);
        const uint32_t umma_wait_phase = (kvg / kNumUmmaStages) & 1;
        const auto tmem_start = ptx::ld_shared(tmem_ptr_in_smem) + umma_buf * kAccumStride;
        full_umma_barriers[umma_buf]->wait(umma_wait_phase);
        ptx::tcgen05_after_thread_sync();
        empty_kv_barriers[kv_stage_idx]->arrive();

        const auto kv_offset =
            kv_start + kv_block_idx * BLOCK_KV + math_thread_idx;
        DG_STATIC_ASSERT(kNumHeads == 8, "MISA requires fused1 plus sparse7");

        uint32_t pass_bits = 0;
        float v_row[BLOCK_Q];
        constexpr uint32_t kTmemRowsEff =
            (32u / kNumHeads) < BLOCK_Q ? (32u / kNumHeads) : BLOCK_Q;
        DG_STATIC_ASSERT(BLOCK_Q % kTmemRowsEff == 0,
                         "BLOCK_Q must divide evenly into TMEM load groups");
#define LITETOPK_SCORE_GATE(i, RC) \
  bool g; \
  const float raw = misa_score; \
  const float bq = fmaf(-raw, inv_reg[i], vth_reg[i]); \
  v_row[i] = bq; \
  g = __float_as_int(bq) < __float_as_int(o_reg[i]); \
  if constexpr (RC) g = g and ((kv_offset - kstart_reg[i]) < kspan_reg[i]); \
  pass_bits |= g ? (1u << i) : 0u;
        const uint32_t kv_base = kv_start + kv_block_idx * BLOCK_KV;
        const bool interior =
            (kv_base >= rs_max) && (kv_base + BLOCK_KV <= re_min);
#define LITETOPK_SCORE_ROWS(RANGE_CHECK)                                       \
  _Pragma("unroll") for (uint32_t pr = 0; pr < BLOCK_Q / kTmemRowsEff; ++pr) { \
    float accum2[kNumHeads * kTmemRowsEff];                                    \
    tmem_load(cute::Int<kNumHeads * kTmemRowsEff>{},                           \
              tmem_start + pr * kTmemRowsEff * kNumHeads, accum2);             \
    if (pr == BLOCK_Q / kTmemRowsEff - 1) {                                    \
      ptx::tcgen05_before_thread_sync();                                       \
      empty_umma_barriers[umma_buf]->arrive();                                 \
    }                                                                          \
    _Pragma("unroll") for (uint32_t k = 0; k < kTmemRowsEff; ++k) {            \
      const uint32_t i = pr * kTmemRowsEff + k;                                \
      const float* accum = accum2 + k * kNumHeads;                             \
      /* MISA: head 0 is linear; heads 1..7 are negative residuals. */ \
      float contribution[kNumHeads]; \
      _Pragma("unroll") for (uint32_t j = 0; j < kNumHeads; ++j) { \
        const float y = __fmul_rn(accum[j], scale_kv); \
        const float term = j == 0 ? y : fmaxf(-y, 0.0f); \
        contribution[j] = __fmul_rn(term, weights[i][j]); \
      } \
      _Pragma("unroll") for (uint32_t step = kNumHeads / 2; step > 0; step /= 2) { \
        _Pragma("unroll") for (uint32_t j = 0; j < step; ++j) \
          contribution[j] = __fadd_rn(contribution[j], contribution[j + step]); \
      } \
      const float misa_score = contribution[0]; \
      LITETOPK_SCORE_GATE(i, RANGE_CHECK)                                      \
    }                                                                          \
  }
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
              {
                {
                  const float x = v_row[i];
                  candidate_score_bits = candidate_fp24_code(x) << 8;
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
                {
                  const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                  int out;
                  {
                    out = atomicAdd(cand_cnt + row_q, 1);
                  }
                  {
                    if (out < static_cast<int>(cand_cap)) {
                      uint32_t kvo = kv_offset;
                      
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
              
            }
          }
        }

        if (((kv_block_idx + 1) % kActiveEmitChunkBlocks) == 0 ||
            kv_block_idx + 1 == num_kv_blocks) {
          {
            
            flush_native_ring<BLOCK_Q,kEmitLaneSlots,BLOCK_KV>(emit_lane_counts,
                emit_smem_records+warp_idx*BLOCK_Q*kEmitLaneSlots*32u,
                cand_val,cand_idx,cand_cnt,cand_cap,
                block_q_idx*BLOCK_Q,seq_len,
                kv_start+(kv_block_idx/kActiveEmitChunkBlocks)*kActiveEmitChunkBlocks*BLOCK_KV,
                math_thread_idx,smem_hist);
            cutlass::arch::NamedBarrier(kNumMathThreads, 1).sync();
            tighten_flush_subset<BLOCK_Q,kNumMathWarps>(smem_hist,
                sparse_gate_bits,topk,block_q_idx*BLOCK_Q,seq_len);
            cutlass::arch::NamedBarrier(kNumMathThreads, 1).sync();
          }
          emit_lane_counts.clear();
        }
      }

      // Every actual chunk was published at its final KV block.
      empty_q_barriers[0]->arrive();
    }

    // Synchronize math warps before freeing tensor memory.
    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    {
      if (threadIdx.x == 0) {
        __threadfence_block();
      }
    }
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(ptx::ld_shared(tmem_ptr_in_smem), kNumTmemCols);
  }
}

} // namespace dsa_litetopk
