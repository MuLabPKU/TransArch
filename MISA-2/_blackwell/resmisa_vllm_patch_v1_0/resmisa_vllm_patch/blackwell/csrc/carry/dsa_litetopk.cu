#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

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
constexpr uint32_t kCandidateIndexBits = 20;

constexpr uint32_t kCandidateIndexMask = (1u << kCandidateIndexBits) - 1u;

}

#include "../dense_topk_litetopk.cuh"

namespace  {
constexpr int kCarryTileItems = 8192;

constexpr int kCarryMaxItems = 1 << 20;

constexpr int kCarryMaxK = 12288;

constexpr int kCarryMaxVote = 8192;

constexpr int kCarryMaxBlocks =
    (kCarryMaxItems + kCarryTileItems - 1) / kCarryTileItems;

constexpr int kCarryThreads = 256;

constexpr int kCarryWarps = kCarryThreads / 32;


enum CarryStateOffset : int {
  kCarryTicket = 0,
  kCarryThreshold = 1,
  kCarryTieBlock = 2,
  kCarryTieTake = 3,
  kCarryBlockOffsets = 6,
};

constexpr int kCarryStateInts = kCarryBlockOffsets + kCarryMaxBlocks + 1;


__device__ __forceinline__ int carry_warp_sum(int value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void carry_votes_plan_litetopk_kernel(
    const int32_t* __restrict__ votes, int count, int min_index, int out_k,
    int max_vote, volatile int16_t* __restrict__ partial, int partial_stride,
    int32_t* __restrict__ state) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;
  const int begin = (0) + block * kCarryTileItems;
  const int end = min(begin + kCarryTileItems, count);
  const int bins = max_vote + 1;

  extern __shared__ uint32_t s_freq[];
  __shared__ int s_warp_sum[kCarryWarps];
  __shared__ int s_last;
  __shared__ int s_scan_base;
  __shared__ int s_found;
  __shared__ int s_threshold;
  __shared__ int s_count_gt;
  __shared__ int s_tie_block;
  __shared__ int s_tie_take;
  __shared__ int s_block_count[kCarryMaxBlocks];

  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    s_freq[bin] = 0;
  }
  __syncthreads();

  // Zero votes dominate most corpora. Count them in registers and reduce
  // once instead of serializing every zero through one shared atomic.
  int local_zero = 0;
  for (int index = begin + tid; index < end; index += kCarryThreads) {
    if (index < min_index) {
      continue;
    }
    int value = votes[index];
    // The selector emits unique winners per sampled row, so this clamp is
    // unreachable under the public ABI. Keep release builds memory-safe
    // if an upstream invariant is violated.
    value = value < 0 ? 0 : (value > max_vote ? max_vote : value);
    if (value == 0) {
      ++local_zero;
    } else {
      atomicAdd(&s_freq[value], 1u);
    }
  }
  local_zero = carry_warp_sum(local_zero);
  if (lane == 0) {
    s_warp_sum[warp] = local_zero;
  }
  __syncthreads();
  if (warp == 0) {
    int value = lane < kCarryWarps ? s_warp_sum[lane] : 0;
    value = carry_warp_sum(value);
    if (lane == 0) {
      s_freq[0] = static_cast<uint32_t>(value);
    }
  }
  __syncthreads();

  volatile int16_t* block_partial =
      partial + static_cast<size_t>(block) * partial_stride;
  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    // A CTA owns at most 8192 positions, safely inside signed int16.
    block_partial[bin] = static_cast<int16_t>(s_freq[bin]);
  }
  // Every thread publishes its own global stores. A fence in tid0 alone
  // would not release the other 255 writers before the completion ticket.
  __threadfence();
  __syncthreads();

  // CUDA's canonical "last block" reduction pattern. No CTA spins: every
  // non-last block exits, while the last ticket holder sees all partial
  // writes made visible before the atomic increment.
  if (tid == 0) {
    const int old = atomicAdd(&state[kCarryTicket], 1);
    s_last = old == gridDim.x - 1;
  }
  __syncthreads();
  if (!s_last) {
    return;
  }

  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    int total = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      total += static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride + bin]);
    }
    s_freq[bin] = static_cast<uint32_t>(total);
  }
  __syncthreads();

  // Descending 256-bin tiles. This is the seed-prep parallel prefix in the
  // opposite direction, extended to the dynamic [0,max_vote] domain.
  if (tid == 0) {
    s_scan_base = 0;
    s_found = 0;
    s_threshold = 0;
    s_count_gt = 0;
  }
  __syncthreads();
  for (int tile = 0; tile < bins; tile += kCarryThreads) {
    const int bin = max_vote - tile - tid;
    const int count_here = bin >= 0 ? static_cast<int>(s_freq[bin]) : 0;
    int inclusive = count_here;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if (lane >= offset) {
        inclusive += other;
      }
    }
    if (lane == 31) {
      s_warp_sum[warp] = inclusive;
    }
    __syncthreads();
    int warp_base = 0;
#pragma unroll
    for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
      if (source_warp < warp) {
        warp_base += s_warp_sum[source_warp];
      }
    }
    const int exclusive = s_scan_base + warp_base + inclusive - count_here;
    const int inclusive_global = exclusive + count_here;
    if (bin >= 0 && exclusive < out_k && out_k <= inclusive_global) {
      s_threshold = bin;
      s_count_gt = exclusive;
      s_found = 1;
    }
    __syncthreads();
    if (s_found) {
      break;
    }
    if (tid == 0) {
      int tile_total = 0;
#pragma unroll
      for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
        tile_total += s_warp_sum[source_warp];
      }
      s_scan_base += tile_total;
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int need_equal = out_k - s_count_gt;
    int equal_before = 0;
    s_tie_block = gridDim.x - 1;
    s_tie_take = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      const int equal_here = static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride +
                  s_threshold]);
      if (equal_before < need_equal &&
          need_equal <= equal_before + equal_here) {
        s_tie_block = source_block;
        s_tie_take = need_equal - equal_before;
        break;
      }
      equal_before += equal_here;
    }
  }
  __syncthreads();

  // Compute each block's exact stable-output size. Warps read one partial
  // row at a time so the second partial pass remains coalesced.
  for (int source_block = warp; source_block < gridDim.x;
       source_block += kCarryWarps) {
    int selected = 0;
    for (int bin = s_threshold + 1 + lane; bin < bins; bin += 32) {
      selected += static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride + bin]);
    }
    selected = carry_warp_sum(selected);
    if (lane == 0) {
      int equal_take = 0;
      if (source_block < s_tie_block) {
        equal_take = static_cast<int>(
            partial[static_cast<size_t>(source_block) * partial_stride +
                    s_threshold]);
      } else if (source_block == s_tie_block) {
        equal_take = s_tie_take;
      }
      s_block_count[source_block] = selected + equal_take;
    }
  }
  __syncthreads();

  __syncthreads();

  if (tid == 0) {
    int offset = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      state[kCarryBlockOffsets + source_block] = offset;
      offset += s_block_count[source_block];
    }
    state[kCarryBlockOffsets + gridDim.x] = offset;
    state[kCarryThreshold] = s_threshold;
    state[kCarryTieBlock] = s_tie_block;
    state[kCarryTieTake] = s_tie_take;
    __threadfence();
    atomicExch(&state[kCarryTicket], 0);
  }
}

__global__ void carry_votes_emit_reset_litetopk_kernel(
    int32_t* __restrict__ votes, int count, int min_index, int max_vote,
    int64_t* __restrict__ out_idx, const int32_t* __restrict__ state) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;
  const int begin = (0) + block * kCarryTileItems;
  const int threshold = state[kCarryThreshold];
  const int tie_block = state[kCarryTieBlock];
  const int tie_take = state[kCarryTieTake];
  const int output_base = state[kCarryBlockOffsets + block];

  __shared__ int s_warp_count[kCarryWarps];
  __shared__ int s_warp_prefix[kCarryWarps];
  __shared__ int s_tile_output_base;
  __shared__ int s_tie_seen;
  __shared__ int s_tile_total;
  if (tid == 0) {
    s_tile_output_base = 0;
    s_tie_seen = 0;
  }
  __syncthreads();

  

  constexpr unsigned kFullMask = 0xffffffffu;
  const unsigned lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
  for (int tile = 0; tile < kCarryTileItems; tile += kCarryThreads) {
    const int index = begin + tile + tid;
    const bool valid = index < count;
    const int raw_value = valid ? votes[index] : 0;
    const int value =
        raw_value < 0 ? 0 : (raw_value > max_vote ? max_vote : raw_value);
    if (valid) {
      votes[index] = 0;
    }
    const bool eligible = valid && index >= min_index;
    const bool is_equal = eligible && value == threshold;

    bool take_equal = is_equal && block < tie_block;
    if (block == tie_block) {
      const unsigned equal_mask = __ballot_sync(kFullMask, is_equal);
      if (lane == 0) {
        s_warp_count[warp] = __popc(equal_mask);
      }
      __syncthreads();
      if (tid == 0) {
        int prefix = 0;
        for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
          s_warp_prefix[source_warp] = prefix;
          prefix += s_warp_count[source_warp];
        }
        s_tile_total = prefix;
      }
      __syncthreads();
      const int equal_rank =
          s_tie_seen + s_warp_prefix[warp] + __popc(equal_mask & lane_mask);
      take_equal = is_equal && equal_rank < tie_take;
      __syncthreads();
      if (tid == 0) {
        s_tie_seen += s_tile_total;
      }
      __syncthreads();
    }

    const bool selected = eligible && (value > threshold || take_equal);
    const unsigned selected_mask = __ballot_sync(kFullMask, selected);
    if (lane == 0) {
      s_warp_count[warp] = __popc(selected_mask);
    }
    __syncthreads();
    if (tid == 0) {
      int prefix = 0;
      for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
        s_warp_prefix[source_warp] = prefix;
        prefix += s_warp_count[source_warp];
      }
      s_tile_total = prefix;
    }
    __syncthreads();
    const int local_rank =
        s_warp_prefix[warp] + __popc(selected_mask & lane_mask);
    if (selected) {
      out_idx[output_base + s_tile_output_base + local_rank] =
          static_cast<int64_t>(index);
    }
    
    __syncthreads();
    if (tid == 0) {
      s_tile_output_base += s_tile_total;
    }
    __syncthreads();
  }

  
}

__global__ void map_topk_indices_and_accumulate_votes_litetopk_kernel(
    int32_t* __restrict__ out_idx, const int32_t* __restrict__ index_map,
    const int32_t* __restrict__ status, int32_t* __restrict__ votes,
    int64_t total, int rows, int index_map_size, int topk, int votes_len,
    int vote_recent_rows) {
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t global_thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t vote_begin =
      static_cast<int64_t>(rows - vote_recent_rows) * topk;
  // A bad/underfilled row may contain selector padding rather than K valid
  // physical indices.  Preserve fail-closed behavior before treating the
  // complete output matrix as mappable winners.
  for (int row = static_cast<int>(global_thread); row < rows;
       row += static_cast<int>(step)) {
    if (status[row] != 0) {
      asm volatile("trap;");
      return;
    }
    
  }
  
  for (int64_t linear = global_thread; linear < total; linear += step) {
    const int32_t physical_idx = out_idx[linear];
    if (static_cast<uint32_t>(physical_idx) >=
        static_cast<uint32_t>(index_map_size)) {
      asm volatile("trap;");
      return;
    }
    const int32_t original_idx = index_map[physical_idx];
    if (static_cast<uint32_t>(original_idx) >=
            static_cast<uint32_t>(index_map_size) ||
        static_cast<uint32_t>(original_idx) >
            dsa_litetopk::kCandidateIndexMask) {
      asm volatile("trap;");
      return;
    }
    out_idx[linear] = original_idx;

    if (votes != nullptr && votes_len > 0) {
      // The next chunk is best predicted by the most recent query
      // window.  Mapping already visits every Q*K winner, so voting all
      // winners from the last 1536 rows only adds the atomics; it needs
      // no extra winner read or launch.
      if (linear >= vote_begin) {
        const int32_t vote_idx =
            original_idx < 0
                ? 0
                : (original_idx >= votes_len ? votes_len - 1 : original_idx);
        atomicAdd(votes + vote_idx, 1);
      }
    }
  }
}

void carry_votes_topk_reset_litetopk_(
    torch::Tensor votes, torch::Tensor out_idx, torch::Tensor partial,
    torch::Tensor state, int64_t k64, int64_t max_vote64, int64_t min_index64) {
  TORCH_CHECK(votes.is_cuda() && out_idx.is_cuda() && partial.is_cuda() &&
                  state.is_cuda(),
              "votes/out_idx/partial/state must be CUDA tensors");
  TORCH_CHECK(votes.is_contiguous() && out_idx.is_contiguous() &&
                  partial.is_contiguous() && state.is_contiguous(),
              "votes/out_idx/partial/state must be contiguous");
  TORCH_CHECK(votes.scalar_type() == torch::kInt, "votes must be int32");
  TORCH_CHECK(out_idx.scalar_type() == torch::kLong, "out_idx must be int64");
  TORCH_CHECK(partial.scalar_type() == torch::kShort, "partial must be int16");
  TORCH_CHECK(state.scalar_type() == torch::kInt, "state must be int32");
  TORCH_CHECK(votes.device() == out_idx.device() &&
                  votes.device() == partial.device() &&
                  votes.device() == state.device(),
              "votes/out_idx/partial/state must be on the same CUDA device");
  TORCH_CHECK(votes.dim() == 1, "votes must be a 1-D histogram");
  TORCH_CHECK(out_idx.dim() == 1, "out_idx must be 1-D");
  TORCH_CHECK(partial.dim() == 2, "partial must be [blocks,bins]");
  TORCH_CHECK(state.dim() == 1 && state.numel() >= kCarryStateInts,
              "state is too small for the carry top-k ABI");

  

  const int64_t count64 = votes.numel();
  TORCH_CHECK(count64 >= 1 && count64 <= kCarryMaxItems,
              "votes length must be in [1,1048576]");
  TORCH_CHECK(k64 >= 1 && k64 <= kCarryMaxK, "k must be in [1,12288]");
  TORCH_CHECK(max_vote64 >= 1 && max_vote64 <= kCarryMaxVote,
              "max_vote must be in [1,8192]");
  TORCH_CHECK(min_index64 >= 0 && min_index64 < count64,
              "min_index must be in [0,votes.numel())");
  const int count = static_cast<int>(count64);
  const int min_index = static_cast<int>(min_index64);
  const int eligible = count - min_index;
  const int out_k = static_cast<int>(min(k64, static_cast<int64_t>(eligible)));
  const int max_vote = static_cast<int>(max_vote64);
  const int bins = max_vote + 1;
  const int blocks = ((count) + kCarryTileItems - 1) /
                     kCarryTileItems;
  
  TORCH_CHECK(out_idx.numel() == out_k,
              "out_idx must have min(k,votes.numel()-min_index) elements");
  TORCH_CHECK(partial.size(0) >= blocks && partial.size(1) >= bins,
              "partial must provide at least [ceil(N/8192),max_vote+1]");

  const int partial_stride = static_cast<int>(partial.size(1));
  const size_t dynamic_smem = static_cast<size_t>(bins) * sizeof(uint32_t);
  const c10::cuda::CUDAGuard device_guard(votes.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  carry_votes_plan_litetopk_kernel
      <<<blocks, kCarryThreads, dynamic_smem, stream>>>(
          votes.data_ptr<int32_t>(), count, min_index, out_k, max_vote,
          partial.data_ptr<int16_t>(), partial_stride,
          state.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  carry_votes_emit_reset_litetopk_kernel
      <<<blocks, kCarryThreads, 0, stream>>>(
          votes.data_ptr<int32_t>(), count, min_index, max_vote,
          out_idx.data_ptr<int64_t>(), state.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void map_topk_indices_and_accumulate_votes_litetopk_(
    torch::Tensor out_idx, torch::Tensor index_map, torch::Tensor status,
    torch::Tensor votes, int64_t vote_recent_rows64) {
  TORCH_CHECK(out_idx.is_cuda() && index_map.is_cuda() && status.is_cuda() &&
                  votes.is_cuda(),
              "out_idx/index_map/status/votes must be CUDA");
  TORCH_CHECK(out_idx.device() == index_map.device() &&
                  out_idx.device() == status.device() &&
                  out_idx.device() == votes.device(),
              "out_idx/index_map/status/votes must be on one CUDA device");
  TORCH_CHECK(out_idx.is_contiguous() && index_map.is_contiguous() &&
                  status.is_contiguous() && votes.is_contiguous(),
              "out_idx/index_map/status/votes must be contiguous");
  TORCH_CHECK(out_idx.scalar_type() == torch::kInt &&
                  index_map.scalar_type() == torch::kInt &&
                  status.scalar_type() == torch::kInt &&
                  votes.scalar_type() == torch::kInt,
              "out_idx/index_map/status/votes must be int32");
  TORCH_CHECK(out_idx.dim() == 2 && out_idx.numel() > 0,
              "out_idx must be a nonempty [R,K] matrix");
  TORCH_CHECK(index_map.dim() == 1 && index_map.numel() > 0 &&
                  index_map.numel() <=
                      (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
              "index_map must be a nonempty vector with at most 1M entries");
  TORCH_CHECK(status.dim() == 1 && status.numel() == out_idx.size(0),
              "status must contain one value per output row");
  TORCH_CHECK(votes.dim() == 1, "votes must be a 1-D histogram (or empty)");
  TORCH_CHECK(vote_recent_rows64 > 0 && vote_recent_rows64 <= out_idx.size(0),
              "vote_recent_rows must be in [1,R]");

  constexpr int kThreads = 256;
  constexpr int kBlocksPerSm = 8;
  constexpr int kProductionSms = 148;
  const int64_t total = out_idx.numel();
  const int blocks = static_cast<int>(std::min<int64_t>(
      (total + kThreads - 1) / kThreads, kProductionSms * kBlocksPerSm));
  const int votes_len = static_cast<int>(votes.numel());
  const c10::cuda::CUDAGuard device_guard(out_idx.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  map_topk_indices_and_accumulate_votes_litetopk_kernel<<<blocks, kThreads, 0,
                                                          stream>>>(
      out_idx.data_ptr<int32_t>(), index_map.data_ptr<int32_t>(),
      status.data_ptr<int32_t>(),
      votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr, total,
      static_cast<int>(out_idx.size(0)), static_cast<int>(index_map.numel()),
      static_cast<int>(out_idx.size(1)), votes_len,
      static_cast<int>(vote_recent_rows64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("carry_votes_topk_reset_", &carry_votes_topk_reset_litetopk_);
  m.def("map_topk_indices_and_accumulate_votes_litetopk_", &map_topk_indices_and_accumulate_votes_litetopk_);
  m.def("dense_topk_litetopk_", &litetopk_dense::dense_topk_litetopk_);
}
