// GPU-controlled recovery: bounded 64-row score cache, candidate-space search.
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <cstdint>
#include <new>

namespace {
constexpr int GROUP = 64;
// summary: active, invalid, search_error, cursor, initial_bad, coarse_rows,
//          cached_predicate_passes, groups, max_iterations, reserved...
struct Graph {
    cudaGraph_t graph{};
    cudaGraph_t body{};
    cudaGraph_t inner{};
    cudaGraphNode_t inner_node{};
    cudaGraphConditionalHandle inner_condition{};
    cudaGraphExec_t exec{};
    cudaStream_t capture_stream{};
    cudaGraphConditionalHandle condition{};
};
thread_local cudaError_t last_error = cudaSuccess;

__global__ void begin_recovery(cudaGraphConditionalHandle condition, int* summary, int rows) {
    if (threadIdx.x == 0) {
        assert(summary[0] >= 0 && summary[0] <= rows);
        assert(summary[1] == 0 && "invalid causal ranges");
        assert(summary[2] == 0 && "invalid recovery status");
        for (int i = 3; i < 16; ++i) summary[i] = 0;
        summary[4] = summary[0];
        cudaGraphSetConditional(condition, summary[0] != 0);
    }
}

// Nested GPU loop: thresholds are updated per row, counts are evaluated by
// many independent CTAs over the cached score vector (launched by Triton).
__global__ void begin_group(cudaGraphConditionalHandle condition, const int* rowmap,
                            const int* summary, uint64_t* lower, uint64_t* upper,
                            int* active, int* iterations) {
    int slot = threadIdx.x;
    int live = summary[3] + slot < summary[0];
    active[slot] = live;
    lower[slot] = 0x007fffff00000000ull;
    upper[slot] = 0xff80000000000000ull;
    iterations[slot] = 0;
    __syncthreads();
    if (slot == 0) cudaGraphSetConditional(condition, 1);
}

__global__ void update_group(const int* rowmap, int* summary, const int* starts,
                             const int* ends, float* thresholds, uint64_t* keys,
                             int* counts, uint64_t* lower, uint64_t* upper,
                             const int* active, int* iterations,
                             int n, int minimum, int target, int capacity) {
    int slot = threadIdx.x;
    if (!active[slot]) return;
    int row = rowmap[summary[3]+slot];
    int count = counts[row], start = starts[row], end = ends[row];
    uint64_t pivot = keys[row];
    uint64_t lo = lower[slot], hi = upper[slot];
    if (count > capacity) lo = max(lo, pivot+1);
    else if (count < min(minimum,end-start)) hi = min(hi, pivot-1);
    assert(lo <= hi && "no admissible threshold");
    uint64_t next = lo + ((hi-lo) >> 1);
    if (iterations[slot] == 0 && count > capacity) {
        int64_t cutoff = start + max(int64_t(1),int64_t(end-start)*target/count)-1;
        uint64_t probe = (pivot >> 32 << 32) | uint64_t(n-cutoff);
        if (probe >= lo && probe <= hi) next = probe;
    }
    lower[slot] = lo; upper[slot] = hi; keys[row] = next;
    uint32_t code = uint32_t(next >> 32);
    uint32_t bits = code & 0x80000000u ? code ^ 0x80000000u : ~code;
    thresholds[row] = __uint_as_float(bits);
    counts[row] = 0;
    int round = ++iterations[slot];
    atomicAdd(summary+6,1);
    atomicMax(summary+8,round);
}

__global__ void check_group(cudaGraphConditionalHandle condition, const int* rowmap,
                            const int* summary, const int* starts, const int* ends,
                            const int* counts, int* active, const int* iterations,
                            int* expected, int minimum, int capacity) {
    int slot = threadIdx.x;
    int again = 0;
    if (active[slot]) {
        int row = rowmap[summary[3]+slot];
        int count = counts[row];
        int required = min(minimum,ends[row]-starts[row]);
        again = count < required || count > capacity;
        assert(!again || iterations[slot] < 65);
        if (!again) { active[slot] = 0; expected[slot] = count; }
    }
    int any = __syncthreads_or(again);
    if (slot == 0) {
        assert(summary[2] == 0 && "nonfinite coarse score");
        cudaGraphSetConditional(condition,any);
    }
}

__global__ void finish_group(const int* rowmap, const int* summary, int* counts,
                             int* status, const int* active, const int* expected) {
    int slot = threadIdx.x;
    if (summary[3]+slot >= summary[0]) return;
    int row = rowmap[summary[3]+slot];
    assert(active[slot] == 0 && counts[row] == expected[slot]);
    counts[row] = 0; status[row] = 0;
}

__global__ void advance_group(cudaGraphConditionalHandle condition, const int* rowmap,
                              int* summary, const int* counts, const int* status,
                              const int* expected) {
    int slot = threadIdx.x;
    int packed = summary[3] + slot;
    if (slot < GROUP && packed < summary[0]) {
        int row = rowmap[packed];
        assert(status[row] == 0 && "recovery produced invalid status");
        assert(counts[row] == expected[slot] && "cached compaction count mismatch");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        summary[5] += min(GROUP, summary[0]-summary[3]);
        summary[3] += GROUP;
        summary[7] += 1;
        cudaGraphSetConditional(condition, summary[3] < summary[0]);
    }
}
}

extern "C" const char* hisa_last_error() { return cudaGetErrorString(last_error); }

extern "C" void* hisa_graph_create(int64_t summary_address, int rows) {
    Graph* state = new (std::nothrow) Graph;
    if (!state) return nullptr;
    auto fail = [&]() -> void* {
        if (state->graph) cudaGraphDestroy(state->graph);
        if (state->capture_stream) cudaStreamDestroy(state->capture_stream);
        delete state; return nullptr;
    };
    last_error = cudaGraphCreate(&state->graph, 0);
    if (last_error != cudaSuccess) return fail();
    last_error = cudaGraphConditionalHandleCreate(&state->condition, state->graph, 0, 0);
    if (last_error != cudaSuccess) return fail();
    last_error = cudaGraphConditionalHandleCreate(&state->inner_condition, state->graph, 0, 0);
    if (last_error != cudaSuccess) return fail();
    int* summary = reinterpret_cast<int*>(summary_address);
    void* args[] = {&state->condition, &summary, &rows};
    cudaKernelNodeParams init{};
    init.func = reinterpret_cast<void*>(begin_recovery);
    init.gridDim = dim3(1); init.blockDim = dim3(32); init.kernelParams = args;
    cudaGraphNode_t initial;
    last_error = cudaGraphAddKernelNode(&initial, state->graph, nullptr, 0, &init);
    if (last_error != cudaSuccess) return fail();
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = state->condition;
    params.conditional.type = cudaGraphCondTypeWhile;
    params.conditional.size = 1;
    cudaGraphNode_t loop;
    last_error = cudaGraphAddNode(&loop, state->graph, &initial, nullptr, 1, &params);
    if (last_error != cudaSuccess) return fail();
    state->body = params.conditional.phGraph_out[0];
    last_error = cudaStreamCreateWithFlags(&state->capture_stream, cudaStreamNonBlocking);
    if (last_error != cudaSuccess) return fail();
    return state;
}

extern "C" int64_t hisa_graph_stream(void* ptr) {
    return reinterpret_cast<int64_t>(static_cast<Graph*>(ptr)->capture_stream);
}

extern "C" int hisa_graph_begin(void* ptr) {
    auto* state = static_cast<Graph*>(ptr);
    return last_error = cudaStreamBeginCaptureToGraph(state->capture_stream, state->body,
                                                    nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal);
}

extern "C" int hisa_graph_end(void* ptr) {
    auto* state = static_cast<Graph*>(ptr);
    cudaGraph_t graph;
    last_error = cudaStreamEndCapture(state->capture_stream, &graph);
    if (last_error != cudaSuccess) return last_error;
    if (graph != state->body) return last_error = cudaErrorInvalidValue;
    return last_error = cudaGraphInstantiate(&state->exec, state->graph, 0);
}

extern "C" int hisa_graph_launch(void* ptr, int64_t stream) {
    return last_error = cudaGraphLaunch(static_cast<Graph*>(ptr)->exec, reinterpret_cast<cudaStream_t>(stream));
}

extern "C" int hisa_group_begin_launch(void* ptr, int64_t rowmap, int64_t summary,
                                       int64_t lower, int64_t upper, int64_t active,
                                       int64_t iterations, int64_t stream) {
    begin_group<<<1,GROUP,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<Graph*>(ptr)->inner_condition,reinterpret_cast<const int*>(rowmap),
        reinterpret_cast<const int*>(summary),reinterpret_cast<uint64_t*>(lower),
        reinterpret_cast<uint64_t*>(upper),reinterpret_cast<int*>(active),reinterpret_cast<int*>(iterations));
    return last_error = cudaGetLastError();
}

extern "C" int hisa_group_update_launch(int64_t rowmap, int64_t summary, int64_t starts,
                                        int64_t ends, int64_t thresholds, int64_t keys,
                                        int64_t counts, int64_t lower, int64_t upper,
                                        int64_t active, int64_t iterations,
                                        int n, int minimum, int target, int capacity, int64_t stream) {
    update_group<<<1,GROUP,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const int*>(rowmap),reinterpret_cast<int*>(summary),
        reinterpret_cast<const int*>(starts),reinterpret_cast<const int*>(ends),
        reinterpret_cast<float*>(thresholds),reinterpret_cast<uint64_t*>(keys),reinterpret_cast<int*>(counts),
        reinterpret_cast<uint64_t*>(lower),reinterpret_cast<uint64_t*>(upper),
        reinterpret_cast<const int*>(active),reinterpret_cast<int*>(iterations),n,minimum,target,capacity);
    return last_error = cudaGetLastError();
}

extern "C" int hisa_group_check_launch(void* ptr, int64_t rowmap, int64_t summary,
                                       int64_t starts, int64_t ends, int64_t counts,
                                       int64_t active, int64_t iterations, int64_t expected,
                                       int minimum, int capacity, int64_t stream) {
    check_group<<<1,GROUP,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<Graph*>(ptr)->inner_condition,reinterpret_cast<const int*>(rowmap),
        reinterpret_cast<const int*>(summary),reinterpret_cast<const int*>(starts),reinterpret_cast<const int*>(ends),
        reinterpret_cast<const int*>(counts),reinterpret_cast<int*>(active),reinterpret_cast<const int*>(iterations),
        reinterpret_cast<int*>(expected),minimum,capacity);
    return last_error = cudaGetLastError();
}

extern "C" int hisa_group_finish_launch(int64_t rowmap, int64_t summary, int64_t counts,
                                        int64_t status, int64_t active, int64_t expected, int64_t stream) {
    finish_group<<<1,GROUP,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const int*>(rowmap),reinterpret_cast<const int*>(summary),
        reinterpret_cast<int*>(counts),reinterpret_cast<int*>(status),
        reinterpret_cast<const int*>(active),reinterpret_cast<const int*>(expected));
    return last_error = cudaGetLastError();
}

// End the first outer-body segment, insert a WHILE node after its leaves, then
// capture the parallel threshold loop into that node's body.
extern "C" int hisa_inner_begin(void* ptr) {
    auto* state = static_cast<Graph*>(ptr);
    cudaGraph_t graph;
    last_error = cudaStreamEndCapture(state->capture_stream,&graph);
    if (last_error != cudaSuccess) return last_error;
    if (graph != state->body) return last_error = cudaErrorInvalidValue;
    size_t size=0;
    last_error=cudaGraphGetNodes(graph,nullptr,&size);
    if (last_error != cudaSuccess) return last_error;
    std::vector<cudaGraphNode_t> nodes(size),leaves;
    last_error=cudaGraphGetNodes(graph,nodes.data(),&size);
    if (last_error != cudaSuccess) return last_error;
    for (auto node:nodes) {
        size_t children=0;
        last_error=cudaGraphNodeGetDependentNodes(node,nullptr,nullptr,&children);
        if (last_error != cudaSuccess) return last_error;
        if (!children) leaves.push_back(node);
    }
    cudaGraphNodeParams params{};
    params.type=cudaGraphNodeTypeConditional;
    params.conditional.handle=state->inner_condition;
    params.conditional.type=cudaGraphCondTypeWhile;
    params.conditional.size=1;
    last_error=cudaGraphAddNode(&state->inner_node,graph,leaves.data(),nullptr,leaves.size(),&params);
    if (last_error != cudaSuccess) return last_error;
    state->inner=params.conditional.phGraph_out[0];
    return last_error=cudaStreamBeginCaptureToGraph(state->capture_stream,state->inner,
                                                   nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal);
}

// Resume outer-body capture with an explicit dependency on the completed
// inner loop, before compaction and advancing the 64-row cache group.
extern "C" int hisa_inner_end(void* ptr) {
    auto* state=static_cast<Graph*>(ptr);
    cudaGraph_t graph;
    last_error=cudaStreamEndCapture(state->capture_stream,&graph);
    if (last_error != cudaSuccess) return last_error;
    if (graph != state->inner) return last_error=cudaErrorInvalidValue;
    return last_error=cudaStreamBeginCaptureToGraph(state->capture_stream,state->body,
                                                   &state->inner_node,nullptr,1,cudaStreamCaptureModeThreadLocal);
}

extern "C" int hisa_advance_launch(void* ptr, int64_t rowmap, int64_t summary, int64_t counts,
                                   int64_t status, int64_t expected, int64_t stream) {
    advance_group<<<1,GROUP,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<Graph*>(ptr)->condition, reinterpret_cast<const int*>(rowmap),
        reinterpret_cast<int*>(summary),reinterpret_cast<const int*>(counts),
        reinterpret_cast<const int*>(status),reinterpret_cast<const int*>(expected));
    return last_error = cudaGetLastError();
}

extern "C" int hisa_graph_destroy(void* ptr) {
    auto* state = static_cast<Graph*>(ptr);
    if (state->exec) cudaGraphExecDestroy(state->exec);
    if (state->graph) cudaGraphDestroy(state->graph);
    if (state->capture_stream) cudaStreamDestroy(state->capture_stream);
    delete state;
    return 0;
}
