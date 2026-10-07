#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

// radix select: used when cub DeviceTopK is not available (HIP without cub, CCCL < 3.4.3 e.g. CUDA 13.3)
#if !defined(CUB_TOP_K_AVAILABLE) && (!defined(GGML_USE_HIP) || !defined(GGML_CUDA_USE_CUB))
#    define GGML_CUDA_TOP_K_RADIX
#endif

#ifdef GGML_CUDA_TOP_K_RADIX

// max ties at the k-th value kept per row; more ties use an ordered scan in top_k_radix_finalize
#define TOP_K_RADIX_EQ_CAP 1024

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    // -0.0 == +0.0, same as the cub argsort fallback
    const uint32_t bits = __float_as_uint(value == 0.0f ? 0.0f : value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        int * __restrict__ eq_idx,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    int * row_eq  = eq_idx + (size_t) row * TOP_K_RADIX_EQ_CAP;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            // top_k_radix_finalize picks which ties to keep
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < TOP_K_RADIX_EQ_CAP) {
                row_eq[pos] = col;
            }
        }
    }
}

// one block per row: add the lowest-index ties, sort by (value desc, index asc) - same result as a stable argsort
template<int BLOCK_SIZE>
static __global__ void top_k_radix_finalize(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const int * __restrict__ eq_idx,
        const top_k_radix_state * __restrict__ states,
        int ncols,
        int k) {
    extern __shared__ int smem[];
    uint32_t * keys = (uint32_t *) smem;          // [k]
    int      * idxs = smem + k;                   // [k]
    int      * eq   = smem + 2*k;                 // [TOP_K_RADIX_EQ_CAP]
    __shared__ int taken;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    const top_k_radix_state state = states[row];
    const int n_gt = k - state.rank;              // elements strictly greater than the k-th value

    for (int i = tid; i < n_gt; i += BLOCK_SIZE) {
        idxs[i] = row_dst[i];
        keys[i] = top_k_float_to_ordered(row_src[idxs[i]]);
    }

    if (state.equal_count <= TOP_K_RADIX_EQ_CAP) {
        const int n_eq = state.equal_count;
        for (int i = tid; i < n_eq; i += BLOCK_SIZE) {
            eq[i] = eq_idx[(size_t) row * TOP_K_RADIX_EQ_CAP + i];
        }
        __syncthreads();
        // keep the state.rank lowest indices among the ties
        for (int i = tid; i < n_eq; i += BLOCK_SIZE) {
            int pos = 0;
            for (int j = 0; j < n_eq; ++j) {
                pos += eq[j] < eq[i];
            }
            if (pos < state.rank) {
                idxs[n_gt + pos] = eq[i];
                keys[n_gt + pos] = state.prefix;
            }
        }
    } else {
        // many ties (e.g. masked -inf logits): ordered scan, lowest indices first
        if (tid == 0) {
            taken = 0;
        }
        __syncthreads();
        for (int col0 = 0; col0 < ncols; col0 += BLOCK_SIZE) {
            if (taken >= state.rank) {
                break; // uniform: taken is only written by thread 0 between barriers
            }
            const int col = col0 + tid;
            const bool hit = col < ncols && top_k_float_to_ordered(row_src[col]) == state.prefix;
            if (__syncthreads_count(hit) > 0) {
                eq[tid] = hit;
                __syncthreads();
                if (hit) {
                    int pos = taken;
                    for (int j = 0; j < tid; ++j) {
                        pos += eq[j];
                    }
                    if (pos < state.rank) {
                        idxs[n_gt + pos] = col;
                        keys[n_gt + pos] = state.prefix;
                    }
                }
                __syncthreads();
                if (tid == 0) {
                    int n = 0;
                    for (int j = 0; j < BLOCK_SIZE; ++j) {
                        n += eq[j];
                    }
                    taken += n;
                }
            }
            __syncthreads();
        }
    }
    __syncthreads();

    // rank sort: value desc, index asc
    for (int i = tid; i < k; i += BLOCK_SIZE) {
        int pos = 0;
        for (int j = 0; j < k; ++j) {
            pos += keys[j] > keys[i] || (keys[j] == keys[i] && idxs[j] < idxs[i]);
        }
        row_dst[pos] = idxs[i];
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    ggml_cuda_pool_alloc<int> eq_alloc(pool, (size_t) nrows * TOP_K_RADIX_EQ_CAP);

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, eq_alloc.get(), states, ncols, k, blocks_per_row);

    const size_t smem = (2 * (size_t) k + TOP_K_RADIX_EQ_CAP) * sizeof(int);
    top_k_radix_finalize<BLOCK_SIZE>
        <<<nrows, BLOCK_SIZE, smem, stream>>>(src, dst, eq_alloc.get(), states, ncols, k);
}

// GGML_CUDA_TOPK_RADIX=0 uses argsort; finalize sorts in O(k^2), so large k stays on argsort
static bool top_k_use_radix(int64_t ncols, int64_t k) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_TOPK_RADIX");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled && ncols > 1024 && k <= 256;
}

#endif // GGML_CUDA_TOP_K_RADIX

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if (top_k_use_radix(ncols, k)) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    // bitonic argsort needs the whole row in shared memory: radix select is the only option for long rows
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
    }
#endif
}
