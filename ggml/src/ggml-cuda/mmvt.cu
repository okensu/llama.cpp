#include "mmvt.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

// Matrix-vector product for 2..8 columns with int8 tensor cores (mma.m16n8k32).
// The activations get the same q8_1 quantization as MMVQ, but are stored in mma fragment order: for each 32-value
// sub-block, 8 columns x 32 bytes, plus one float scale per column. One mma gives the exact integer dot product of
// 16 rows x 8 columns for one sub-block; the float scaling is done once per sub-block (MMVQ: once per 8 values),
// so the results differ from MMVQ only by float rounding.
// Persistent CTAs stream 16-row weight tiles through shared memory with cp.async; the warps of a CTA split the
// superblocks of K and their sums are added in a fixed order, so the results are deterministic.

#define MMVT_NCOLS   8
#define MMVT_NWARPS  4
#define MMVT_NSTAGES 2

static __global__ void mmvt_quantize(
        const float * __restrict__ x, int8_t * __restrict__ yq, float * __restrict__ yd, const int64_t s11, const int ncols) {
    const int ib   = blockIdx.x; // 32-value sub-block
    const int col  = threadIdx.y;
    const int lane = threadIdx.x;

    const float xi = col < ncols ? x[col*s11 + ib*QK8_1 + lane] : 0.0f;
    const float amax = warp_reduce_max<WARP_SIZE>(fabsf(xi));

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    yq[(ib*MMVT_NCOLS + col)*QK8_1 + lane] = q;
    if (lane == 0) {
        // MMVQ reads the scale back from the half in block_q8_1::ds
        yd[ib*MMVT_NCOLS + col] = __half2float(__float2half(d));
    }
}

static __device__ __forceinline__ void mmvt_mma(int (&c)[4], const int a0, const int a1, const int a2, const int a3, const int b0, const int b1) {
#if defined(AMPERE_MMA_AVAILABLE)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static __device__ __forceinline__ void mmvt_cp16(void * dst, const void * src) {
#if defined(CP_ASYNC_AVAILABLE)
    const unsigned dst_s = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_s), "l"(src));
#else
    *(int4 *) dst = *(const int4 *) src;
#endif // defined(CP_ASYNC_AVAILABLE)
}

static __device__ __forceinline__ void mmvt_commit() {
#if defined(CP_ASYNC_AVAILABLE)
    asm volatile("cp.async.commit_group;");
#endif // defined(CP_ASYNC_AVAILABLE)
}

template <int n>
static __device__ __forceinline__ void mmvt_wait() {
#if defined(CP_ASYNC_AVAILABLE)
    asm volatile("cp.async.wait_group %0;" :: "n"(n));
#endif // defined(CP_ASYNC_AVAILABLE)
}

// 6-bit scale and min of sub-block s (same as get_scale_min_k4)
static __device__ __forceinline__ void mmvt_scale_min_k4(const uint8_t * q, const int s, int & sc, int & m) {
    if (s < 4) {
        sc = q[s] & 63;
        m  = q[s + 4] & 63;
    } else {
        sc = (q[s + 4] & 0xF) | ((q[s - 4] >> 6) << 4);
        m  = (q[s + 4] >>  4) | ((q[s]     >> 6) << 4);
    }
}

// Per-type access to one superblock of a row in shared memory. load(): data shared by the 8 sub-blocks,
// dec(): int8 values 8t..8t+7 of sub-block s (two ints) and its 6-bit scale and min.
template <ggml_type type> struct mmvt_row;

template <> struct mmvt_row<GGML_TYPE_Q5_K> {
    static constexpr int  bs      = sizeof(block_q5_K);
    static constexpr bool has_min = true;
    static constexpr bool paired  = true; // lane t holds values 8t..8t+7
    int4 h;  // dm + scales
    int2 qh;
    __device__ __forceinline__ void load(const char * p, const int t) {
        h  = *(const int4 *) p;
        qh = *(const int2 *) (p + 16 + 8*t);
    }
    __device__ __forceinline__ void dec(const char * p, const int t, const int s, int & lo, int & hi, int & sc, int & m) const {
        const int2 q  = *(const int2 *) (p + 48 + 32*(s/2) + 8*t);
        const int  sh = 4*(s & 1);
        lo = ((q.x >> sh) & 0x0F0F0F0F) | (((qh.x >> s) << 4) & 0x10101010);
        hi = ((q.y >> sh) & 0x0F0F0F0F) | (((qh.y >> s) << 4) & 0x10101010);
        mmvt_scale_min_k4((const uint8_t *) &h + 4, s, sc, m);
    }
    __device__ __forceinline__ float2 dm() const {
        return __half22float2(*(const half2 *) &h.x);
    }
};

template <> struct mmvt_row<GGML_TYPE_Q4_K> {
    static constexpr int  bs      = sizeof(block_q4_K);
    static constexpr bool has_min = true;
    static constexpr bool paired  = true;
    int4 h;  // dm + scales
    __device__ __forceinline__ void load(const char * p, const int t) {
        GGML_UNUSED(t);
        h = *(const int4 *) p;
    }
    __device__ __forceinline__ void dec(const char * p, const int t, const int s, int & lo, int & hi, int & sc, int & m) const {
        const int2 q  = *(const int2 *) (p + 16 + 32*(s/2) + 8*t);
        const int  sh = 4*(s & 1);
        lo = (q.x >> sh) & 0x0F0F0F0F;
        hi = (q.y >> sh) & 0x0F0F0F0F;
        mmvt_scale_min_k4((const uint8_t *) &h + 4, s, sc, m);
    }
    __device__ __forceinline__ float2 dm() const {
        return __half22float2(*(const half2 *) &h.x);
    }
};


template <> struct mmvt_row<GGML_TYPE_IQ4_XS> {
    static constexpr int  bs      = sizeof(block_iq4_xs);
    static constexpr bool has_min = false;
    static constexpr bool paired  = false; // lane t holds values 4t..4t+3 and 16+4t..16+4t+3
    int2 h; // d, scales_h, scales_l
    __device__ __forceinline__ void load(const char * p, const int t) {
        GGML_UNUSED(t);
        h = *(const int2 *) p;
    }
    __device__ __forceinline__ void dec(const char * p, const int t, const int s, int & lo, int & hi, int & sc, int & m) const {
        const int2 v = get_int_from_table_16(*(const int *) (p + 8 + 16*s + 4*t), kvalues_iq4nl);
        lo = v.x;
        hi = v.y;
        const uint32_t scales_h = (uint32_t) h.x >> 16;
        const uint32_t scales_l = (uint32_t) h.y;
        sc = (int) (((scales_l >> (4*s)) & 0xF) | (((scales_h >> (2*s)) & 3) << 4)) - 32;
        m  = 0;
    }
    __device__ __forceinline__ float2 dm() const {
        return make_float2(__half2float(*(const half *) &h.x), 0.0f);
    }
};

// Partial sums of one superblock (warp w of the stage) for rows g and g + 8 of the tile, columns 2t and 2t + 1.
template <ggml_type type>
static __device__ __forceinline__ void mmvt_superblock(
        const char * xa, const char * xb, const int2 * sq, const float * sd, const int g, const int t, const int lane, float (&acc)[4]) {
    mmvt_row<type> ra;
    mmvt_row<type> rb;
    ra.load(xa, t);
    rb.load(xb, t);

    float sum_d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int s = 0; s < QK_K/QK8_1; ++s) {
        // lane (g, t): B = 8 bytes of column g, A = the same 8 values of rows g and g + 8
        int2 b;
        if constexpr (mmvt_row<type>::paired) {
            b = sq[s*WARP_SIZE + lane];
        } else {
            const int * bc = (const int *) (sq + s*WARP_SIZE + 4*g);
            b = make_int2(bc[t], bc[t + 4]);
        }
        const float2 d8 = *(const float2 *) (sd + s*MMVT_NCOLS + 2*t);

        int a0, a1, a2, a3, sca, ma, scb, mb;
        ra.dec(xa, t, s, a0, a2, sca, ma);
        rb.dec(xb, t, s, a1, a3, scb, mb);

        int ci[4] = {0, 0, 0, 0};
        mmvt_mma(ci, a0, a1, a2, a3, b.x, b.y);
        sum_d[0] += d8.x * (float) (ci[0] * sca);
        sum_d[1] += d8.y * (float) (ci[1] * sca);
        sum_d[2] += d8.x * (float) (ci[2] * scb);
        sum_d[3] += d8.y * (float) (ci[3] * scb);

        if constexpr (mmvt_row<type>::has_min) {
            int cu[4] = {0, 0, 0, 0};
            mmvt_mma(cu, 0x01010101, 0x01010101, 0x01010101, 0x01010101, b.x, b.y); // column sums for the min term
            sum_m[0] += d8.x * (float) (cu[0] * ma);
            sum_m[1] += d8.y * (float) (cu[1] * ma);
            sum_m[2] += d8.x * (float) (cu[2] * mb);
            sum_m[3] += d8.y * (float) (cu[3] * mb);
        }
    }
    const float2 dma = ra.dm();
    const float2 dmb = rb.dm();
    acc[0] += dma.x*sum_d[0] - dma.y*sum_m[0];
    acc[1] += dma.x*sum_d[1] - dma.y*sum_m[1];
    acc[2] += dmb.x*sum_d[2] - dmb.y*sum_m[2];
    acc[3] += dmb.x*sum_d[3] - dmb.y*sum_m[3];
}

// glu == false: dst = x0 * y. glu == true: dst = silu(x0 * y) * (x1 * y) with x0 = gate, x1 = up (same shape);
// each tile then streams the gate superblocks first and the up superblocks second.
template <ggml_type type0, ggml_type type1, bool glu, int nwarps, int nstages>
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void mul_mat_vec_t(
        const char * __restrict__ x0, const char * __restrict__ x1, const int8_t * __restrict__ yq, const float * __restrict__ yd,
        float * __restrict__ dst, const int ncols_x, const int nrows, const int64_t stride_row_x0, const int64_t stride_row_x1,
        const int ncols_dst, const int64_t stride_col_dst) {
    constexpr int bs0    = mmvt_row<type0>::bs;
    constexpr int bs1    = mmvt_row<type1>::bs;
    constexpr int bs_max = bs0 > bs1 ? bs0 : bs1;
    constexpr int row_sz = nwarps*bs_max + 16;                      // padded row of one stage in shared memory
    constexpr int x_sz   = 16*row_sz;
    constexpr int yq_sz  = nwarps*(QK_K/QK8_1)*MMVT_NCOLS*QK8_1;
    constexpr int yd_sz  = nwarps*(QK_K/QK8_1)*MMVT_NCOLS*sizeof(float);
    constexpr int st_sz  = x_sz + yq_sz + yd_sz;
    constexpr int nparts = glu ? 2 : 1;

    extern __shared__ __align__(16) char smem[];
    float * red = (float *) (smem + nstages*st_sz);

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int tid  = WARP_SIZE*w + lane;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    const int nsb      = ncols_x / QK_K;
    const int nst      = (nsb + nwarps - 1) / nwarps;
    const int nst_tile = nparts*nst;
    const int ntiles   = (nrows + 15) / 16;
    const int my_tiles = ntiles > (int) blockIdx.x ? (ntiles - 1 - blockIdx.x) / gridDim.x + 1 : 0;
    const int nq       = my_tiles * nst_tile;

    auto load_stage = [&](const int q) {
        const int  tile = blockIdx.x + (q / nst_tile) * gridDim.x;
        const int  qt   = q % nst_tile;
        const bool p1   = glu && qt >= nst;
        const int  kb0  = (qt - (p1 ? nst : 0)) * nwarps;
        const int  nh   = min(nwarps, nsb - kb0);
        const int  bs   = p1 ? bs1 : bs0;
        const char *  x            = p1 ? x1 : x0;
        const int64_t stride_row_x = p1 ? stride_row_x1 : stride_row_x0;
        char * buf = smem + (q % nstages) * st_sz;

        const int cpr      = nwarps*bs/16;
        const int cpr_here = nh*bs/16;
        for (int c = tid; c < 16*cpr; c += nwarps*WARP_SIZE) {
            const int r = c / cpr;
            const int o = c - r*cpr;
            if (o < cpr_here) {
                const int row = min(16*tile + r, nrows - 1);
                mmvt_cp16(buf + r*row_sz + 16*o, x + (row*stride_row_x + kb0)*bs + 16*o);
            }
        }
        const char * yq_src = (const char *) yq + (size_t) kb0*(QK_K/QK8_1)*MMVT_NCOLS*QK8_1;
        for (int c = tid; c < nh*(QK_K/QK8_1)*MMVT_NCOLS*QK8_1/16; c += nwarps*WARP_SIZE) {
            mmvt_cp16(buf + x_sz + 16*c, yq_src + 16*c);
        }
        const char * yd_src = (const char *) (yd + (size_t) kb0*(QK_K/QK8_1)*MMVT_NCOLS);
        for (int c = tid; c < nh*(QK_K/QK8_1)*MMVT_NCOLS*(int) sizeof(float)/16; c += nwarps*WARP_SIZE) {
            mmvt_cp16(buf + x_sz + yq_sz + 16*c, yd_src + 16*c);
        }
    };

#pragma unroll
    for (int s = 0; s < nstages - 1; ++s) {
        if (s < nq) {
            load_stage(s);
        }
        mmvt_commit();
    }

    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int q = 0; q < nq; ++q) {
        mmvt_wait<nstages - 2>();
        __syncthreads();
        if (q + nstages - 1 < nq) {
            load_stage(q + nstages - 1);
        }
        mmvt_commit();

        const int  qt = q % nst_tile;
        const bool p1 = glu && qt >= nst;
        const int  kb = (qt - (p1 ? nst : 0))*nwarps + w;
        if (kb < nsb) {
            const char  * buf = smem + (q % nstages) * st_sz;
            const int2  * sq  = (const int2  *) (buf + x_sz) + w*(QK_K/QK8_1)*WARP_SIZE;
            const float * sd  = (const float *) (buf + x_sz + yq_sz) + w*(QK_K/QK8_1)*MMVT_NCOLS;
            if (p1) {
                mmvt_superblock<type1>(buf + w*bs1 + g*row_sz, buf + w*bs1 + (g + 8)*row_sz, sq, sd, g, t, lane, acc1);
            } else {
                mmvt_superblock<type0>(buf + w*bs0 + g*row_sz, buf + w*bs0 + (g + 8)*row_sz, sq, sd, g, t, lane, acc0);
            }
        }

        if (qt == nst_tile - 1) {
            // tile done: add the partial sums of the warps in fixed order and write the 16 rows
            const int row0 = 16*(blockIdx.x + (q / nst_tile) * gridDim.x);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                red[(w*8 + i)*WARP_SIZE + lane] = acc0[i];
                if constexpr (glu) {
                    red[(w*8 + 4 + i)*WARP_SIZE + lane] = acc1[i];
                }
            }
            __syncthreads();
            if (w == 0) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int v = 1; v < nwarps; ++v) {
                        acc0[i] += red[(v*8 + i)*WARP_SIZE + lane];
                        if constexpr (glu) {
                            acc1[i] += red[(v*8 + 4 + i)*WARP_SIZE + lane];
                        }
                    }
                    if constexpr (glu) {
                        acc0[i] = ggml_cuda_op_silu_single(acc0[i]) * acc1[i];
                    }
                }
                const int c0 = 2*t;
                const int c1 = 2*t + 1;
                if (row0 + g < nrows) {
                    if (c0 < ncols_dst) dst[c0*stride_col_dst + row0 + g] = acc0[0];
                    if (c1 < ncols_dst) dst[c1*stride_col_dst + row0 + g] = acc0[1];
                }
                if (row0 + g + 8 < nrows) {
                    if (c0 < ncols_dst) dst[c0*stride_col_dst + row0 + g + 8] = acc0[2];
                    if (c1 < ncols_dst) dst[c1*stride_col_dst + row0 + g + 8] = acc0[3];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc0[i] = 0.0f;
                acc1[i] = 0.0f;
            }
        }
    }
    mmvt_wait<0>();
}

static bool ggml_cuda_mmvt_type_ok(const ggml_tensor * w) {
    switch (w->type) {
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q4_K:
            return true;
        case GGML_TYPE_IQ4_XS:
            // 136-byte blocks: stages of MMVT_NWARPS superblocks are 16-byte aligned only for full stages
            return (w->ne[0]/QK_K) % MMVT_NWARPS == 0;
        default:
            return false;
    }
}

bool ggml_cuda_should_use_mmvt(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MMVT");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled || !GGML_CUDA_CC_IS_NVIDIA(cc) || ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    return ggml_cuda_mmvt_type_ok(src0) && (uintptr_t) src0->data % 16 == 0 &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= 2 && src1->ne[1] <= MMVT_NCOLS &&
        ggml_is_matrix(src0) && ggml_is_matrix(src1) && ggml_is_contiguous(src0) &&
        src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) && src0->ne[1] >= 16;
}

bool ggml_cuda_should_use_mmvt_glu(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * glu, int cc) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MMVT_GLU");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled) {
        return false;
    }
    const ggml_tensor * w0 = gate->src[0];
    const ggml_tensor * w1 = up->src[0];
    const bool pair_ok = (w0->type == GGML_TYPE_IQ4_XS && w1->type == GGML_TYPE_Q5_K) || (w0->type == w1->type);
    return pair_ok && gate->src[1] == up->src[1] && ggml_are_same_shape(w0, w1) &&
        ggml_cuda_should_use_mmvt(w0, gate->src[1], gate, cc) && ggml_cuda_should_use_mmvt(w1, up->src[1], up, cc) &&
        glu->type == GGML_TYPE_F32 && glu->nb[0] == sizeof(float) && ggml_is_matrix(glu) &&
        ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU && ggml_get_op_params_i32(glu, 1) == 0 &&
        glu->src[0] == gate && glu->src[1] == up;
}

template <ggml_type type0, ggml_type type1, bool glu>
static void mul_mat_vec_t_cuda(
        const char * x0, const char * x1, const int8_t * yq, const float * yd, float * dst, const int ncols_x, const int nrows,
        const int64_t stride_row_x0, const int64_t stride_row_x1, const int ncols_dst, const int64_t stride_col_dst, cudaStream_t stream) {
    constexpr int nwarps  = MMVT_NWARPS;
    constexpr int nstages = MMVT_NSTAGES;
    constexpr size_t bs_max = mmvt_row<type0>::bs > mmvt_row<type1>::bs ? mmvt_row<type0>::bs : mmvt_row<type1>::bs;
    constexpr size_t st_sz  = 16*(nwarps*bs_max + 16) + nwarps*(QK_K/QK8_1)*MMVT_NCOLS*(QK8_1 + sizeof(float));
    const size_t smem = nstages*st_sz + nwarps*8*WARP_SIZE*sizeof(float);

    const int id = ggml_cuda_get_device();
    static bool smem_set[GGML_CUDA_MAX_DEVICES] = {false};
    if (!smem_set[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_vec_t<type0, type1, glu, nwarps, nstages>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        smem_set[id] = true;
    }

    const int ntiles  = (nrows + 15) / 16;
    const int nblocks = std::min(ntiles, 2*ggml_cuda_info().devices[id].nsm);

    mul_mat_vec_t<type0, type1, glu, nwarps, nstages><<<nblocks, dim3(WARP_SIZE, nwarps), smem, stream>>>(
        x0, x1, yq, yd, dst, ncols_x, nrows, stride_row_x0, stride_row_x1, ncols_dst, stride_col_dst);
    CUDA_CHECK(cudaGetLastError());
}

static void mmvt_quantize_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
        ggml_cuda_pool_alloc<int8_t> & yq, ggml_cuda_pool_alloc<float> & yd) {
    const int nsub = src1->ne[0] / QK8_1;
    yq.alloc(ctx.pool(), (size_t) nsub*MMVT_NCOLS*QK8_1);
    yd.alloc(ctx.pool(), (size_t) nsub*MMVT_NCOLS);
    mmvt_quantize<<<nsub, dim3(WARP_SIZE, MMVT_NCOLS), 0, ctx.stream()>>>(
        (const float *) src1->data, yq.get(), yd.get(), src1->nb[1] / sizeof(float), src1->ne[1]);
}

void ggml_cuda_mul_mat_vec_t(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->ne[0] % QK_K == 0);

    ggml_cuda_pool_alloc<int8_t> yq;
    ggml_cuda_pool_alloc<float>  yd;
    mmvt_quantize_cuda(ctx, src1, yq, yd);

    const char *  x              = (const char *) src0->data;
    float *       dst_d          = (float *) dst->data;
    const int     ncols_x        = src0->ne[0];
    const int     nrows          = src0->ne[1];
    const int     ncols          = src1->ne[1];
    const int64_t stride_row_x   = src0->nb[1] / ggml_type_size(src0->type);
    const int64_t stride_col_dst = dst->nb[1] / sizeof(float);
    cudaStream_t  stream         = ctx.stream();

    switch (src0->type) {
        case GGML_TYPE_Q5_K:
            mul_mat_vec_t_cuda<GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, false>(x, x, yq.get(), yd.get(), dst_d, ncols_x, nrows,
                stride_row_x, stride_row_x, ncols, stride_col_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_t_cuda<GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, false>(x, x, yq.get(), yd.get(), dst_d, ncols_x, nrows,
                stride_row_x, stride_row_x, ncols, stride_col_dst, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_t_cuda<GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, false>(x, x, yq.get(), yd.get(), dst_d, ncols_x, nrows,
                stride_row_x, stride_row_x, ncols, stride_col_dst, stream);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}

void ggml_cuda_mul_mat_vec_t_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * glu) {
    const ggml_tensor * w0   = gate->src[0];
    const ggml_tensor * w1   = up->src[0];
    const ggml_tensor * src1 = gate->src[1];
    GGML_ASSERT(w0->ne[0] % QK_K == 0);

    ggml_cuda_pool_alloc<int8_t> yq;
    ggml_cuda_pool_alloc<float>  yd;
    mmvt_quantize_cuda(ctx, src1, yq, yd);

    const char *  x0             = (const char *) w0->data;
    const char *  x1             = (const char *) w1->data;
    float *       dst_d          = (float *) glu->data;
    const int     ncols_x        = w0->ne[0];
    const int     nrows          = w0->ne[1];
    const int     ncols          = src1->ne[1];
    const int64_t stride_row_x0  = w0->nb[1] / ggml_type_size(w0->type);
    const int64_t stride_row_x1  = w1->nb[1] / ggml_type_size(w1->type);
    const int64_t stride_col_dst = glu->nb[1] / sizeof(float);
    cudaStream_t  stream         = ctx.stream();

#define MMVT_GLU_CASE(t0, t1)                                                                                         \
    if (w0->type == t0 && w1->type == t1) {                                                                            \
        mul_mat_vec_t_cuda<t0, t1, true>(x0, x1, yq.get(), yd.get(), dst_d, ncols_x, nrows, stride_row_x0, stride_row_x1, \
            ncols, stride_col_dst, stream);                                                                            \
        return;                                                                                                        \
    }
    MMVT_GLU_CASE(GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_K)
    MMVT_GLU_CASE(GGML_TYPE_Q5_K,   GGML_TYPE_Q5_K)
    MMVT_GLU_CASE(GGML_TYPE_Q4_K,   GGML_TYPE_Q4_K)
    MMVT_GLU_CASE(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS)
#undef MMVT_GLU_CASE
    GGML_ABORT("unsupported type pair");
}
