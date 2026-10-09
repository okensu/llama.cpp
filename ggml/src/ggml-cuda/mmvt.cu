#include "mmvt.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <climits>

// Matrix-vector product for 2..32 columns with int8 tensor cores (mma.m16n8k32), in groups of 8 columns.
// The activations get the same q8_1 quantization as MMVQ, but are stored in mma fragment order: for each 32-value
// sub-block, 8 columns x 32 bytes, plus one float scale per column. One mma gives the exact integer dot product of
// 16 rows x 8 columns for one sub-block; the float scaling is done once per sub-block (MMVQ: once per 8 values),
// so the results differ from MMVQ only by float rounding.
// Persistent CTAs stream 16-row weight tiles through shared memory with cp.async; the warps of a CTA split the
// superblocks of K and their sums are added in a fixed order, so the results are deterministic.

#define MMVT_NCOLS   8   // columns per mma (one column group)
#define MMVT_MAX_NCG 4   // up to 4 column groups: 32 columns
#define MMVT_NWARPS  4
#define MMVT_NSTAGES 2

template <int nc>
static __global__ void mmvt_quantize(
        const float * __restrict__ x, int8_t * __restrict__ yq, float * __restrict__ yd, const int64_t s11, const int ncols) {
    const int ib   = blockIdx.x; // 32-value sub-block
    const int col  = threadIdx.y;
    const int lane = threadIdx.x;

    const float xi = col < ncols ? x[col*s11 + ib*QK8_1 + lane] : 0.0f;
    const float amax = warp_reduce_max<WARP_SIZE>(fabsf(xi));

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    yq[(ib*nc + col)*QK8_1 + lane] = q;
    if (lane == 0) {
        // MMVQ reads the scale back from the half in block_q8_1::ds
        yd[ib*nc + col] = __half2float(__float2half(d));
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

static __device__ __forceinline__ void mmvt_cp8(void * dst, const void * src) {
#if defined(CP_ASYNC_AVAILABLE)
    const unsigned dst_s = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst_s), "l"(src));
#else
    *(int2 *) dst = *(const int2 *) src;
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

// 210-byte blocks: a superblock in shared memory is only 2-byte aligned, loads are 16 bit wide.
// Each 32-value sub-block has two scales (16 values each), so the superblock routine has its own path for this type.
template <> struct mmvt_row<GGML_TYPE_Q6_K> {
    static constexpr int  bs      = sizeof(block_q6_K);
    static constexpr bool has_min = false;
    static constexpr bool paired  = true;
};

// 8 bytes at a 2-byte aligned address
static __device__ __forceinline__ int2 mmvt_ld8_a2(const char * p) {
    const uint16_t * h = (const uint16_t *) p;
    return make_int2((int) (h[0] | ((uint32_t) h[1] << 16)), (int) (h[2] | ((uint32_t) h[3] << 16)));
}

// int8 values 8t..8t+7 of sub-block s of a q6_K superblock (minus 32), as two ints
static __device__ __forceinline__ void mmvt_dec_q6_K(const char * p, const int t, const int s, int & lo, int & hi) {
    const int ip = s >> 2;
    const int j  = s & 3;
    const int2 ql = mmvt_ld8_a2(p + 64*ip + 32*(j & 1) + 8*t);
    const int2 qh = mmvt_ld8_a2(p + 128 + 32*ip + 8*t);
    const int sl = 4*(j >> 1);
    const int sh = 2*j;
    lo = ((ql.x >> sl) & 0x0F0F0F0F) | (((qh.x >> sh) & 0x03030303) << 4);
    hi = ((ql.y >> sl) & 0x0F0F0F0F) | (((qh.y >> sh) & 0x03030303) << 4);
    lo = __vsubss4(lo, 0x20202020);
    hi = __vsubss4(hi, 0x20202020);
}

template <int ncg>
static __device__ __forceinline__ void mmvt_superblock_q6_K(
        const char * xa, const char * xb, const int2 * sq, const float * sd, const int t, const int lane, float (&acc)[4*ncg]) {
    float sum_d[4*ncg];
#pragma unroll
    for (int i = 0; i < 4*ncg; ++i) {
        sum_d[i] = 0.0f;
    }
    // lanes t = 0, 1 hold values 0..15 of a sub-block (first scale), lanes t = 2, 3 values 16..31 (second scale):
    // one mma per half, with the A values of the other half set to zero
    const bool first = t < 2;
#pragma unroll
    for (int s = 0; s < QK_K/QK8_1; ++s) {
        int a0, a1, a2, a3;
        mmvt_dec_q6_K(xa, t, s, a0, a2);
        mmvt_dec_q6_K(xb, t, s, a1, a3);
        const int is  = 8*(s >> 2) + 2*(s & 3);
        const int sa0 = ((const int8_t *) xa)[192 + is];
        const int sa1 = ((const int8_t *) xa)[192 + is + 1];
        const int sb0 = ((const int8_t *) xb)[192 + is];
        const int sb1 = ((const int8_t *) xb)[192 + is + 1];
        const int f0 = first ? a0 : 0, f1 = first ? a1 : 0, f2 = first ? a2 : 0, f3 = first ? a3 : 0;
        const int g0 = first ? 0 : a0, g1 = first ? 0 : a1, g2 = first ? 0 : a2, g3 = first ? 0 : a3;

#pragma unroll
        for (int cg = 0; cg < ncg; ++cg) {
            const int2   b  = sq[(s*ncg + cg)*WARP_SIZE + lane];
            const float2 d8 = *(const float2 *) (sd + s*MMVT_NCOLS*ncg + 8*cg + 2*t);

            int c0[4] = {0, 0, 0, 0};
            int c1[4] = {0, 0, 0, 0};
            mmvt_mma(c0, f0, f1, f2, f3, b.x, b.y);
            mmvt_mma(c1, g0, g1, g2, g3, b.x, b.y);
            sum_d[4*cg + 0] += d8.x * (float) (c0[0]*sa0 + c1[0]*sa1);
            sum_d[4*cg + 1] += d8.y * (float) (c0[1]*sa0 + c1[1]*sa1);
            sum_d[4*cg + 2] += d8.x * (float) (c0[2]*sb0 + c1[2]*sb1);
            sum_d[4*cg + 3] += d8.y * (float) (c0[3]*sb0 + c1[3]*sb1);
        }
    }
    const float da = __half2float(*(const half *) (xa + 208));
    const float db = __half2float(*(const half *) (xb + 208));
#pragma unroll
    for (int cg = 0; cg < ncg; ++cg) {
        acc[4*cg + 0] += da*sum_d[4*cg + 0];
        acc[4*cg + 1] += da*sum_d[4*cg + 1];
        acc[4*cg + 2] += db*sum_d[4*cg + 2];
        acc[4*cg + 3] += db*sum_d[4*cg + 3];
    }
}

// Partial sums of one superblock (warp w of the stage) for rows g and g + 8 of the tile and, per group of 8 columns
// cg, columns 8 cg + 2t and 8 cg + 2t + 1. The weights are decoded once for all column groups.
template <ggml_type type, int ncg>
static __device__ __forceinline__ void mmvt_superblock(
        const char * xa, const char * xb, const int2 * sq, const float * sd, const int g, const int t, const int lane,
        float (&acc)[4*ncg]) {
    if constexpr (type == GGML_TYPE_Q6_K) {
        GGML_UNUSED(g);
        mmvt_superblock_q6_K<ncg>(xa, xb, sq, sd, t, lane, acc);
    } else {
    mmvt_row<type> ra;
    mmvt_row<type> rb;
    ra.load(xa, t);
    rb.load(xb, t);

    float sum_d[4*ncg];
    float sum_m[4*ncg];
#pragma unroll
    for (int i = 0; i < 4*ncg; ++i) {
        sum_d[i] = 0.0f;
        sum_m[i] = 0.0f;
    }
#pragma unroll
    for (int s = 0; s < QK_K/QK8_1; ++s) {
        int a0, a1, a2, a3, sca, ma, scb, mb;
        ra.dec(xa, t, s, a0, a2, sca, ma);
        rb.dec(xb, t, s, a1, a3, scb, mb);

#pragma unroll
        for (int cg = 0; cg < ncg; ++cg) {
            // lane (g, t): B = 8 bytes of column 8 cg + g, A = the same 8 values of rows g and g + 8
            const int2 * sqc = sq + (s*ncg + cg)*WARP_SIZE;
            int2 b;
            if constexpr (mmvt_row<type>::paired) {
                b = sqc[lane];
            } else {
                const int * bc = (const int *) (sqc + 4*g);
                b = make_int2(bc[t], bc[t + 4]);
            }
            const float2 d8 = *(const float2 *) (sd + s*MMVT_NCOLS*ncg + 8*cg + 2*t);

            int ci[4] = {0, 0, 0, 0};
            mmvt_mma(ci, a0, a1, a2, a3, b.x, b.y);
            sum_d[4*cg + 0] += d8.x * (float) (ci[0] * sca);
            sum_d[4*cg + 1] += d8.y * (float) (ci[1] * sca);
            sum_d[4*cg + 2] += d8.x * (float) (ci[2] * scb);
            sum_d[4*cg + 3] += d8.y * (float) (ci[3] * scb);

            if constexpr (mmvt_row<type>::has_min) {
                int cu[4] = {0, 0, 0, 0};
                mmvt_mma(cu, 0x01010101, 0x01010101, 0x01010101, 0x01010101, b.x, b.y); // column sums for the min term
                sum_m[4*cg + 0] += d8.x * (float) (cu[0] * ma);
                sum_m[4*cg + 1] += d8.y * (float) (cu[1] * ma);
                sum_m[4*cg + 2] += d8.x * (float) (cu[2] * mb);
                sum_m[4*cg + 3] += d8.y * (float) (cu[3] * mb);
            }
        }
    }
    const float2 dma = ra.dm();
    const float2 dmb = rb.dm();
#pragma unroll
    for (int cg = 0; cg < ncg; ++cg) {
        acc[4*cg + 0] += dma.x*sum_d[4*cg + 0] - dma.y*sum_m[4*cg + 0];
        acc[4*cg + 1] += dma.x*sum_d[4*cg + 1] - dma.y*sum_m[4*cg + 1];
        acc[4*cg + 2] += dmb.x*sum_d[4*cg + 2] - dmb.y*sum_m[4*cg + 2];
        acc[4*cg + 3] += dmb.x*sum_d[4*cg + 3] - dmb.y*sum_m[4*cg + 3];
    }
    }
}

#define MMVT_MAX_MATS 3

struct mmvt_mat {
    const char * x;
    float      * dst;
    int64_t      stride_row_x;   // in blocks
    int64_t      stride_col_dst; // in floats
    int          nrows;
    int          tile0;          // first tile of this matrix (multi mode)
};

struct mmvt_args {
    mmvt_mat m[MMVT_MAX_MATS];
    int      ntiles;
    int      ncols_x;
    int      ncols_dst;
};

// glu == false: up to 3 matrices with the same src1, their 16-row tiles concatenated; dst_i = x_i * y.
// glu == true: m[0] = gate, m[1] = up (same shape); each tile streams the gate superblocks first and the up superblocks
// second and writes silu(gate) * up to m[0].dst.
template <ggml_type type0, ggml_type type1, ggml_type type2, bool glu, int ncg, int nwarps, int nstages>
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void mul_mat_vec_t(const __grid_constant__ mmvt_args args, const int8_t * __restrict__ yq, const float * __restrict__ yd) {
    constexpr int bs0    = mmvt_row<type0>::bs;
    constexpr int bs1    = mmvt_row<type1>::bs;
    constexpr int bs2    = mmvt_row<type2>::bs;
    constexpr int bs_01  = bs0 > bs1 ? bs0 : bs1;
    constexpr int bs_max = bs_01 > bs2 ? bs_01 : bs2;
    constexpr int row_sz = nwarps*bs_max + 16;                      // padded row of one stage in shared memory
    constexpr int x_sz   = 16*row_sz;
    constexpr int nc     = MMVT_NCOLS*ncg;
    constexpr int yq_sz  = nwarps*(QK_K/QK8_1)*nc*QK8_1;
    constexpr int yd_sz  = nwarps*(QK_K/QK8_1)*nc*sizeof(float);
    constexpr int st_sz  = x_sz + yq_sz + yd_sz;
    constexpr int nparts = glu ? 2 : 1;
    // more than 16 columns: the partial sums reuse the stage just processed, to stay within the shared memory limit
    constexpr bool red_in_stage = ncg > 2;
    constexpr int  ch = (nwarps*bs0) % 16 == 0 && (nwarps*bs1) % 16 == 0 && (nwarps*bs2) % 16 == 0 ? 16 : 8;

    extern __shared__ __align__(16) char smem[];
    float * red = (float *) (smem + nstages*st_sz);

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int tid  = WARP_SIZE*w + lane;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    const int nsb      = args.ncols_x / QK_K;
    const int nst      = (nsb + nwarps - 1) / nwarps;
    const int nst_tile = nparts*nst;
    const int my_tiles = args.ntiles > (int) blockIdx.x ? (args.ntiles - 1 - blockIdx.x) / gridDim.x + 1 : 0;
    const int nq       = my_tiles * nst_tile;

    // stage q -> matrix, tile of that matrix, first superblock
    auto stage_info = [&](const int q, int & mi, int & tile, int & kb0) {
        const int T  = blockIdx.x + (q / nst_tile) * gridDim.x;
        const int qt = q % nst_tile;
        if constexpr (glu) {
            mi   = qt >= nst;
            tile = T;
            kb0  = (qt - mi*nst) * nwarps;
        } else {
            mi   = T >= args.m[2].tile0 ? 2 : (T >= args.m[1].tile0 ? 1 : 0);
            tile = T - args.m[mi].tile0;
            kb0  = qt * nwarps;
        }
    };

    auto load_stage = [&](const int q) {
        int mi, tile, kb0;
        stage_info(q, mi, tile, kb0);
        const mmvt_mat & m = args.m[mi];
        const int bs = mi == 0 ? bs0 : (mi == 1 ? bs1 : bs2);
        const int nh = min(nwarps, nsb - kb0);
        char * buf = smem + (q % nstages) * st_sz;

        // 16-byte copies, 8-byte copies for blocks whose stages are not 16-byte aligned (q6_K)
        const int cpr      = nwarps*bs/ch;
        const int cpr_here = nh*bs/ch;
        for (int c = tid; c < 16*cpr; c += nwarps*WARP_SIZE) {
            const int r = c / cpr;
            const int o = c - r*cpr;
            if (o < cpr_here) {
                const int row = min(16*tile + r, m.nrows - 1);
                if constexpr (ch == 16) {
                    mmvt_cp16(buf + r*row_sz + 16*o, m.x + (row*m.stride_row_x + kb0)*bs + 16*o);
                } else {
                    mmvt_cp8(buf + r*row_sz + 8*o, m.x + (row*m.stride_row_x + kb0)*bs + 8*o);
                }
            }
        }
        const char * yq_src = (const char *) yq + (size_t) kb0*(QK_K/QK8_1)*nc*QK8_1;
        for (int c = tid; c < nh*(QK_K/QK8_1)*nc*QK8_1/16; c += nwarps*WARP_SIZE) {
            mmvt_cp16(buf + x_sz + 16*c, yq_src + 16*c);
        }
        const char * yd_src = (const char *) (yd + (size_t) kb0*(QK_K/QK8_1)*nc);
        for (int c = tid; c < nh*(QK_K/QK8_1)*nc*(int) sizeof(float)/16; c += nwarps*WARP_SIZE) {
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

    float acc0[4*ncg];
    float acc1[4*ncg];
#pragma unroll
    for (int i = 0; i < 4*ncg; ++i) {
        acc0[i] = 0.0f;
        acc1[i] = 0.0f;
    }

    for (int q = 0; q < nq; ++q) {
        mmvt_wait<nstages - 2>();
        __syncthreads();
        if (q + nstages - 1 < nq) {
            load_stage(q + nstages - 1);
        }
        mmvt_commit();

        int mi, tile, kb0;
        stage_info(q, mi, tile, kb0);
        const int kb = kb0 + w;
        if (kb < nsb) {
            const char  * buf = smem + (q % nstages) * st_sz;
            const int2  * sq  = (const int2  *) (buf + x_sz) + w*(QK_K/QK8_1)*ncg*WARP_SIZE;
            const float * sd  = (const float *) (buf + x_sz + yq_sz) + w*(QK_K/QK8_1)*nc;
            if (mi == 0) {
                mmvt_superblock<type0, ncg>(buf + w*bs0 + g*row_sz, buf + w*bs0 + (g + 8)*row_sz, sq, sd, g, t, lane, acc0);
            } else if (mi == 1) {
                mmvt_superblock<type1, ncg>(buf + w*bs1 + g*row_sz, buf + w*bs1 + (g + 8)*row_sz, sq, sd, g, t, lane, glu ? acc1 : acc0);
            } else {
                mmvt_superblock<type2, ncg>(buf + w*bs2 + g*row_sz, buf + w*bs2 + (g + 8)*row_sz, sq, sd, g, t, lane, acc0);
            }
        }

        if (q % nst_tile == nst_tile - 1) {
            // tile done: add the partial sums of the warps in fixed order and write the 16 rows
            constexpr int nacc = 4*ncg;
            if constexpr (red_in_stage) {
                __syncthreads();
                red = (float *) (smem + (q % nstages) * st_sz);
            }
#pragma unroll
            for (int i = 0; i < nacc; ++i) {
                red[(w*2*nacc + i)*WARP_SIZE + lane] = acc0[i];
                if constexpr (glu) {
                    red[(w*2*nacc + nacc + i)*WARP_SIZE + lane] = acc1[i];
                }
            }
            __syncthreads();
            if (w == 0) {
#pragma unroll
                for (int i = 0; i < nacc; ++i) {
#pragma unroll
                    for (int v = 1; v < nwarps; ++v) {
                        acc0[i] += red[(v*2*nacc + i)*WARP_SIZE + lane];
                        if constexpr (glu) {
                            acc1[i] += red[(v*2*nacc + nacc + i)*WARP_SIZE + lane];
                        }
                    }
                    if constexpr (glu) {
                        acc0[i] = ggml_cuda_op_silu_single(acc0[i]) * acc1[i];
                    }
                }
                const mmvt_mat & m = args.m[glu ? 0 : mi];
                const int row0 = 16*tile;
#pragma unroll
                for (int cg = 0; cg < ncg; ++cg) {
                    const int c0 = 8*cg + 2*t;
                    const int c1 = 8*cg + 2*t + 1;
                    if (row0 + g < m.nrows) {
                        if (c0 < args.ncols_dst) m.dst[c0*m.stride_col_dst + row0 + g] = acc0[4*cg + 0];
                        if (c1 < args.ncols_dst) m.dst[c1*m.stride_col_dst + row0 + g] = acc0[4*cg + 1];
                    }
                    if (row0 + g + 8 < m.nrows) {
                        if (c0 < args.ncols_dst) m.dst[c0*m.stride_col_dst + row0 + g + 8] = acc0[4*cg + 2];
                        if (c1 < args.ncols_dst) m.dst[c1*m.stride_col_dst + row0 + g + 8] = acc0[4*cg + 3];
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < nacc; ++i) {
                acc0[i] = 0.0f;
                acc1[i] = 0.0f;
            }
        }
    }
    mmvt_wait<0>();
}

// GGML_CUDA_MMVT_MAX_NCOLS=N restricts the kernel to N columns (rounded up to a multiple of 8)
static int mmvt_max_ncg() {
    static const int ncg = [] {
        const char * env = getenv("GGML_CUDA_MMVT_MAX_NCOLS");
        return env != nullptr ? std::max(1, std::min(MMVT_MAX_NCG, (atoi(env) + MMVT_NCOLS - 1) / MMVT_NCOLS)) : MMVT_MAX_NCG;
    }();
    return ncg;
}

// column groups of 8 columns for ncols columns
static int mmvt_ncg(const int64_t ncols) {
    return (int) std::min<int64_t>(MMVT_MAX_NCG, (ncols + MMVT_NCOLS - 1) / MMVT_NCOLS);
}

static bool mmvt_q6_K_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MMVT_Q6K");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

static bool ggml_cuda_mmvt_type_ok(const ggml_tensor * w) {
    switch (w->type) {
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q4_K:
            return true;
        case GGML_TYPE_IQ4_XS:
            // 136-byte blocks: stages of MMVT_NWARPS superblocks are 16-byte aligned only for full stages
            return (w->ne[0]/QK_K) % MMVT_NWARPS == 0;
        case GGML_TYPE_Q6_K:
            // 210-byte blocks: rows and stages of MMVT_NWARPS superblocks are 8-byte aligned if a row has 4k superblocks
            return mmvt_q6_K_enabled() && (w->ne[0]/QK_K) % 4 == 0 && MMVT_NWARPS % 4 == 0;
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
        src1->ne[1] >= 2 && src1->ne[1] <= MMVT_NCOLS*mmvt_max_ncg() &&
        ggml_is_matrix(src0) && ggml_is_matrix(src1) && ggml_is_contiguous(src0) &&
        src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) && src0->ne[1] >= 16 &&
        // q6_K: MMVQ is as fast up to 8 columns, mmvt replaces MMQ above
        (src0->type != GGML_TYPE_Q6_K || src1->ne[1] > 8);
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

bool ggml_cuda_should_use_mmvt_multi(const ggml_tensor * const * mm, int n, int cc) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MMVT_MULTI");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled || n < 2 || n > MMVT_MAX_MATS) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (mm[i]->src[1] != mm[0]->src[1] || !ggml_cuda_should_use_mmvt(mm[i]->src[0], mm[i]->src[1], mm[i], cc)) {
            return false;
        }
    }
    // supported type combinations (see ggml_cuda_mul_mat_vec_t_multi)
    const ggml_type t0 = mm[0]->src[0]->type;
    const ggml_type t1 = mm[1]->src[0]->type;
    const ggml_type t2 = n > 2 ? mm[2]->src[0]->type : t1;
    return t0 == GGML_TYPE_Q5_K && t1 == GGML_TYPE_Q5_K && (t2 == GGML_TYPE_Q5_K || t2 == GGML_TYPE_Q4_K);
}

template <ggml_type type0, ggml_type type1, ggml_type type2, bool glu, int ncg>
static void mul_mat_vec_t_launch(const mmvt_args & args, const int8_t * yq, const float * yd, cudaStream_t stream) {
    constexpr int nwarps  = MMVT_NWARPS;
    constexpr int nstages = MMVT_NSTAGES;
    constexpr size_t bs_01  = mmvt_row<type0>::bs > mmvt_row<type1>::bs ? mmvt_row<type0>::bs : mmvt_row<type1>::bs;
    constexpr size_t bs_max = bs_01 > mmvt_row<type2>::bs ? bs_01 : mmvt_row<type2>::bs;
    constexpr size_t st_sz  = 16*(nwarps*bs_max + 16) + nwarps*(QK_K/QK8_1)*MMVT_NCOLS*ncg*(QK8_1 + sizeof(float));
    const size_t smem = nstages*st_sz + (ncg > 2 ? 0 : nwarps*8*ncg*WARP_SIZE*sizeof(float));
    static_assert(nwarps*8*ncg*WARP_SIZE*sizeof(float) <= st_sz, "partial sums must fit in one stage");

    const int id = ggml_cuda_get_device();
    static bool smem_set[GGML_CUDA_MAX_DEVICES] = {false};
    if (!smem_set[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_vec_t<type0, type1, type2, glu, ncg, nwarps, nstages>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        smem_set[id] = true;
    }

    const int nblocks = std::min(args.ntiles, 2*ggml_cuda_info().devices[id].nsm);

    mul_mat_vec_t<type0, type1, type2, glu, ncg, nwarps, nstages><<<nblocks, dim3(WARP_SIZE, nwarps), smem, stream>>>(args, yq, yd);
    CUDA_CHECK(cudaGetLastError());
}

template <ggml_type type0, ggml_type type1, ggml_type type2, bool glu>
static void mul_mat_vec_t_cuda(const mmvt_args & args, const int8_t * yq, const float * yd, cudaStream_t stream) {
    switch (mmvt_ncg(args.ncols_dst)) {
        case 1:  mul_mat_vec_t_launch<type0, type1, type2, glu, 1>(args, yq, yd, stream); break;
        case 2:  mul_mat_vec_t_launch<type0, type1, type2, glu, 2>(args, yq, yd, stream); break;
        case 3:  mul_mat_vec_t_launch<type0, type1, type2, glu, 3>(args, yq, yd, stream); break;
        default: mul_mat_vec_t_launch<type0, type1, type2, glu, 4>(args, yq, yd, stream); break;
    }
}

static void mmvt_quantize_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
        ggml_cuda_pool_alloc<int8_t> & yq, ggml_cuda_pool_alloc<float> & yd) {
    const int nsub = src1->ne[0] / QK8_1;
    // same column group count as mul_mat_vec_t_cuda
    const int ncg = mmvt_ncg(src1->ne[1]);
    const int nc  = MMVT_NCOLS*ncg;
    yq.alloc(ctx.pool(), (size_t) nsub*nc*QK8_1);
    yd.alloc(ctx.pool(), (size_t) nsub*nc);
    const float * x   = (const float *) src1->data;
    const int64_t s11 = src1->nb[1] / sizeof(float);
    switch (ncg) {
        case 1:  mmvt_quantize<1*MMVT_NCOLS><<<nsub, dim3(WARP_SIZE, 1*MMVT_NCOLS), 0, ctx.stream()>>>(x, yq.get(), yd.get(), s11, src1->ne[1]); break;
        case 2:  mmvt_quantize<2*MMVT_NCOLS><<<nsub, dim3(WARP_SIZE, 2*MMVT_NCOLS), 0, ctx.stream()>>>(x, yq.get(), yd.get(), s11, src1->ne[1]); break;
        case 3:  mmvt_quantize<3*MMVT_NCOLS><<<nsub, dim3(WARP_SIZE, 3*MMVT_NCOLS), 0, ctx.stream()>>>(x, yq.get(), yd.get(), s11, src1->ne[1]); break;
        default: mmvt_quantize<4*MMVT_NCOLS><<<nsub, dim3(WARP_SIZE, 4*MMVT_NCOLS), 0, ctx.stream()>>>(x, yq.get(), yd.get(), s11, src1->ne[1]); break;
    }
}

// matrices of mm[] (mul_mat nodes with the same src1) -> args, tiles concatenated in order
static mmvt_args mmvt_make_args(const ggml_tensor * const * mm, int n, float * dst0 = nullptr) {
    mmvt_args args = {};
    int ntiles = 0;
    for (int i = 0; i < MMVT_MAX_MATS; ++i) {
        const ggml_tensor * node = mm[std::min(i, n - 1)];
        const ggml_tensor * w    = node->src[0];
        mmvt_mat & m = args.m[i];
        m.x              = (const char *) w->data;
        m.dst            = (float *) node->data;
        m.stride_row_x   = w->nb[1] / ggml_type_size(w->type);
        m.stride_col_dst = node->nb[1] / sizeof(float);
        m.nrows          = w->ne[1];
        m.tile0          = i < n ? ntiles : INT_MAX;
        if (i < n) {
            ntiles += (m.nrows + 15) / 16;
        }
    }
    if (dst0) {
        args.m[0].dst = dst0;
    }
    args.ntiles    = ntiles;
    args.ncols_x   = mm[0]->src[0]->ne[0];
    args.ncols_dst = mm[0]->src[1]->ne[1];
    return args;
}

void ggml_cuda_mul_mat_vec_t(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->ne[0] % QK_K == 0);
    GGML_ASSERT(dst->src[0] == src0 && dst->src[1] == src1);

    ggml_cuda_pool_alloc<int8_t> yq;
    ggml_cuda_pool_alloc<float>  yd;
    mmvt_quantize_cuda(ctx, src1, yq, yd);

    const ggml_tensor * mm[1] = { dst };
    const mmvt_args args = mmvt_make_args(mm, 1);
    cudaStream_t stream = ctx.stream();

    switch (src0->type) {
        case GGML_TYPE_Q5_K:
            mul_mat_vec_t_cuda<GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, false>(args, yq.get(), yd.get(), stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_t_cuda<GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, false>(args, yq.get(), yd.get(), stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_t_cuda<GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, false>(args, yq.get(), yd.get(), stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_t_cuda<GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, false>(args, yq.get(), yd.get(), stream);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}

void ggml_cuda_mul_mat_vec_t_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * glu) {
    GGML_ASSERT(gate->src[0]->ne[0] % QK_K == 0);

    ggml_cuda_pool_alloc<int8_t> yq;
    ggml_cuda_pool_alloc<float>  yd;
    mmvt_quantize_cuda(ctx, gate->src[1], yq, yd);

    const ggml_tensor * mm[2] = { gate, up };
    mmvt_args args = mmvt_make_args(mm, 2, (float *) glu->data);
    args.m[0].stride_col_dst = glu->nb[1] / sizeof(float);
    args.ntiles = (args.m[0].nrows + 15) / 16; // tiles cover gate and up rows together
    cudaStream_t stream = ctx.stream();

    const ggml_type t0 = gate->src[0]->type;
    const ggml_type t1 = up->src[0]->type;
#define MMVT_GLU_CASE(type0, type1)                                                         \
    if (t0 == type0 && t1 == type1) {                                                       \
        mul_mat_vec_t_cuda<type0, type1, type1, true>(args, yq.get(), yd.get(), stream);    \
        return;                                                                             \
    }
    MMVT_GLU_CASE(GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_K)
    MMVT_GLU_CASE(GGML_TYPE_Q5_K,   GGML_TYPE_Q5_K)
    MMVT_GLU_CASE(GGML_TYPE_Q4_K,   GGML_TYPE_Q4_K)
    MMVT_GLU_CASE(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS)
#undef MMVT_GLU_CASE
    GGML_ABORT("unsupported type pair");
}

void ggml_cuda_mul_mat_vec_t_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * mm, int n) {
    GGML_ASSERT(n >= 2 && n <= MMVT_MAX_MATS);

    ggml_cuda_pool_alloc<int8_t> yq;
    ggml_cuda_pool_alloc<float>  yd;
    mmvt_quantize_cuda(ctx, mm[0]->src[1], yq, yd);

    const mmvt_args args = mmvt_make_args(mm, n);
    cudaStream_t stream = ctx.stream();

    const ggml_type t2 = n > 2 ? mm[2]->src[0]->type : GGML_TYPE_Q5_K;
    if (t2 == GGML_TYPE_Q4_K) {
        mul_mat_vec_t_cuda<GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K, false>(args, yq.get(), yd.get(), stream);
    } else {
        mul_mat_vec_t_cuda<GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, false>(args, yq.get(), yd.get(), stream);
    }
}
