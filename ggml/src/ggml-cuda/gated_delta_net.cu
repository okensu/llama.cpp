#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     bool          ends_only) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K && (!ends_only || target_slot == 0 || target_slot == K - 1)) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// [TAG_GDN_CHUNKED] chunked prefill for the scalar-gate delta rule (S_v = 128, single sequence).
// Within a chunk of GDN_CT tokens with cumulative log gates G_t, the recurrence
//   S_t = g_t S_{t-1} + k_t d_t^T,  d_t = beta_t (v_t - g_t S_{t-1}^T k_t),  o_t = scale S_t^T q_t
// is rewritten in WY form with T = (I + L)^-1, L_tj = beta_t e^(G_t-G_j) k_t.k_j (j < t):
//   W = T diag(beta e^G) K,  U = T diag(beta) V,  P_tj = e^(G_t-G_j) q_t.k_j (j <= t)
//   D = U - W S_0,  O = scale (diag(e^G) Q S_0 + P D),  S_L = e^(G_L) S_0 + K^T diag(e^(G_L-G)) D
// The prep kernel computes W, U, P and the decays for all chunks in parallel, the scan kernel walks the chunks.
// Same math as the sequential kernel, but a different floating-point evaluation order.
#define GDN_CT 32
// per (head, chunk) scratch: W [CT][S_v], U [CT][S_v], P [CT][CT], e^G [CT], e^(G_L - G) [CT], e^G_L, padding
static constexpr __host__ __device__ int64_t gdn_chunk_stride_w(int S_v) { return 2*GDN_CT*S_v + GDN_CT*GDN_CT + 2*GDN_CT + 4; }

template <int S_v>
__global__ void __launch_bounds__(256) gated_delta_net_chunk_prep(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        float * scratch, int64_t n_tokens, int64_t sq1, int64_t sq2, int64_t sv1, int64_t sv2, int64_t sb1, int64_t sb2,
        const uint3 neqk1_magic) {
    constexpr int CT  = GDN_CT;
    constexpr int NPF = CT*S_v/(4*256);
    static_assert(S_v == 128, "chunked GDN expects S_v == 128");
    const int h   = blockIdx.x;
    const int c   = blockIdx.y;
    const int tid = threadIdx.x;
    const int t0  = c*CT;
    const int Lc  = min(CT, (int) (n_tokens - t0));
    const uint32_t iq1 = fastmodulo(h, neqk1_magic);

    __shared__ __align__(16) float Ks[CT][S_v + 4];
    __shared__ __align__(16) float QVs[CT][S_v + 4]; // Q, later V
    __shared__ float As[CT][CT + 1];
    __shared__ float Ts[CT][CT + 1];
    __shared__ float Gs[CT];
    __shared__ float Bs[CT];

    {
        float4 kpf[NPF];
        float4 qpf[NPF];
#pragma unroll
        for (int r = 0; r < NPF; r++) {
            const int idx4 = tid + 256*r;
            const int t    = idx4 / (S_v/4);
            const int64_t off = (t0 + t)*sq2 + iq1*sq1 + 4*(idx4 % (S_v/4));
            kpf[r] = t < Lc ? *(const float4 *) (k + off) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            qpf[r] = t < Lc ? *(const float4 *) (q + off) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        if (tid < CT) {
            const bool ok = tid < Lc;
            Gs[tid] = ok ? g   [(t0 + tid)*sb2 + h*sb1] : 0.0f;
            Bs[tid] = ok ? beta[(t0 + tid)*sb2 + h*sb1] : 0.0f;
        }
#pragma unroll
        for (int r = 0; r < NPF; r++) {
            const int idx4 = tid + 256*r;
            const int t    = idx4 / (S_v/4);
            const int i    = 4*(idx4 % (S_v/4));
            *(float4 *) &Ks [t][i] = kpf[r];
            *(float4 *) &QVs[t][i] = qpf[r];
        }
    }
    __syncthreads();
    if (tid < 32) {
        // inclusive prefix sum of the log gates (padding tokens have G = 0)
        float x = Gs[tid];
#pragma unroll
        for (int o = 1; o < 32; o *= 2) {
            const float y = __shfl_up_sync(0xFFFFFFFF, x, o);
            if (tid >= o) {
                x += y;
            }
        }
        Gs[tid] = x;
    }
    __syncthreads();

    float * W  = scratch + (int64_t) (h*gridDim.y + c)*gdn_chunk_stride_w(S_v);
    float * U  = W + CT*S_v;
    float * P  = U + CT*S_v;
    float * Eg = P + CT*CT;
    float * Ed = Eg + CT;

    // L (into As) and P from K K^T and Q K^T; thread = 4 rows (t) x 2 columns (j), 128 threads per product
    {
        const bool is_q = tid >= 128;
        const int  tt   = tid % 128;
        const int  t    = (tt / 16) * 4;
        const int  j    = (tt % 16) * 2;
        const float (*X)[S_v + 4] = is_q ? QVs : Ks;
        float acc[4][2] = {{0.0f}};
#pragma unroll 4
        for (int i = 0; i < S_v; i += 4) {
            float4 xr[4], kr[2];
#pragma unroll
            for (int a = 0; a < 4; a++) {
                xr[a] = *(const float4 *) &X[t + a][i];
            }
#pragma unroll
            for (int b = 0; b < 2; b++) {
                kr[b] = *(const float4 *) &Ks[j + b][i];
            }
#pragma unroll
            for (int a = 0; a < 4; a++) {
#pragma unroll
                for (int b = 0; b < 2; b++) {
                    acc[a][b] += xr[a].x*kr[b].x + xr[a].y*kr[b].y + xr[a].z*kr[b].z + xr[a].w*kr[b].w;
                }
            }
        }
#pragma unroll
        for (int a = 0; a < 4; a++) {
#pragma unroll
            for (int b = 0; b < 2; b++) {
                const int ta = t + a;
                const int jb = j + b;
                const float decay = expf(Gs[ta] - Gs[jb]);
                if (is_q) {
                    P[ta*CT + jb] = jb <= ta ? decay * acc[a][b] : 0.0f;
                } else {
                    As[ta][jb] = jb < ta ? Bs[ta] * decay * acc[a][b] : 0.0f;
                }
            }
        }
    }
    if (tid < CT) {
        const float gl = Gs[Lc - 1];
        Eg[tid] = expf(Gs[tid]);
        Ed[tid] = expf(gl - Gs[tid]);
        if (tid == 0) {
            Ed[CT] = expf(gl);
        }
    }
    // prefetch V while solving for T
    float4 vpf[NPF];
#pragma unroll
    for (int r = 0; r < NPF; r++) {
        const int idx4 = tid + 256*r;
        const int t    = idx4 / (S_v/4);
        vpf[r] = t < Lc ? *(const float4 *) (v + (t0 + t)*sv2 + h*sv1 + 4*(idx4 % (S_v/4))) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    // T = (I + L)^-1 by forward substitution: 8 lanes per column, lane l keeps x_j for j = l mod 8; Ts = T diag(beta)
    {
        const int col = tid / 8;
        const int l   = tid % 8;
        float x[CT/8];
#pragma unroll
        for (int t = 0; t < CT; t++) {
            float part = 0.0f;
#pragma unroll
            for (int jj = 0; jj < CT/8; jj++) {
                const int j = 8*jj + l;
                if (j < t) {
                    part += As[t][j] * x[jj];
                }
            }
            part += __shfl_xor_sync(0xFFFFFFFF, part, 4, 8);
            part += __shfl_xor_sync(0xFFFFFFFF, part, 2, 8);
            part += __shfl_xor_sync(0xFFFFFFFF, part, 1, 8);
            if (l == t % 8) {
                x[t / 8] = t < col ? 0.0f : (t == col ? 1.0f : 0.0f) - part;
            }
        }
        const float bc = Bs[col];
#pragma unroll
        for (int jj = 0; jj < CT/8; jj++) {
            Ts[8*jj + l][col] = x[jj] * bc;
        }
    }
#pragma unroll
    for (int r = 0; r < NPF; r++) {
        const int idx4 = tid + 256*r;
        *(float4 *) &QVs[idx4 / (S_v/4)][4*(idx4 % (S_v/4))] = vpf[r];
    }
    __syncthreads();

    // W = Ts diag(e^G) K, U = Ts V; thread = column i, 16 rows t
    {
        const int i  = tid % S_v;
        const int tb = (tid / S_v) * (CT/2);
        float kcol[CT];
        float vcol[CT];
#pragma unroll
        for (int j = 0; j < CT; j++) {
            kcol[j] = expf(Gs[j]) * Ks[j][i];
            vcol[j] = QVs[j][i];
        }
#pragma unroll
        for (int tt = 0; tt < CT/2; tt++) {
            const int t = tb + tt;
            float w = 0.0f;
            float u = 0.0f;
#pragma unroll
            for (int j = 0; j < CT; j++) {
                const float tj = Ts[t][j]; // zero above the diagonal
                w += tj * kcol[j];
                u += tj * vcol[j];
            }
            W[t*S_v + i] = w;
            U[t*S_v + i] = u;
        }
    }
}

// one block of 64 threads per (head, 16 state columns); register-blocked: each thread computes 2 tokens x 4 columns
// of D and O and 8 rows x 4 columns of the state
#define GDN_SCAN_VS 16
#define GDN_SCAN_NT 64
template <int S_v>
__global__ void __launch_bounds__(GDN_SCAN_NT) gated_delta_net_chunk_scan(
        const float * q, const float * k, const float * scratch, const float * curr_state,
        float * dst, float * state_out, int64_t H, int64_t n_tokens, int64_t sq1, int64_t sq2,
        const uint3 neqk1_magic, float scale) {
    constexpr int CT  = GDN_CT;
    constexpr int VS  = GDN_SCAN_VS;
    constexpr int NT  = GDN_SCAN_NT;
    constexpr int NPF = CT*S_v/(4*NT); // float4 per thread for one [CT, S_v] tile
    static_assert(VS == 16 && NT == 64 && CT == 32 && S_v == 128, "thread layout below assumes these sizes");

    const int h   = blockIdx.x;
    const int c0  = blockIdx.y*VS;
    const int tid = threadIdx.x;
    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const int nchunks = (int) ((n_tokens + CT - 1) / CT);

    __shared__ __align__(16) float Ss[S_v][VS];
    __shared__ __align__(16) float Bufs[CT][S_v + 4];
    __shared__ __align__(16) float Ds[CT][VS];
    __shared__ float Ps[CT][CT + 1];
    __shared__ float Es[2*CT + 4]; // e^G_t [CT], e^(G_L - G_t) [CT], e^G_L
    const float * Eg = Es;
    const float * Ed = Es + CT;

    curr_state += (int64_t) h*S_v*S_v;
    state_out  += (int64_t) h*S_v*S_v;
    for (int idx = tid; idx < S_v*VS; idx += NT) {
        const int c = idx / S_v;
        const int i = idx % S_v;
        Ss[i][c] = curr_state[(c0 + c)*S_v + i];
    }

    const int tc = (tid % 4) * 4;  // first column (D, O, state)
    const int tr = (tid / 4) * 2;  // first token row (D, O)
    const int sr = (tid / 4) * 8;  // first state row

    const float * qk_base[2] = { q + iq1*sq1, k + iq1*sq1 };
    float4 pf[NPF];
    auto load_qk = [&](const int which, const int t0, const int Lc) {
#pragma unroll
        for (int r = 0; r < NPF; r++) {
            const int idx4 = tid + NT*r;
            const int t    = idx4 / (S_v/4);
            pf[r] = t < Lc ? *(const float4 *) (qk_base[which] + (t0 + t)*sq2 + 4*(idx4 % (S_v/4))) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto load_w = [&](const float * W) {
#pragma unroll
        for (int r = 0; r < NPF; r++) {
            pf[r] = ((const float4 *) W)[tid + NT*r];
        }
    };
    auto store_buf = [&]() {
#pragma unroll
        for (int r = 0; r < NPF; r++) {
            const int idx4 = tid + NT*r;
            *(float4 *) &Bufs[idx4 / (S_v/4)][4*(idx4 % (S_v/4))] = pf[r];
        }
    };
    // acc[r][c] = sum_i Bufs[tr + r][i] * Ss[i][tc + c]
    auto buf_times_state = [&](float acc[2][4]) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                acc[r][c] = 0.0f;
            }
        }
#pragma unroll 4
        for (int i = 0; i < S_v; i += 4) {
            const float4 b0 = *(const float4 *) &Bufs[tr + 0][i];
            const float4 b1 = *(const float4 *) &Bufs[tr + 1][i];
            const float bb[2][4] = { {b0.x, b0.y, b0.z, b0.w}, {b1.x, b1.y, b1.z, b1.w} };
#pragma unroll
            for (int ii = 0; ii < 4; ii++) {
                const float4 sv = *(const float4 *) &Ss[i + ii][tc];
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    acc[r][0] += bb[r][ii] * sv.x;
                    acc[r][1] += bb[r][ii] * sv.y;
                    acc[r][2] += bb[r][ii] * sv.z;
                    acc[r][3] += bb[r][ii] * sv.w;
                }
            }
        }
    };

    // per-chunk small inputs (P, e^G, e^(G_L-G), e^G_L, U rows), prefetched together with W
    constexpr int NPP = CT*CT/(4*NT);
    float4 ppf[NPP];
    float4 u[2];
    float  epf[2];
    auto load_small = [&](const float * W) {
        const float * U = W + CT*S_v;
        const float * P = U + CT*S_v;
        const float * E = P + CT*CT;
#pragma unroll
        for (int r = 0; r < NPP; r++) {
            ppf[r] = ((const float4 *) P)[tid + NT*r];
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            u[r]   = *(const float4 *) &U[(tr + r)*S_v + c0 + tc];
            epf[r] = tid + NT*r < 2*CT + 1 ? E[tid + NT*r] : 0.0f;
        }
    };

    load_w    (scratch + (int64_t) (h*nchunks)*gdn_chunk_stride_w(S_v));
    load_small(scratch + (int64_t) (h*nchunks)*gdn_chunk_stride_w(S_v));

    for (int c = 0; c < nchunks; c++) {
        const int t0 = c*CT;
        const int Lc = min(CT, (int) (n_tokens - t0));
        const float * W = scratch + (int64_t) (h*nchunks + c)*gdn_chunk_stride_w(S_v);

        store_buf();
#pragma unroll
        for (int r = 0; r < NPP; r++) {
            const int idx = 4*(tid + NT*r);
            Ps[idx / CT][idx % CT + 0] = ppf[r].x;
            Ps[idx / CT][idx % CT + 1] = ppf[r].y;
            Ps[idx / CT][idx % CT + 2] = ppf[r].z;
            Ps[idx / CT][idx % CT + 3] = ppf[r].w;
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            if (tid + NT*r < 2*CT + 1) {
                Es[tid + NT*r] = epf[r];
            }
        }
        const float4 uc[2] = { u[0], u[1] };
        __syncthreads();

        // D = U - W S_0
        load_qk(0, t0, Lc);
        {
            float acc[2][4];
            buf_times_state(acc);
#pragma unroll
            for (int r = 0; r < 2; r++) {
                *(float4 *) &Ds[tr + r][tc] = make_float4(uc[r].x - acc[r][0], uc[r].y - acc[r][1], uc[r].z - acc[r][2], uc[r].w - acc[r][3]);
            }
        }
        __syncthreads();
        store_buf();
        __syncthreads();

        // O = scale (e^G Q S_0 + P D)
        load_qk(1, t0, Lc);
        {
            float acc[2][4];
            buf_times_state(acc);
#pragma unroll
            for (int r = 0; r < 2; r++) {
                const int t = tr + r;
                float accp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int j = 0; j <= t; j++) {
                    const float  pv = Ps[t][j];
                    const float4 dv = *(const float4 *) &Ds[j][tc];
                    accp[0] += pv * dv.x;
                    accp[1] += pv * dv.y;
                    accp[2] += pv * dv.z;
                    accp[3] += pv * dv.w;
                }
                if (t < Lc) {
                    const float eg = Eg[t];
                    float * o = dst + (int64_t) (t0 + t)*S_v*H + h*S_v + c0 + tc;
                    *(float4 *) o = make_float4(scale * (eg * acc[r][0] + accp[0]), scale * (eg * acc[r][1] + accp[1]),
                                                scale * (eg * acc[r][2] + accp[2]), scale * (eg * acc[r][3] + accp[3]));
                }
            }
        }
        __syncthreads();
        store_buf();
        __syncthreads();

        // S_L = e^(G_L) S_0 + K^T diag(e^(G_L - G)) D
        if (c + 1 < nchunks) {
            load_w    (W + gdn_chunk_stride_w(S_v));
            load_small(W + gdn_chunk_stride_w(S_v));
        }
        {
            float acc[8][4];
#pragma unroll
            for (int rr = 0; rr < 8; rr++) {
                const float4 sv = *(const float4 *) &Ss[sr + rr][tc];
                acc[rr][0] = Es[2*CT] * sv.x;
                acc[rr][1] = Es[2*CT] * sv.y;
                acc[rr][2] = Es[2*CT] * sv.z;
                acc[rr][3] = Es[2*CT] * sv.w;
            }
            for (int t = 0; t < Lc; t++) {
                const float  ed = Ed[t];
                const float4 dv = *(const float4 *) &Ds[t][tc];
                const float4 k0 = *(const float4 *) &Bufs[t][sr + 0];
                const float4 k1 = *(const float4 *) &Bufs[t][sr + 4];
                const float kk[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
                const float d[4]  = {ed*dv.x, ed*dv.y, ed*dv.z, ed*dv.w};
#pragma unroll
                for (int rr = 0; rr < 8; rr++) {
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) {
                        acc[rr][cc] += kk[rr] * d[cc];
                    }
                }
            }
#pragma unroll
            for (int rr = 0; rr < 8; rr++) {
                *(float4 *) &Ss[sr + rr][tc] = make_float4(acc[rr][0], acc[rr][1], acc[rr][2], acc[rr][3]);
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < S_v*VS; idx += NT) {
        const int c = idx / S_v;
        const int i = idx % S_v;
        state_out[(c0 + c)*S_v + i] = Ss[i][c];
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, bool ends_only, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, ends_only);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, ends_only);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, ends_only);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, ends_only);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;
    const bool ends_only = ggml_get_op_params_i32(dst, 1) != 0;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // [TAG_GDN_CHUNKED] prompt batches of one sequence: chunked kernels for all but the last K-1 tokens, whose
    // snapshots (slots K-2..0) the sequential kernel then writes from the chunked state in slot K-1
    static const bool chunked_enabled = [] {
        const char * e = getenv("GGML_CUDA_GDN_CHUNKED");
        return e == nullptr || atoi(e) != 0;
    }();
    const int64_t n_tail    = keep_rs ? K - 1 : 0;
    const int64_t n_chunked = n_tokens - n_tail;
    if (chunked_enabled && !kda && S_v == 128 && n_seqs == 1 && neq3 == 1 && n_chunked >= 64 &&
            (uintptr_t) q_d % 16 == 0 && (uintptr_t) k_d % 16 == 0 && (uintptr_t) v_d % 16 == 0 &&
            sq1 % 4 == 0 && sq2 % 4 == 0 && sv1 % 4 == 0 && sv2 % 4 == 0) {
        const int64_t nchunks = (n_chunked + GDN_CT - 1) / GDN_CT;
        const int64_t nchunks_tail = (n_tail + GDN_CT - 1) / GDN_CT;
        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), std::max(nchunks, nchunks_tail)*H*gdn_chunk_stride_w(128));
        const uint3 neqk1_magic = init_fastdiv_values(neqk1);

        gated_delta_net_chunk_prep<128><<<dim3(H, nchunks), 256, 0, stream>>>(
            q_d, k_d, v_d, g_d, b_d, scratch.get(), n_chunked, sq1, sq2, sv1, sv2, sb1, sb2, neqk1_magic);

        float * state_chunked = state_d + n_tail*state_slot_stride;
        gated_delta_net_chunk_scan<128><<<dim3(H, 128/GDN_SCAN_VS), GDN_SCAN_NT, 0, stream>>>(
            q_d, k_d, scratch.get(), s_d, dst_d, state_chunked, H, n_chunked, sq1, sq2, neqk1_magic, scale);

        if (n_tail > 0 && ends_only) {
            // no snapshots between slot K-1 and slot 0: the tail is a second chunked pass from the slot K-1 state
            gated_delta_net_chunk_prep<128><<<dim3(H, nchunks_tail), 256, 0, stream>>>(
                q_d + n_chunked*sq2, k_d + n_chunked*sq2, v_d + n_chunked*sv2, g_d + n_chunked*sb2, b_d + n_chunked*sb2,
                scratch.get(), n_tail, sq1, sq2, sv1, sv2, sb1, sb2, neqk1_magic);
            gated_delta_net_chunk_scan<128><<<dim3(H, 128/GDN_SCAN_VS), GDN_SCAN_NT, 0, stream>>>(
                q_d + n_chunked*sq2, k_d + n_chunked*sq2, scratch.get(), state_chunked, dst_d + n_chunked*S_v*H, state_d,
                H, n_tail, sq1, sq2, neqk1_magic, scale);
        } else if (n_tail > 0) {
            launch_gated_delta_net<false, true>(q_d + n_chunked*sq2, k_d + n_chunked*sq2, v_d + n_chunked*sv2,
                g_d + n_chunked*sb2, b_d + n_chunked*sb2, state_chunked, dst_d + n_chunked*S_v*H, state_d,
                S_v, H, n_tail, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ends_only, stream);
        }
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ends_only, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ends_only, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ends_only, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ends_only, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
