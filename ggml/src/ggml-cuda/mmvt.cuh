#include "common.cuh"

// Tensor core matrix-vector product for Q5_K / Q4_K / IQ4_XS and 2..8 columns (int8 mma per 32-value sub-block).
bool ggml_cuda_should_use_mmvt(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_vec_t(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// gate/up matmuls with the same src1 followed by swiglu(gate, up): one kernel, writes the glu result
bool ggml_cuda_should_use_mmvt_glu(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * glu, int cc);

void ggml_cuda_mul_mat_vec_t_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * glu);
