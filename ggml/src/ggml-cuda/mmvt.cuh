#include "common.cuh"

// Tensor core matrix-vector product for Q5_K / Q4_K and 2..8 columns (int8 mma per 32-value sub-block).
bool ggml_cuda_should_use_mmvt(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_vec_t(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
