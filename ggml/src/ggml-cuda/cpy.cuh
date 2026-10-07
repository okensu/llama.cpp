#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_CPY_BATCH_MAX 16

// up to CUDA_CPY_BATCH_MAX independent f32 -> f32 copies in one kernel launch
void ggml_cuda_cpy_batch(ggml_backend_cuda_context & ctx, const ggml_tensor * const * srcs, const ggml_tensor * const * dsts, int n);
