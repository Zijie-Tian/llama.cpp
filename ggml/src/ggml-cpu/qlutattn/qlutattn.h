#pragma once

#include <arm_neon.h>

#include "ggml-backend.h"  // 建议恢复，如果里面定义 ggml_fp16_t

#ifdef __cplusplus
extern "C" {
#endif

void ggml_vec_dot_qlutattn_kv1_128x128(int n, ggml_fp16_t * s, size_t bs, const uint8_t * GGML_RESTRICT x, size_t bx,
                                       const ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc);

void ggml_vec_dot_qlutattn_kv2_128x128(int n, ggml_fp16_t * s, size_t bs, const uint8_t * GGML_RESTRICT x, size_t bx,
                                       const ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc);

void ggml_vec_dot_qlutattn_kv4_128x128(int n, ggml_fp16_t * s, size_t bs, const uint8_t * GGML_RESTRICT x, size_t bx,
                                       const ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc);

// Function moved to qlutattn-config.h for global access
// struct qlutattn_kernel_config * find_qlutattn_128x128_kernel_config(int M, int K, int bits);
#ifdef __cplusplus
}
#endif
