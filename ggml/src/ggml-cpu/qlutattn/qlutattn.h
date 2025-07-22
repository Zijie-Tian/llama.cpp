#pragma once

#include "ggml-backend.h"  // 建议恢复，如果里面定义 ggml_fp16_t

#ifdef __cplusplus
extern "C" {
#endif

void ggml_vec_dot_qlutattn_kv4_128x128(int n, float * s, size_t bs, const uint8_t * GGML_RESTRICT x, size_t bx,
                                       const ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc);
#ifdef __cplusplus
}
#endif
