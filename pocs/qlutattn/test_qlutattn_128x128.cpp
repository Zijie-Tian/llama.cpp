#include <float.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <sys/types.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>

#include "qlut_ctor.h"
#include "qlutattn.h"
#include "tbl.h"

#ifdef __ARM_NEON
#    include <arm_neon.h>
#elif defined __AVX2__
#    include <immintrin.h>
#endif

#ifdef __ARM_NEON
typedef float16_t tmac_float_type;
#else
#    include <stdbool.h>
#    include <stdint.h>
typedef float tmac_float_type;
#endif

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

// Use fixed seed for reproducible results
static std::mt19937 g_rng(42);
// static std::mt19937 g_rng(std::random_device{}());

constexpr size_t kAllocAlignment = 64;

static void * aligned_malloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, kAllocAlignment);
#else
    void * ptr = nullptr;
    posix_memalign(&ptr, kAllocAlignment, size);
    return ptr;
#endif
}

static void aligned_free(void * ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static void fill_tensor_f32(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    float *                               data       = (float *) dst->data;
    size_t                                n_elements = ggml_nelements(dst);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = dis(g_rng);
    }
}

static void fill_tensor_f16(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    ggml_fp16_t *                         data       = (ggml_fp16_t *) dst->data;
    size_t                                n_elements = ggml_nelements(dst);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = ggml_fp32_to_fp16(dis(g_rng));
    }
}

static void set_tensor_f32(ggml_tensor * dst, float value) {
    float * data       = (float *) dst->data;
    size_t  n_elements = ggml_nelements(dst);
    for (size_t i = 0; i < n_elements; i++) {
        data[i] = value;
    }
}

static void set_tensor_f16(ggml_tensor * dst, float value) {
    ggml_fp16_t * data       = (ggml_fp16_t *) dst->data;
    size_t        n_elements = ggml_nelements(dst);
    for (size_t i = 0; i < n_elements; i++) {
        data[i] = ggml_fp32_to_fp16(value);
    }
}

//> Added QLUTATTN block quantization.
#define QKLUTATTN_KV1_128x128 (128 * 128)

typedef struct {
    ggml_half d[128];                         // scale
    ggml_half m[128];                         // min
    uint8_t   qs[QKLUTATTN_KV1_128x128 / 8];  // 8-bit quants
} block_qlutattn_kv1_128x128;

static_assert(sizeof(block_qlutattn_kv1_128x128) ==
                  (sizeof(ggml_half) + sizeof(ggml_half)) * 128 + QKLUTATTN_KV1_128x128 / 8,
              "wrong qlutattn_w1_128x128 block size/padding");

#define QKLUTATTN_KV2_128x128 (128 * 128)

typedef struct {
    ggml_half d[128];                         // scale
    ggml_half m[128];                         // min
    uint8_t   qs[QKLUTATTN_KV2_128x128 / 4];  // 8-bit quants
} block_qlutattn_kv2_128x128;

static_assert(sizeof(block_qlutattn_kv2_128x128) ==
                  (sizeof(ggml_half) + sizeof(ggml_half)) * 128 + QKLUTATTN_KV2_128x128 / 4,
              "wrong qlutattn_w2_128x128 block size/padding");

#define QKLUTATTN_KV4_128x128 (128 * 128)

typedef struct {
    uint8_t qs[QKLUTATTN_KV4_128x128 / 2 + 128 * sizeof(float) * 2];  // 2-bit quants
} block_qlutattn_kv4_128x128;

static_assert(sizeof(block_qlutattn_kv4_128x128) == (sizeof(float) + sizeof(float)) * 128 + QKLUTATTN_KV4_128x128 / 2,
              "wrong qlutattn_w4_128x128 block size/padding");

/**
 * @brief Pseudo symmetric quantization of a float array.
 *      NOTICE : This function is per-CHANNEL quantization.
 * @param input
 * @param quantized
 * @param scales
 * @param zeros
 * @param n
 * @param n_bit
 * @param q_group_size
 */
static void pseudo_symmetric_quantize_128x128_simd_f32(int8_t * quantized, const float * input, float * scales,
                                                       float * zeros, int n, int n_bit) {
    const int64_t group_size = 128;
    GGML_ASSERT(n % QKLUTATTN_KV4_128x128 == 0);

    const int64_t n_groups = n / group_size;

    float group_max_abs[n_groups];
    memset(group_max_abs, -std::numeric_limits<float>::infinity(), n_groups * sizeof(float));

    //> [-2^(n_bit - 1) + 1, 2^(n_bit - 1) - 1]
    const int max_int = (1 << (n_bit - 1)) - 1;
    const int min_int = -max_int;

    for (int g = 0; g < n_groups; ++g) {
        for (int i = g * group_size; i < (g + 1) * group_size; ++i) {
            float abs_val = fabsf(input[i]);
            if (std::isnan(abs_val)) {
                abs_val = 0.0f;  // Handle NaN values
            }

            if (abs_val > group_max_abs[i % group_size]) {
                group_max_abs[i % group_size] = abs_val;
            }
        }
    }

    int g_idx = 0;
    for (int idx = 0; idx < n; ++idx) {
        g_idx = idx % group_size;

        scales[g_idx] = group_max_abs[g_idx] / max_int;
        zeros[g_idx]  = 0.0f;

        quantized[idx] = (int8_t) roundf(input[idx] / scales[g_idx]);
        quantized[idx]<min_int ? quantized[idx] = min_int : quantized[idx]> max_int ? quantized[idx] = max_int :
                                                                                      quantized[idx];
    }
}

static void pseudo_symmetric_dequantize_128x128_simd_f32(float * dequantized, const int8_t * quantized,
                                                         const float * scales, const float * zeros, int n, int n_bit) {
    const int64_t group_size = 128;
    GGML_ASSERT(n % QKLUTATTN_KV4_128x128 == 0);

    for (int i = 0; i < n; ++i) {
        int g_idx      = i % group_size;
        dequantized[i] = quantized[i] * scales[g_idx] + zeros[g_idx];
    }
}

static struct ggml_tensor * compute_graph(ggml_context * ctx, struct ggml_cgraph * gf, int n_threads = 1) {
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    return ggml_graph_node(gf, -1);
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) {
                printf("                                      ..., \n");
                i2 = ne[2] - n;
            }
            printf("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) {
                    printf("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                printf("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float  v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    printf("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) {
                        printf(", ");
                    }
                }
                printf("],\n");
            }
            printf("                                      ],\n");
        }
        printf("                                     ]\n");
        printf("                                     sum = %f\n", sum);
    }
}

int main() {
    //> ===================================================================================================
    //> Allocate GGML context
    //> ===================================================================================================

    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize GGML context\n");
        return 1;
    }

    //> ===================================================================================================
    //> Init Tensors
    //> ===================================================================================================

    const int64_t head_dim   = 128;
    const int64_t kv_len     = 128 * 2;
    const int64_t n_kv_heads = 1;
    const int     nbits      = 4;  //> nbits >= 2

    ggml_tensor * activation = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, 1, 1, 1);
    ggml_set_name(activation, "activation");
    set_tensor_f16(activation, 1.0f);

    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim * kv_len, 1, 1, 1);
    ggml_set_name(k, "k_source");

    ggml_tensor * k_quantized = ggml_new_tensor_4d(ctx, GGML_TYPE_QLUTATTN_KV4_128x128, head_dim * kv_len, 1, 1, 1);
    ggml_set_name(k_quantized, "k_quantized");

    ggml_tensor * k_dequantized = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim * kv_len, 1, 1, 1);
    ggml_set_name(k_dequantized, "k_dequantized");

    fill_tensor_f16(k);
    // set_tensor_f16(k, 1.0f);

    ggml_print_tensor((uint8_t *) k->data, GGML_TYPE_F16, k->ne, k->nb, 3);

    int8_t * q_vals   = (int8_t *) aligned_malloc(head_dim * kv_len * sizeof(int8_t));
    float *  k_scales = (float *) aligned_malloc(head_dim * sizeof(float));
    float *  k_zeros  = (float *) aligned_malloc(head_dim * sizeof(float));

    pseudo_symmetric_quantize_128x128_simd_f32((int8_t *) q_vals, (float *) k->data, k_scales, k_zeros,
                                               head_dim * kv_len, nbits);

    // float * k_dequantized = (float * )aligned_malloc(head_dim * kv_len * sizeof(float));

    // pseudo_symmetric_dequantize_128x128_simd_f32(
    //     k_dequantized,
    //     (int8_t *)q_vals,
    //     k_scales,
    //     k_zeros,
    //     head_dim * kv_len,
    //     nbits
    // );

    //> ===================================================================================================
    //> Call quantization.
    //> ===================================================================================================

    // NOTE: Quantization operation
    ggml_tensor *        quant_op = ggml_cpy(ctx, k, k_quantized);  //> k -> k_quantized
    struct ggml_cgraph * gf       = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, quant_op);

    ggml_graph_compute_with_ctx(ctx, gf, 4);

    printf("Quantized results :\n");

    //> ===================================================================================================
    //> Reshape the k into 128x128
    //> ===================================================================================================

    ggml_tensor *        k_reshaped = ggml_reshape_4d(ctx, k, head_dim, kv_len, 1, 1);
    struct ggml_cgraph * gf_reshape = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_reshape, k_reshaped);
    ggml_graph_compute_with_ctx(ctx, gf_reshape, 4);

    //> ===================================================================================================
    //> Do MUL_MAT with quantized tensor.
    //> ===================================================================================================

    // NOTE: FP32 MUL_MAT
    struct ggml_cgraph * gf_mul = ggml_new_graph(ctx);
    ggml_tensor *        mul_op = ggml_mul_mat(ctx, k_reshaped, activation);  //> k_quantized * k_dequantized
    ggml_build_forward_expand(gf_mul, mul_op);
    ggml_graph_compute_with_ctx(ctx, gf_mul, 4);

    // NOTE: Mixed precision MUL_MAT
    struct qlutattn_kernel_config * kernel_config =
        find_qlutattn_128x128_kernel_config(1, 1, 4);  // NOTE: Just for test
    if (kernel_config == nullptr) {
        fprintf(stderr, "Failed to find qlutattn kernel config for %d x %d x %d\n", head_dim, head_dim, nbits);
        return 1;
    }
    float *   ret        = (float *) aligned_malloc(head_dim * sizeof(float));
    uint8_t * LUT_buffer = (uint8_t *) aligned_malloc(
        head_dim / 4 * 16 * sizeof(uint8_t) + head_dim / kernel_config->act_group_size * sizeof(tmac_float_type) * 2);

    int8_t *          qlut       = (int8_t *) LUT_buffer;
    tmac_float_type * lut_scales = (tmac_float_type *) ((uint8_t *) LUT_buffer + head_dim / 4 * 16 * sizeof(uint8_t));
    tmac_float_type * lut_biases =
        (tmac_float_type *) ((uint8_t *) LUT_buffer + head_dim / 4 * 16 * sizeof(uint8_t) +
                             head_dim / kernel_config->act_group_size * sizeof(tmac_float_type));

    // NOTE: Call QLUT build & quantization.
    ggml::cpu::qlutattn::qlutattn_lut_ctor_int8_g4(activation->data, lut_scales, lut_biases, qlut, head_dim,
                                                   kernel_config);

    // NOTE: Do lut_gemv
    ggml_vec_dot_t qlutattn_vec_dot = ggml_get_type_traits_cpu(k_quantized->type)->vec_dot;
    qlutattn_vec_dot(head_dim, ret, head_dim, (uint8_t *) k_quantized->data, head_dim, LUT_buffer, head_dim, head_dim);

    //> ===================================================================================================
    //> Print results
    //> ===================================================================================================

    printf("MUL_MAT results:\n");
    ggml_print_tensor((uint8_t *) mul_op->data, GGML_TYPE_F32, mul_op->ne, mul_op->nb, 8);

    printf("Quantized results:\n");
    size_t nb[4] = { sizeof(tmac_float_type), sizeof(tmac_float_type) * mul_op->ne[0], sizeof(int8_t) * mul_op->ne[1],
                     sizeof(int8_t) * mul_op->ne[2] };
    ggml_print_tensor((uint8_t *) ret, GGML_TYPE_F16, mul_op->ne, nb, 8);

    ggml_free(ctx);

    return 0;
}
