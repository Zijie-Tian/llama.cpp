#include <cstdint>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <random>
#include <stdlib.h>
#include <float.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "tmac.h"
#include "lut_mul_mat.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
typedef float16_t tmac_float_type;
#else
#include <stdbool.h>
#include <stdint.h>
typedef float tmac_float_type;
#endif

//> ===================================================================================================
//> Copy from llama.cpp
//> ===================================================================================================

typedef uint16_t ggml_half;

#define GGML_FP16_TO_FP32(x) ggml_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_fp32_to_fp16(x)

#define CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))

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
//> Added QLUTATTN block
#define QKLUTATTN_W1G64 64
typedef struct {
    ggml_half d;                // scale
    ggml_half m;                // min
    uint8_t   qs[QKLUTATTN_W1G64 / 8];    // 8-bit quants
} block_qlutattn_w1g64;
static_assert(sizeof(block_qlutattn_w1g64) == sizeof(ggml_half) + sizeof(ggml_half) + QKLUTATTN_W1G64 / 8, "wrong qlutattn_w1g64 block size/padding");

#define QKLUTATTN_W2G64 64
typedef struct {
    ggml_half d;                // scale
    ggml_half m;                // min
    uint8_t   qs[QKLUTATTN_W2G64 / 4];    // 8-bit quants
} block_qlutattn_w2g64;
static_assert(sizeof(block_qlutattn_w2g64) == sizeof(ggml_half) + sizeof(ggml_half) + QKLUTATTN_W2G64 / 4, "wrong qlutattn_w2g64 block size/padding");

#define QKLUTATTN_W4G64 64
typedef struct {
    ggml_half d;                // scale
    ggml_half m;                // min
    uint8_t   qs[QKLUTATTN_W4G64 / 2];    // 8-bit quants
} block_qlutattn_w4g64;
static_assert(sizeof(block_qlutattn_w4g64) == sizeof(ggml_half) + sizeof(ggml_half) + QKLUTATTN_W4G64 / 2, "wrong qlutattn_w4g64 block size/padding");

//> ===================================================================================================
//> Above are copy from llama.cpp.
//> ===================================================================================================

// Use fixed seed for reproducible results
static std::mt19937 g_rng(42);
// static std::mt19937 g_rng(std::random_device{}());

static void fill_tensor_f32(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    float *                         data       = (float *) dst->data;
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
    float * data = (float *) dst->data;
    size_t  n_elements = ggml_nelements(dst);
    for (size_t i = 0; i < n_elements; i++) {
        data[i] = value;
    }
}

static void set_tensor_f16(ggml_tensor * dst, float value) {
    ggml_fp16_t * data = (ggml_fp16_t *) dst->data;
    size_t  n_elements = ggml_nelements(dst);
    for (size_t i = 0; i < n_elements; i++) {
        data[i] = ggml_fp32_to_fp16(value);
    }
}


struct BlockI2TypeAccessor {
    static constexpr int BITS = 2;
    static constexpr int n_elem = 8 / BITS;

    static uint8_t get_q(const void * data, int idx) {
        const uint8_t * qs = (const uint8_t *) data;
        int elem_idx = idx % n_elem;
        return qs[idx / n_elem] >> ((n_elem - 1 - elem_idx) * BITS);
    }

    static tmac_float_type get_scale(const void * data, int idx, int group_size) {
        const float * ss = (const float *) data;
        float s = ss[idx / group_size];
        return (tmac_float_type) s;
    }

    static tmac_float_type get_zero_point(const void * data, int idx, int group_size) {
        const float * zs = (const float *) data;
        float z = zs[idx / group_size];
        return (tmac_float_type) z;
    }
};

struct BlockI4TypeAccessor {
    static constexpr int BITS = 4;
    static constexpr int n_elem = 8 / BITS;

    static uint8_t get_q(const void * data, int idx) {
        const uint8_t * qs = (const uint8_t *) data;
        int elem_idx = idx % n_elem;
        return qs[idx / n_elem] >> ((n_elem - 1 - elem_idx) * BITS);
    }

    static tmac_float_type get_scale(const void * data, int idx, int group_size) {
        const float * ss = (const float *) data;
        float s = ss[idx / group_size];
        return (tmac_float_type) s;
    }

    static tmac_float_type get_zero_point(const void * data, int idx, int group_size) {
        const float * zs = (const float *) data;
        float z = zs[idx / group_size];
        return (tmac_float_type) z;
    }
};

static void pseudo_symmetric_quantize_f32(
    const float* input,
    int8_t* quantized,
    float* scales,
    float* zeros,
    int n,
    int n_bit,
    int q_group_size
) {
    int num_groups;
    if (q_group_size > 0) {
        if (n % q_group_size != 0) {
            GGML_ASSERT(0);
        }
        num_groups = n / q_group_size;
    } else if (q_group_size == -1) {
        num_groups = 1;
        q_group_size = n;
    } else {
        num_groups = 1;
        q_group_size = n;
    }

    //> [-2^(n_bit - 1) + 1, 2^(n_bit - 1) - 1]
    const int max_int = (1 << (n_bit - 1)) - 1;
    const int min_int = -max_int;

    for (int g = 0; g < num_groups; ++g) {
        int start_idx = g * q_group_size;
        int end_idx = start_idx + q_group_size;

        float max_abs_val = -FLT_MAX;

        for (int i = start_idx; i < end_idx; ++i) {
            float abs_val = fabsf(input[i]);
            if (abs_val > max_abs_val) max_abs_val = abs_val;
        }

        scales[g] = max_abs_val / max_int;
        zeros[g]  = 0.0f;    // NOTE : zero point is 0 for symmetric quantization.

        for (int i = start_idx; i < end_idx; ++i) {
            int quantized_val = (int)roundf(input[i] / scales[g]);
            quantized_val = quantized_val < min_int ? min_int : (quantized_val > max_int ? max_int : quantized_val);
            quantized[i] = (int8_t)quantized_val;
        }
    }
}

void quantize_row_qlutattn_w4g64_pg_ref(block_qlutattn_w4g64 * GGML_RESTRICT y, const float * GGML_RESTRICT x, int64_t k) {
    const int qk = QKLUTATTN_W4G64 / 2;
    const int nelem_per_byte = 64 / qk;
    assert(k % QKLUTATTN_W4G64 == 0);
    const int nb = k / 64;

    float scale[nb];
    float zero[nb];
    int8_t quantized[nb * 64];    //> pesudo quantize results.
    //> Per-channel quantization. for head_dim = 64.
    pseudo_symmetric_quantize_f32(x, quantized, scale, zero, k, 4, 64);

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < qk; j++) {
            //> [-2^(n_bit - 1) + 1, 2^(n_bit - 1) - 1] -> [1, 2^n_bit - 1]
            const uint8_t x0 = (uint8_t) (quantized[i * 64 + j * nelem_per_byte + 0] + (1 << (4 - 1)));
            const uint8_t x1 = (uint8_t) (quantized[i * 64 + j * nelem_per_byte + 1] + (1 << (4 - 1)));

            y[i].qs[j] = (x0 << 4) | (x1 << 0);
        }

        y[i].d = GGML_FP32_TO_FP16(scale[i]);
        y[i].m = GGML_FP32_TO_FP16(zero[i]);
    }
}

void dequantize_row_qlutattn_w4g64_pg_ref(float * GGML_RESTRICT y, const block_qlutattn_w4g64 * GGML_RESTRICT x, int64_t k) {
    const int qk = QKLUTATTN_W4G64 / 2;
    const int nelem_per_byte = 64 / qk;
    assert(k % QKLUTATTN_W4G64 == 0);
    const int nb = k / 64;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < qk; j++) {
            const uint8_t x0 = (uint8_t) (x[i].qs[j] >> 4);
            const uint8_t x1 = (uint8_t) (x[i].qs[j] & 0x0F);

            y[i * 64 + j * nelem_per_byte + 0] = (float) (x0 - (1 << (4 - 1)));
            y[i * 64 + j * nelem_per_byte + 1] = (float) (x1 - (1 << (4 - 1)));
        }

        float scale = GGML_FP16_TO_FP32(x[i].d);
        float zero  = GGML_FP16_TO_FP32(x[i].m);

        for (int j = 0; j < 64; j++) {
            y[i * 64 + j] = (float) (y[i * 64 + j] - zero) * scale;
        }
    }
}


static struct ggml_tensor * compute_graph(
    ggml_context* ctx,
    struct ggml_cgraph * gf,
    int n_threads = 1
) {
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    return ggml_graph_node(gf, -1);
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                printf("                                      ..., \n");
                i2 = ne[2] - n;
            }
            printf("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    printf("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                printf("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
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
                    if (i0 < ne[0] - 1) printf(", ");
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
    // Initialize T-MAC
    ggml_tmac_init();
    printf("✓ T-MAC initialized successfully\n");

    // Get T-MAC buffer type
    ggml_backend_buffer_type_t tmac_buft = ggml_backend_tmac_buffer_type();
    if (!tmac_buft) {
        printf("ERROR: Failed to get T-MAC buffer type\n");
        return 1;
    }

    // Allocate the buffer from tmac BUFT.
    const size_t test_size = 1024; // 1KB test
    ggml_backend_buffer_t test_buf = ggml_backend_buft_alloc_buffer(tmac_buft, test_size);
    if (!test_buf) {
        printf("ERROR: Failed to allocate T-MAC buffer\n");
        return 1;
    }

    //> ===================================================================================================
    //> Create Contexts
    //> ===================================================================================================

    struct ggml_init_params main_params = {
        .mem_size   = 1024 * 1024 * 1024, // 1GB
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * main_ctx = ggml_init(main_params);
    if (!main_ctx) {
        fprintf(stderr, "Failed to initialize GGML context\n");
        return 1;
    }

    ggml_init_params params = {
        .mem_size   = 1024 * 1024 * 1024,  // 1GB
        .mem_buffer = NULL,
        .no_alloc   = true,
    };

    ggml_context* tmac_ctx = ggml_init(params);
    if (!tmac_ctx) {
        printf("ERROR: Failed to create GGML context\n");
        ggml_backend_buffer_free(test_buf);
        return 1;
    }

    //> ===================================================================================================
    //> Create Tensors in Main Context and do Quant.
    //> ===================================================================================================

    // Create a simple 2D tensor
    const int M = 256, K = 256, N = 1;
    const int nbits = 2;

    ggml_tensor* tensor_f32 = ggml_new_tensor_2d(main_ctx, GGML_TYPE_F32, K, M);
    fill_tensor_f32(tensor_f32);

    //> Activation must be F32 for llama.cpp ggml op.
    ggml_tensor* activation = ggml_new_tensor_2d(main_ctx, GGML_TYPE_F32, K, N);
    // fill_tensor_f32(activation);
    set_tensor_f32(activation, 1.0f);

    //> Activation must be F16 for T-MAC op.
    ggml_tensor* activation_f16 = ggml_new_tensor_2d(main_ctx, GGML_TYPE_F16, K, N);

    ggml_tensor* tensor_q4_0 = ggml_new_tensor_2d(main_ctx, GGML_TYPE_Q4_0, K, M);

    //> Quantize Activation to F16
    struct ggml_cgraph * gf_f16 = ggml_new_graph(main_ctx);
    ggml_tensor* quant_op_f16 = ggml_cpy(main_ctx, activation, activation_f16);
    ggml_build_forward_expand(gf_f16, quant_op_f16);

    ggml_tensor * quantized_activation_f16 = compute_graph(main_ctx, gf_f16);

    //> Quantize Tensor to Q4_0
    struct ggml_cgraph * gf = ggml_new_graph(main_ctx);
    ggml_tensor* quant_op = ggml_cpy(main_ctx, tensor_f32, tensor_q4_0);
    ggml_build_forward_expand(gf, quant_op);

    ggml_tensor * quantized_tensor = compute_graph(main_ctx, gf);

    //> ===================================================================================================
    //> Pesudo Quantization of FP32 Tensor NOTE: Just compute the quantized NMSE.
    //> ===================================================================================================

    int64_t group_size = 64;
    float * dequantized_tensor = (float *) aligned_malloc(M * K * sizeof(float));

    block_qlutattn_w4g64 * qweights_block = (block_qlutattn_w4g64 *) aligned_malloc(M * K / group_size * sizeof(block_qlutattn_w4g64));
    float * tensor_f32_ptr = (float *) tensor_f32->data;

    quantize_row_qlutattn_w4g64_pg_ref(
        qweights_block,
        (float *) tensor_f32->data,
        M * K
    );

    dequantize_row_qlutattn_w4g64_pg_ref(
        dequantized_tensor,
        qweights_block,
        M * K
    );

    float nmse = 0.0f;
    float norm_sq = 0.0f;
    for (int i = 0; i < M * K; i++) {
        float diff = dequantized_tensor[i] - tensor_f32_ptr[i];
        nmse += diff * diff;
        norm_sq += tensor_f32_ptr[i] * tensor_f32_ptr[i];
    }
    nmse /= norm_sq;
    printf("NMSE = %.8f\n", nmse);

    //> ===================================================================================================
    //> Prepare the TMAC input.
    //> ===================================================================================================

    uint8_t * tmac_qs   = (uint8_t *) aligned_malloc(
        M * K / group_size * group_size / nbits * sizeof(uint8_t) +
        M * K / group_size * sizeof(float) * 2
    );

    for (int i = 0; i < M * K / group_size; i++) {
        for (int j = 0; j < group_size / nbits; j++) {
            tmac_qs[i * (group_size / nbits) + j] = qweights_block[i].qs[j];
            // tmac_qs[i * (QKLUTATTN_W4G64 / nbits) + j] = 1;
        }
    }

    float * scale_ptr = (float *) (tmac_qs + M * K / group_size * (group_size / nbits));
    float * zp_ptr    = (float *) (tmac_qs + M * K / group_size * (group_size / nbits) + M * K / group_size * sizeof(float));

    for (int i = 0; i < M * K / group_size; i++) {
        float scale = GGML_FP16_TO_FP32(qweights_block[i].d);   // NOTE:  Scaling factor.
        scale_ptr[i] = scale; // Scale
        // scale_ptr[i] = 1.0f; // Scale
    }

    for (int i = 0; i < M * K / group_size; i++) {
        float zero = GGML_FP16_TO_FP32(qweights_block[i].m);    // NOTE: Zero point.
        zp_ptr[i] = zero;
        // zp_ptr[i] = 0.0f;
    }

    //> ===================================================================================================
    //> Repack Quantized Tensor to T-MAC Context
    //> ===================================================================================================

    ggml_tensor* tensor = ggml_new_tensor_2d(tmac_ctx, GGML_TYPE_TMAC_W4G64_1, K, M);
    ggml_set_name(tensor, "tmac_tensor");

    // Allocate memory for T-MAC tensors
    ggml_backend_buffer_t tmac_buf = ggml_backend_alloc_ctx_tensors_from_buft(tmac_ctx, tmac_buft);
    if (!tmac_buf) {
        printf("ERROR: Failed to allocate T-MAC buffer\n");
        throw std::runtime_error("T-MAC buffer allocation failed");
    }
    if (!tensor) {
        printf("ERROR: Failed to create tensor\n");
        ggml_free(tmac_ctx);
        ggml_backend_buffer_free(test_buf);
        return 1;
    }

    ggml_backend_tmac_convert_weight(tensor, tmac_qs, 0, ggml_backend_buft_get_alloc_size(tmac_buft, tensor));

    struct ggml::cpu::tmac::tensor_traits * tensor_extra = (struct ggml::cpu::tmac::tensor_traits *) tensor->extra;
    tmac_tensor_extra * tmac_extra = tensor_extra -> get_tmac_tensor_extra(tensor->name);

    //> ===================================================================================================
    //> Do MUL_MAT compute.
    //> ===================================================================================================

    // NOTE: FP32 MUL_MAT
    struct ggml_cgraph * gf_mul_mat = ggml_new_graph(main_ctx);
    ggml_tensor* mul_op = ggml_mul_mat(main_ctx, tensor_f32, activation);
    ggml_build_forward_expand(gf_mul_mat, mul_op);
    ggml_tensor * result = compute_graph(main_ctx, gf_mul_mat);

    // NOTE: TMAC MUL_MAT
    struct ggml_cgraph * gf_mul_mat_tmac = ggml_new_graph(tmac_ctx);
    ggml_tensor* mul_op_tmac = ggml_mul_mat(tmac_ctx, tensor, activation);
    ggml_build_forward_expand(gf_mul_mat_tmac, mul_op_tmac);

    ggml_backend_buffer_t tmac_buf2 = ggml_backend_alloc_ctx_tensors_from_buft(tmac_ctx, tmac_buft);
    if (!tmac_buf2) {
        printf("ERROR: Failed to allocate T-MAC buffer\n");
        throw std::runtime_error("T-MAC buffer allocation failed");
    }

    ggml_tensor * result_tmac = compute_graph(tmac_ctx, gf_mul_mat_tmac);

    printf("Standard results :\n");
    ggml_print_tensor((uint8_t *)result->data, GGML_TYPE_F32, result->ne, result->nb, 4);
    printf("T-MAC results :\n");
    ggml_print_tensor((uint8_t *)result_tmac->data, GGML_TYPE_F32, result_tmac->ne, result_tmac->nb, 4);

    return 0;
}
