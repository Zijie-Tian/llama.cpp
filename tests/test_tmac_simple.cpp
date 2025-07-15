#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <random>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "tmac.h"
#include "lut_mul_mat.h"

// Use fixed seed for reproducible results
// static std::mt19937 g_rng(42);
static std::mt19937 g_rng(std::random_device{}());

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

static struct ggml_tensor * compute_graph(
    ggml_context* ctx,
    struct ggml_cgraph * gf,
    int n_threads = 12
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
        .mem_size   = 1024*1024*1024,
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
    const int M = 64, K = 256, N = 1;
    
    
    ggml_tensor* tensor_f32 = ggml_new_tensor_2d(main_ctx, GGML_TYPE_F32, K, M);
    fill_tensor_f32(tensor_f32);

    //> Activation must be F32 for llama.cpp ggml op.
    ggml_tensor* activation = ggml_new_tensor_2d(main_ctx, GGML_TYPE_F32, K, N);
    fill_tensor_f32(activation);

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
    //> Repack Quantized Tensor to T-MAC Context
    //> =================================================================================================== 

    ggml_tensor* tensor = ggml_new_tensor_2d(tmac_ctx, GGML_TYPE_TMAC_W4G64_0, K, M);
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

    ggml_backend_tmac_convert_weight(tensor, quantized_tensor->data, 0, ggml_backend_buft_get_alloc_size(tmac_buft, tensor));

    //> ===================================================================================================
    //> Do MUL_MAT compute.
    //> ===================================================================================================

    struct ggml_cgraph * gf_mul_mat = ggml_new_graph(main_ctx);
    ggml_tensor* mul_op = ggml_mul_mat(main_ctx, tensor_f32, activation);
    ggml_build_forward_expand(gf_mul_mat, mul_op);
    ggml_tensor * result = compute_graph(main_ctx, gf_mul_mat);

    // struct ggml_cgraph * gf_mul_mat_tmac = ggml_new_graph(tmac_ctx);
    // ggml_tensor* mul_op_tmac = ggml_mul_mat(tmac_ctx, tensor, quantized_activation_f16);
    // ggml_build_forward_expand(gf_mul_mat_tmac, mul_op_tmac);
    // ggml_tensor * result_tmac = compute_graph(tmac_ctx, gf_mul_mat_tmac);

    printf("Standard results :\n");
    ggml_print_tensor((uint8_t *)result->data, GGML_TYPE_F32, result->ne, result->nb, 3);
    // printf("T-MAC results :\n");
    // ggml_print_tensor((uint8_t *)result_tmac->data, GGML_TYPE_F32, result_tmac->ne, result_tmac->nb, 3);
    
    return 0;
} 