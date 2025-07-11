#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <random>

// Use fixed seed for reproducible results
// static std::mt19937 g_rng(42);
static std::mt19937 g_rng(std::random_device{}());

static void fill_tensor_f16(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    ggml_fp16_t *                         data       = (ggml_fp16_t *) dst->data;
    size_t                                n_elements = ggml_nelements(dst);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = ggml_fp32_to_fp16(dis(g_rng));
    }
}

static void print_tensor_info(const char * name, ggml_tensor * tensor) {
    printf("%s: [%ld, %ld, %ld, %ld] type=%s, elements=%ld\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2],
           tensor->ne[3], ggml_type_name(tensor->type), ggml_nelements(tensor));
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

// Calculate NMSE (Normalized Mean Square Error)
static double calculate_nmse(ggml_tensor * original, ggml_tensor * quantized) {
    if (ggml_nelements(original) != ggml_nelements(quantized) || ggml_nelements(original) == 0) {
        return -1.0;
    }
    
    size_t n_elements = ggml_nelements(original);
    double mse = 0.0;
    double signal_power = 0.0;
    
    // Extract values from tensors
    for (size_t i = 0; i < n_elements; i++) {
        float orig_val = 0.0f;
        float quant_val = 0.0f;
        
        // Get original value
        if (original->type == GGML_TYPE_F32) {
            orig_val = ((float*)original->data)[i];
        } else if (original->type == GGML_TYPE_F16) {
            orig_val = ggml_fp16_to_fp32(((ggml_fp16_t*)original->data)[i]);
        } else {
            auto tt = ggml_get_type_traits(original->type);
            if (tt && tt->to_float) {
                tt->to_float(original->data, &orig_val, 1);
            }
        }
        
        // Get quantized value
        if (quantized->type == GGML_TYPE_F32) {
            quant_val = ((float*)quantized->data)[i];
        } else if (quantized->type == GGML_TYPE_F16) {
            quant_val = ggml_fp16_to_fp32(((ggml_fp16_t*)quantized->data)[i]);
        } else {
            auto tt = ggml_get_type_traits(quantized->type);
            if (tt && tt->to_float) {
                tt->to_float(quantized->data, &quant_val, 1);
            }
        }
        
        double diff = orig_val - quant_val;
        mse += diff * diff;
        signal_power += orig_val * orig_val;
    }
    
    mse /= n_elements;
    signal_power /= n_elements;
    
    if (signal_power == 0.0) {
        return -1.0;
    }
    
    return mse / signal_power;
}

int main() {
    struct ggml_init_params params = {
        .mem_size   = 1024*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize GGML context\n");
        return 1;
    }

    const int64_t head_dim   = 128;
    const int64_t kv_len     = 128;
    const int64_t n_kv_heads = 4;

    // Create source tensor (FP16)
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);
    ggml_set_name(k, "k_source");
    
    // Create quantized tensor with GGML_TYPE_QLUTATTN_W4G128_PC
    ggml_tensor * k_quantized = ggml_new_tensor_4d(ctx, GGML_TYPE_QLUTATTN_W4G128_PC, k->ne[0], k->ne[1], k->ne[2], k->ne[3]);
    ggml_set_name(k_quantized, "k_quantized");

    ggml_tensor * k_quantized_pt = ggml_new_tensor_4d(ctx, GGML_TYPE_QLUTATTN_W4G128_PT, k->ne[0], k->ne[1], k->ne[2], k->ne[3]);
    ggml_set_name(k_quantized_pt, "k_quantized_pt");

    // ggml_tensor * k_dequantized = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k->ne[0], k->ne[1], k->ne[2], k->ne[3]);
    // ggml_set_name(k_dequantized, "k_dequantized");

    // ggml_tensor * k_dequantized_pt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k->ne[0], k->ne[1], k->ne[2], k->ne[3]);
    // ggml_set_name(k_dequantized_pt, "k_dequantized_pt");

    // Fill source tensor with test data
    printf("Generating test data...\n");
    fill_tensor_f16(k, -0.6f, 0.6f);

    printf("\nTensor Information:\n");
    print_tensor_info("K (source)", k);
    print_tensor_info("K (quantized)", k_quantized);

    //> Build computation graph for quantization
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    
    //> Quantize: k -> k_quantized, k_quantized_pt
    ggml_tensor * quant_op = ggml_cpy(ctx, k, k_quantized);
    ggml_build_forward_expand(gf, quant_op);
    ggml_tensor * quant_op_pt = ggml_cpy(ctx, k, k_quantized_pt);
    ggml_build_forward_expand(gf, quant_op_pt);

    // //> Dequantize: k_quantized -> k_dequantized, k_dequantized_pt
    // ggml_tensor * dequant_op = ggml_cpy(ctx, k_quantized, k_dequantized);
    // ggml_build_forward_expand(gf, dequant_op);
    // ggml_tensor * dequant_op_pt = ggml_cpy(ctx, k_quantized_pt, k_dequantized_pt);
    // ggml_build_forward_expand(gf, dequant_op_pt);

    //> Do compute.
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    
    printf("K (source):\n");
    ggml_print_tensor((uint8_t *)k->data, GGML_TYPE_F16, k->ne, k->nb, 3);

    // Clean up
    ggml_free(ctx);
    
    return 0;
}