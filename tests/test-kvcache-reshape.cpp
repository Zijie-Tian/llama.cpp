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
// static std::mt19937 g_rng(std::random_device{}());
static std::mt19937 g_rng(42);

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

static void print_tensor_info(const char * name, ggml_tensor * tensor) {
    printf("%s: [%ld, %ld, %ld, %ld] type=%s, elements=%ld\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2],
           tensor->ne[3], ggml_type_name(tensor->type), ggml_nelements(tensor));
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

    const int head_dim   = 128;
    const int seq_len    = 1024;
    const int n_heads    = 16;
    const int n_kv_heads = 16;
    const int kv_len     = 1024;

    // Create tensors for flash attention
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads,   1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);

    // Fill with FIXED reproducible data
    printf("\nGenerating fixed test data (seed=42)...\n");
    fill_tensor_f32(q, -0.8f, 0.8f);
    fill_tensor_f16(k, -0.6f, 0.6f);
    fill_tensor_f16(v, -0.7f, 0.7f);

    print_tensor_info("Q", q);
    print_tensor_info("K", k);
    print_tensor_info("V", v);

    ggml_tensor * k_quant = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, head_dim, kv_len, n_kv_heads, 1);    

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * cpy_dst = ggml_cpy(ctx, k, k_quant);
    ggml_build_forward_expand(gf, cpy_dst);

    struct ggml_tensor * result = compute_graph(ctx, gf, 12);
    
    print_tensor_info("K_quant", k_quant);

    // Clean up
    ggml_free(ctx);
    
    return 0;
}