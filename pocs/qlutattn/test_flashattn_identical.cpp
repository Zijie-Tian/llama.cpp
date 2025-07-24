#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

// Configuration
struct test_config {
    int64_t batch_size = 1;
    int64_t n_heads    = 32;
    int64_t q_len      = 32;
    int64_t kv_len     = 128;  // Total KV length
    int64_t head_dim   = 128;
    bool    verbose    = false;
};

#define GGML_KQ_MASK_PAD 64
#define GGML_PAD(x, n)   (((x) + (n) - 1) & ~((n) - 1))

// Use fixed seed for reproducibility
static std::mt19937 g_rng(42);

// Initialize random tensor
static void fill_tensor_f16(ggml_tensor * tensor, float min_val = -1.0f, float max_val = 1.0f) {
    ggml_fp16_t *                         data       = (ggml_fp16_t *) tensor->data;
    size_t                                n_elements = ggml_nelements(tensor);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = ggml_fp32_to_fp16(dis(g_rng));
    }
}

// Initialize attention mask (causal mask)
static void init_attention_mask(ggml_tensor * mask, int64_t actual_q_len) {
    GGML_ASSERT(mask->type == GGML_TYPE_F32);

    float * data         = (float *) mask->data;
    int64_t kv_len       = mask->ne[0];
    int64_t q_len_padded = mask->ne[1];
    int64_t batch        = mask->ne[3];

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t q = 0; q < q_len_padded; q++) {
            for (int64_t kv = 0; kv < kv_len; kv++) {
                size_t idx = b * mask->nb[3] / sizeof(float) + q * mask->nb[1] / sizeof(float) +
                             kv * mask->nb[0] / sizeof(float);

                // For padded queries, always mask out
                if (q >= actual_q_len) {
                    data[idx] = -INFINITY;
                    continue;
                }

                // Causal mask: can only attend to previous positions
                int64_t kv_pos = kv;
                int64_t q_pos  = q + kv_len - actual_q_len;

                if (kv_pos <= q_pos) {
                    data[idx] = 0.0f;       // Can attend
                } else {
                    data[idx] = -INFINITY;  // Cannot attend
                }
            }
        }
    }
}

// Calculate MSE between tensors
static double calculate_mse(ggml_tensor * a, ggml_tensor * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    size_t             n = ggml_nelements(a);
    std::vector<float> a_data(n), b_data(n);

    // Convert to float
    if (a->type == GGML_TYPE_F32) {
        memcpy(a_data.data(), a->data, n * sizeof(float));
    } else if (a->type == GGML_TYPE_F16) {
        ggml_fp16_t * fp16_data = (ggml_fp16_t *) a->data;
        for (size_t i = 0; i < n; i++) {
            a_data[i] = ggml_fp16_to_fp32(fp16_data[i]);
        }
    }

    if (b->type == GGML_TYPE_F32) {
        memcpy(b_data.data(), b->data, n * sizeof(float));
    } else if (b->type == GGML_TYPE_F16) {
        ggml_fp16_t * fp16_data = (ggml_fp16_t *) b->data;
        for (size_t i = 0; i < n; i++) {
            b_data[i] = ggml_fp16_to_fp32(fp16_data[i]);
        }
    }

    double mse      = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = a_data[i] - b_data[i];
        mse += diff * diff;
        max_diff = std::max(max_diff, std::abs(diff));
    }

    printf("Max absolute difference: %.6e\n", max_diff);

    return mse / n;
}

// Print tensor info
static void print_tensor_info(const char * name, ggml_tensor * tensor) {
    printf("%-20s: [%4ld, %4ld, %4ld, %4ld] type=%-8s elements=%zu\n", name, tensor->ne[0], tensor->ne[1],
           tensor->ne[2], tensor->ne[3], ggml_type_name(tensor->type), ggml_nelements(tensor));
}

int main() {
    printf("Flash Attention Identical Test\n");
    printf("==============================\n\n");
    printf("This test uses the SAME K,V tensors for both methods\n");
    printf("to verify they produce identical results.\n\n");

    test_config cfg;

    // Initialize ggml
    struct ggml_init_params params = {
        .mem_size   = 2ULL * 1024 * 1024 * 1024,  // 2GB
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to initialize ggml context\n");
        return 1;
    }

    // Create tensors
    printf("Creating tensors...\n");

    // Query tensor
    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.head_dim, cfg.q_len, cfg.n_heads, cfg.batch_size);
    ggml_set_name(Q, "Q");

    // KV tensors
    ggml_tensor * K = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.head_dim, cfg.kv_len, cfg.n_heads, cfg.batch_size);
    ggml_set_name(K, "K");

    ggml_tensor * V = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.head_dim, cfg.kv_len, cfg.n_heads, cfg.batch_size);
    ggml_set_name(V, "V");

    // Attention mask (with padded q_len dimension)
    int64_t q_len_padded = GGML_PAD(cfg.q_len, GGML_KQ_MASK_PAD);

    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.kv_len, q_len_padded, 1, cfg.batch_size);
    ggml_set_name(mask, "mask");

    // For with_state version, split KV into two parts
    int64_t kv_split = cfg.kv_len / 2;

    ggml_tensor * K_part1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.head_dim, kv_split, cfg.n_heads, cfg.batch_size);
    ggml_set_name(K_part1, "K_part1");

    ggml_tensor * V_part1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.head_dim, kv_split, cfg.n_heads, cfg.batch_size);
    ggml_set_name(V_part1, "V_part1");

    ggml_tensor * K_part2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.head_dim, kv_split, cfg.n_heads, cfg.batch_size);
    ggml_set_name(K_part2, "K_part2");

    ggml_tensor * V_part2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.head_dim, kv_split, cfg.n_heads, cfg.batch_size);
    ggml_set_name(V_part2, "V_part2");

    ggml_tensor * mask_part1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_split, q_len_padded, 1, cfg.batch_size);
    ggml_set_name(mask_part1, "mask_part1");

    ggml_tensor * mask_part2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_split, q_len_padded, 1, cfg.batch_size);
    ggml_set_name(mask_part2, "mask_part2");

    // Initialize with random data
    printf("\nInitializing data...\n");
    fill_tensor_f16(Q);
    fill_tensor_f16(K);
    fill_tensor_f16(V);

    // Initialize mask
    init_attention_mask(mask, cfg.q_len);

    // Copy data to split tensors
    printf("Splitting KV tensors...\n");
    struct ggml_cgraph * split_graph = ggml_new_graph(ctx);

    // Create views and copy
    ggml_tensor * K_part1_view = ggml_view_3d(ctx, K, cfg.head_dim, kv_split, cfg.n_heads, K->nb[1], K->nb[2], 0);
    ggml_tensor * copy_k1      = ggml_cpy(ctx, K_part1_view, K_part1);
    ggml_build_forward_expand(split_graph, copy_k1);

    ggml_tensor * V_part1_view = ggml_view_3d(ctx, V, cfg.head_dim, kv_split, cfg.n_heads, V->nb[1], V->nb[2], 0);
    ggml_tensor * copy_v1      = ggml_cpy(ctx, V_part1_view, V_part1);
    ggml_build_forward_expand(split_graph, copy_v1);

    ggml_tensor * K_part2_view =
        ggml_view_3d(ctx, K, cfg.head_dim, kv_split, cfg.n_heads, K->nb[1], K->nb[2], kv_split * K->nb[1]);
    ggml_tensor * copy_k2 = ggml_cpy(ctx, K_part2_view, K_part2);
    ggml_build_forward_expand(split_graph, copy_k2);

    ggml_tensor * V_part2_view =
        ggml_view_3d(ctx, V, cfg.head_dim, kv_split, cfg.n_heads, V->nb[1], V->nb[2], kv_split * V->nb[1]);
    ggml_tensor * copy_v2 = ggml_cpy(ctx, V_part2_view, V_part2);
    ggml_build_forward_expand(split_graph, copy_v2);

    ggml_graph_compute_with_ctx(ctx, split_graph, 4);

    // Split mask
    float * mask_data       = (float *) mask->data;
    float * mask_part1_data = (float *) mask_part1->data;
    float * mask_part2_data = (float *) mask_part2->data;

    for (int64_t b = 0; b < cfg.batch_size; b++) {
        for (int64_t q = 0; q < q_len_padded; q++) {
            // Copy first part
            for (int64_t kv = 0; kv < kv_split; kv++) {
                size_t src_idx = b * (mask->nb[3] / sizeof(float)) + q * (mask->nb[1] / sizeof(float)) +
                                 kv * (mask->nb[0] / sizeof(float));
                size_t dst_idx = b * (mask_part1->nb[3] / sizeof(float)) + kv * (mask_part1->nb[0] / sizeof(float));
                q *(mask_part1->nb[1] / sizeof(float)) + mask_part1_data[dst_idx] = mask_data[src_idx];
            }
            // Copy second part
            for (int64_t kv = 0; kv < kv_split; kv++) {
                size_t src_idx = b * (mask->nb[3] / sizeof(float)) + q * (mask->nb[1] / sizeof(float)) +
                                 (kv_split + kv) * (mask->nb[0] / sizeof(float));
                size_t dst_idx = b * (mask_part2->nb[3] / sizeof(float)) + q * (mask_part2->nb[1] / sizeof(float)) +
                                 kv * (mask_part2->nb[0] / sizeof(float));
                mask_part2_data[dst_idx] = mask_data[src_idx];
            }
        }
    }

    // Print tensor info
    printf("\nTensor Information:\n");
    print_tensor_info("Q", Q);
    print_tensor_info("K", K);
    print_tensor_info("V", V);
    print_tensor_info("mask", mask);
    printf("\nSplit tensors:\n");
    print_tensor_info("K_part1", K_part1);
    print_tensor_info("K_part2", K_part2);

    // Build flash attention graphs
    printf("\nBuilding flash attention graphs...\n");
    float scale = 1.0f / sqrtf((float) cfg.head_dim);

    // Standard flash attention graph
    struct ggml_cgraph * flash_std_graph = ggml_new_graph(ctx);
    ggml_tensor *        result_std      = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(result_std, GGML_PREC_F32);
    ggml_set_name(result_std, "result_std");
    ggml_build_forward_expand(flash_std_graph, result_std);

    // Flash attention with state graph
    struct ggml_cgraph * flash_state_graph = ggml_new_graph(ctx);
    ggml_tensor * result_state = ggml_flash_attn_ext_with_state(ctx, Q, K_part2, V_part2, mask_part2, K_part1, V_part1,
                                                                mask_part1, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(result_state, GGML_PREC_WITH_STATE);
    ggml_set_name(result_state, "result_state");
    ggml_build_forward_expand(flash_state_graph, result_state);

    // Compute flash attention
    printf("\nComputing flash attention...\n");

    clock_t start_std = clock();
    ggml_graph_compute_with_ctx(ctx, flash_std_graph, 4);
    clock_t end_std  = clock();
    double  time_std = (double) (end_std - start_std) / CLOCKS_PER_SEC * 1000.0;

    clock_t start_state = clock();
    ggml_graph_compute_with_ctx(ctx, flash_state_graph, 4);
    clock_t end_state  = clock();
    double  time_state = (double) (end_state - start_state) / CLOCKS_PER_SEC * 1000.0;

    // Compare results
    printf("\nResults:\n");
    printf("----------------------------------------\n");
    print_tensor_info("Standard result", result_std);
    print_tensor_info("With-state result", result_state);

    double mse = calculate_mse(result_std, result_state);
    printf("\nMean Squared Error: %.6e\n", mse);

    // Performance comparison
    printf("\nPerformance:\n");
    printf("Standard flash attention: %.2f ms\n", time_std);
    printf("Flash attention with state: %.2f ms\n", time_state);
    if (time_state > 0) {
        printf("Speedup: %.2fx\n", time_std / time_state);
    }

    // Assessment
    printf("\nAssessment:\n");
    if (mse < 1e-10) {
        printf("✅ Identical: Results are practically the same (MSE < 1e-10)\n");
    } else if (mse < 1e-6) {
        printf("✅ Excellent: Results match very closely (MSE < 1e-6)\n");
    } else if (mse < 1e-4) {
        printf("✅ Good: Results match well (MSE < 1e-4)\n");
    } else if (mse < 1e-2) {
        printf("⚠️  Acceptable: Some differences observed (MSE < 1e-2)\n");
    } else {
        printf("❌ Poor: Significant differences (MSE >= 1e-2)\n");
    }

    // Cleanup
    ggml_free(ctx);

    return 0;
}
