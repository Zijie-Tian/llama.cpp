#include "../ggml/src/ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef LLAMA_TORCH_AVAILABLE
#    include <torch/torch.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

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

#ifdef LLAMA_TORCH_AVAILABLE
// Convert ggml tensor to a torch::Tensor (float32)
static torch::Tensor ggml_to_torch(ggml_tensor * tensor) {
    auto               tt = ggml_get_type_traits(tensor->type);
    size_t             n  = ggml_nelements(tensor);
    std::vector<float> data(n);
    if (tensor->type == GGML_TYPE_F32) {
        memcpy(data.data(), tensor->data, n * sizeof(float));
    } else if (tt->to_float) {
        tt->to_float(tensor->data, data.data(), n);
    } else {
        printf("Unsupported tensor type for torch conversion: %s\n", ggml_type_name(tensor->type));
        return {};
    }

    std::vector<int64_t> sizes;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] > 1 || i == 0) {
            sizes.push_back(tensor->ne[i]);
        }
    }
    return torch::from_blob(data.data(), sizes, torch::kFloat32).clone();
}
#endif  // LLAMA_TORCH_AVAILABLE

static void print_tensor_info(const char * name, ggml_tensor * tensor) {
    printf("%s: [%ld, %ld, %ld, %ld] type=%s, elements=%ld\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2],
           tensor->ne[3], ggml_type_name(tensor->type), ggml_nelements(tensor));
}

static void print_f32_sample(const char * name, ggml_tensor * tensor, int max_elements = 10) {
    if (tensor->type != GGML_TYPE_F32) {
        printf("%s: Not F32 tensor (type=%s)\n", name, ggml_type_name(tensor->type));
        return;
    }

    float * data              = (float *) tensor->data;
    size_t  n_elements        = ggml_nelements(tensor);
    size_t  elements_to_print = std::min((size_t) max_elements, n_elements);

    printf("%s sample values: ", name);
    for (size_t i = 0; i < elements_to_print; i++) {
        printf("%.6f ", data[i]);
    }
    if (elements_to_print < n_elements) {
        printf("... (total %ld elements)", n_elements);
    }
    printf("\n");
}

/**
 * Print a visualization of the KQV attention mask.
 * Shows which tokens can attend to which other tokens.
 * x = can attend (0 or greater)
 * - = cannot attend (-INFINITY)
 */
static void print_kqv_mask(ggml_tensor * mask) {
    GGML_TENSOR_LOCALS(int64_t, ne_mask, mask, ne)
    GGML_TENSOR_LOCALS(int64_t, nb_mask, mask, nb)

    printf("\n=== KQV Attention Mask ===\n");
    printf("KV tokens →\n");

    // Print column numbers
    for (int i = 0; i < ne_mask0; i++) {
        printf("%d", i % 10);
    }
    printf("\n");

    // Print separator
    for (int i = 0; i < ne_mask0; i++) {
        printf("=");
    }
    printf("\n");

    // Get mask data pointer
    const char *  mask_nonfp32 = (const char *) mask->data;
    const float * mask_fp32    = (const float *) mask->data;

    ggml_to_float_t to_float = ggml_get_type_traits(mask->type)->to_float;

    float * row_buffer = (float *) malloc(ne_mask0 * sizeof(float));
    for (int i = 0; i < ne_mask1; ++i) {
        if (to_float) {
            to_float(mask_nonfp32 + i * nb_mask1, row_buffer, ne_mask0);

            for (int j = 0; j < ne_mask0; ++j) {
                if (row_buffer[j] == 0.f) {
                    printf("x");
                } else if (row_buffer[j] == -INFINITY) {
                    printf("-");
                } else {
                    printf("?");
                }
            }
            printf("\n");
        } else {
            for (int j = 0; j < ne_mask0; ++j) {
                if (mask_fp32[j] == 0.f) {
                    printf("x");
                } else if (mask_fp32[j] == -INFINITY) {
                    printf("-");
                } else {
                    printf("?");
                }
            }
            printf("\n");
        }
    }

    free(row_buffer);
}

static float tensor_max_diff(ggml_tensor * a, ggml_tensor * b) {
    if (ggml_nelements(a) != ggml_nelements(b) || a->type != b->type) {
        printf("ERROR: Tensors have different sizes or types\n");
        return -1.0f;
    }

    if (a->type != GGML_TYPE_F32) {
        printf("ERROR: Only F32 tensors supported for comparison\n");
        return -1.0f;
    }

    float * data_a     = (float *) a->data;
    float * data_b     = (float *) b->data;
    size_t  n_elements = ggml_nelements(a);

    float max_diff = 0.0f;
    for (size_t i = 0; i < n_elements; i++) {
        float diff = std::abs(data_a[i] - data_b[i]);
        max_diff   = std::max(max_diff, diff);
    }

    return max_diff;
}

static void reset_state_tensor(ggml_tensor * state) {
    float * state_data = (float *) state->data;
    size_t  n_pairs    = ggml_nelements(state) / 2;

    for (size_t i = 0; i < n_pairs; i++) {
        state_data[i * 2 + 0] = -INFINITY;  // M (max KQ value)
        state_data[i * 2 + 1] = 0.0f;       // S (sum)
    }
}

// Simple tensor info without detailed data
static void print_tensor_summary(ggml_tensor * tensor, const std::string & name) {
    if (!tensor) {
        printf(
            "| %-20s | NULL                                | NULL                                | NULL     | NULL     "
            "  |\n",
            name.c_str());
        return;
    }
    printf("| %-21s | [%4ld,%4ld,%4ld,%4ld]                 | [%8ld,%8ld,%8ld,%8ld]            | %-8s | %10zu |\n",
           name.c_str(), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[0], tensor->nb[1],
           tensor->nb[2], tensor->nb[3], ggml_type_name(tensor->type), ggml_nelements(tensor));
}

/**
 * Print a visualization of the KQV attention mask.
 * Shows which tokens can attend to which other tokens.
 * x = can attend (0 or greater)
 * - = cannot attend (-INFINITY)
 * For large n_kv, only prints first and last few columns with ellipsis
 */
static void ggml_print_mask(const ggml_tensor * mask, int64_t n_kv, int64_t n_tokens) {
    printf("\n=== KQV Attention Mask ===\n");
    printf("KV tokens →\n");

    const int  preview_size  = 8;  // Number of columns to show at start/end
    // const bool truncate = n_kv > 3 * preview_size;
    const bool truncate      = false;
    const int  display_width = truncate ? 2 * preview_size + 3 : n_kv;

    // Print column numbers
    printf("     ");
    for (int i = 0; i < display_width; i++) {
        if (truncate && i == preview_size) {
            printf("...");
        } else if (truncate && i > preview_size) {
            printf("%d", (n_kv - (2 * preview_size - i)) % 10);
        } else {
            printf("%d", i % 10);
        }
    }
    printf("\n");

    // Print separator
    printf("     ");
    for (int i = 0; i < display_width; i++) {
        if (truncate && i == preview_size) {
            printf("...");
        } else {
            printf("=");
        }
    }
    printf("\n");

    const int  row_preview   = 5;  // Number of rows to show at start/end
    // const bool truncate_rows = n_tokens > 2 * row_preview + 1;
    const bool truncate_rows = false;

    // printf("mask type : %s", ggml_type_name(mask->type));

    if (mask->type == GGML_TYPE_F32) {
        float * mask_data = (float *) mask->data;

        // Print each row of the mask
        for (int j = 0; j < n_tokens; j++) {
            // Skip middle rows if truncating
            if (truncate_rows && j == row_preview) {
                printf("... |\n");
                j = n_tokens - row_preview - 1;
                continue;
            }

            printf("%3d |", j);  // Row number
            for (int i = 0; i < display_width; i++) {
                if (truncate && i == preview_size) {
                    printf("...");
                } else {
                    int   idx = truncate && i > preview_size ? n_kv - (2 * preview_size - i) : i;
                    float val = mask_data[j * n_kv + idx];
                    printf("%c", (val == 0.0f) ? 'x' : '-');
                }
            }
            printf("\n");
        }
    } else {
        ggml_fp16_t * mask_data = (ggml_fp16_t *) mask->data;

        for (int j = 0; j < n_tokens; j++) {
            // Skip middle rows if truncating
            if (truncate_rows && j == row_preview) {
                printf("... |\n");
                j = n_tokens - row_preview - 1;
                continue;
            }

            printf("%3d |", j);  // Row number
            for (int i = 0; i < display_width; i++) {
                if (truncate && i == preview_size) {
                    printf("...");
                } else {
                    int   idx = truncate && i > preview_size ? n_kv - (2 * preview_size - i) : i;
                    float val = ggml_fp16_to_fp32(mask_data[j * n_kv + idx]);
                    printf("%c", (val == 0) ? 'x' : '-');
                }
            }
            printf("\n");
        }
    }
    printf("\n");
}

int main() {
    printf("=== Flash Attention State Tensor - Comprehensive Test ===\n");

    // Test parameters
    const int head_dim       = 128;
    const int n_heads        = 32;
    const int n_kv_heads     = 1;
    const int seq_len        = 1;
    const int kv_len         = 256;  // Will be split into segments
    const int n_threads      = 1;
    const int kv_segments    = 2;     // Split KV into 2 segments
    const int kv_segment_len = 128;

    // // Test parameters
    // const int head_dim       = 4;
    // const int n_heads        = 4;
    // const int n_kv_heads     = 2;
    // const int seq_len        = 2;
    // const int kv_len         = 1024 * 16;  // Will be split into segments
    // const int n_threads      = 12;
    // const int kv_segments    = 2;  // Split KV into 2 segments
    // const int kv_segment_len = kv_len / kv_segments;

    printf("Test Parameters:\n");
    printf("  head_dim=%d, n_heads=%d, n_kv_heads=%d\n", head_dim, n_heads, n_kv_heads);
    printf("  seq_len=%d, kv_len=%d\n", seq_len, kv_len);
    printf("  kv_segments=%d, kv_segment_len=%d\n", kv_segments, kv_segment_len);

    // Initialize ggml context
    const size_t            ctx_size = 1024 * 1024 * 1024;  // 1GB
    struct ggml_init_params params   = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,  //> This will allocate memory for this context.
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize ggml context\n");
        return 1;
    }

    //> ============================================================================
    //> Create and initialize tensors with FIXED data
    //> ============================================================================
    printf("\n--- Creating Fixed Test Data ---\n");

    // Create tensors for flash attention
    // Format: [head_dim, seq_len, n_heads,    1] for Q
    // Format: [head_dim, kv_len,  n_kv_heads, 1] for K, V
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);

    // Create mask tensor with proper padding
    const int     padded_kv_len  = GGML_PAD(kv_len, 64);
    const int     padded_seq_len = GGML_PAD(seq_len, GGML_KQ_MASK_PAD);
    ggml_tensor * mask           = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, padded_kv_len, padded_seq_len);

    // Create state tensor: [2, n_heads * seq_len] for [M, S] pairs
    ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, n_heads * seq_len);

    print_tensor_info("Q", q);
    print_tensor_info("K", k);
    print_tensor_info("V", v);
    print_tensor_info("Mask", mask);
    print_tensor_info("State", state);

    // Fill with FIXED reproducible data
    printf("\nGenerating fixed test data (seed=42)...\n");
    // fill_tensor_f32(q, -0.8f, 0.8f);
    fill_tensor_f16(k, -0.6f, 0.6f);
    fill_tensor_f16(v, -0.7f, 0.7f);

    set_tensor_f32(q, 1.0f);
    // set_tensor_f16(k, 0.5f);
    // set_tensor_f16(v, 0.25f);

    // Initialize mask (causal mask - positions can only see previous and current KV)
    ggml_fp16_t * mask_data = (ggml_fp16_t *) mask->data;
    memset(mask_data, 0, ggml_nbytes(mask));
    for (int i = 0; i < padded_seq_len; i++) {
        for (int j = 0; j < padded_kv_len; j++) {
            // For testing: allow all query positions to see all KV positions < kv_len
            // This ensures both segments have valid attention weights
            if (i < seq_len && j < kv_len) {
                mask_data[i * padded_kv_len + j] = ggml_fp32_to_fp16(0.0f);
            } else {
                mask_data[i * padded_kv_len + j] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
    }

    ggml_print_mask(mask, kv_len, seq_len);

    // Print table header for tensor summary
    printf(
        "\n+-----------------------+---------------------------------------+-------------------------------------------"
        "-------+----------+------------+\n");
    printf("| %-21s | %-37s | %-48s | %-8s | %-10s |\n", "Tensor Name", "Dimensions [d0,d1,d2,d3]",
           "Strides [s0,s1,s2,s3]", "Type", "Elements");
    printf(
        "+-----------------------+---------------------------------------+---------------------------------------------"
        "-----+----------+------------+\n");

    //> ============================================================================
    //> Test 1: Standard Flash Attention (Reference Result)
    //> ============================================================================
    // printf("\n--- Test 1: Standard Flash Attention (Reference) ---\n");

    print_tensor_summary(q, "Q");
    print_tensor_summary(k, "K");
    print_tensor_summary(v, "V");
    print_tensor_summary(mask, "Mask");
    print_tensor_summary(state, "State");

    ggml_tensor * result_standard = ggml_flash_attn_ext(ctx, q, k, v, mask,
                                                        1.0f / std::sqrt(head_dim),   // scale
                                                        0.0f,                        // max_bias
                                                        0.0f                    // logit_softcap
    );
    ggml_flash_attn_ext_set_prec(result_standard, GGML_PREC_F32);

    if (!result_standard) {
        printf("ERROR: Failed to create standard flash attention operation\n");
        ggml_free(ctx);
        return 1;
    }

    struct ggml_cgraph * graph_standard = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_standard, result_standard);

    enum ggml_status status_standard = ggml_graph_compute_with_ctx(ctx, graph_standard, n_threads);

    if (status_standard != GGML_STATUS_SUCCESS) {
        printf("ERROR: Standard flash attention failed with status: %d\n", status_standard);
        ggml_free(ctx);
        return 1;
    }

    // printf("Standard flash attention computation successful\n");
    // print_f32_sample("Standard result", result_standard, 8);

    printf(
        "+-----------------------+---------------------------------------+---------------------------------------------"
        "-----+----------+------------+\n");

    //> ============================================================================
    //> Test 2: Segmented Flash Attention with State Accumulation
    //> ============================================================================
    // printf("\n--- Test 2: Segmented Flash Attention with State ---\n");

    ggml_tensor * q_fp16 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, seq_len, n_heads, 1);
    ggml_tensor * q_quant_op = ggml_cpy(ctx, q, q_fp16);
    ggml_cgraph * gf_quant = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_quant, q_quant_op);
    ggml_graph_compute_with_ctx(ctx, gf_quant, n_threads);

    const int64_t PACK_SIZE       = 128;  //> 128x128
    const int64_t PACK_CHUNK_SIZE = 128;  //> 128x128
    const int64_t n_chunks        = (kv_len - kv_segment_len) / PACK_CHUNK_SIZE;

    // NOTE: Reset state which are the max KQ values and sum for each head
    reset_state_tensor(state);

    // printf("Processing segments using unified op...\n");

    ggml_tensor * k_fp16_seg =
        ggml_view_4d(ctx, k, head_dim, kv_segment_len, n_kv_heads, 1, k->nb[1], k->nb[2], k->nb[3], 0);
    ggml_tensor * v_fp16_seg =
        ggml_view_4d(ctx, v, head_dim, kv_segment_len, n_kv_heads, 1, v->nb[1], v->nb[2], v->nb[3], 0);

    // NOTE: Reshape and quantization.
    ggml_tensor * k_quant_seg = ggml_view_4d(ctx, k, head_dim * PACK_CHUNK_SIZE * n_kv_heads, n_chunks, 1, 1, k->nb[1],
                                             k->nb[2], k->nb[3], kv_segment_len * k->nb[1]);
    ggml_tensor * v_quant_seg = ggml_view_4d(ctx, v, head_dim * PACK_CHUNK_SIZE * n_kv_heads, n_chunks, 1, 1, v->nb[1],
                                             v->nb[2], v->nb[3], kv_segment_len * v->nb[1]);
    ggml_tensor * k_qlutattn_seg = ggml_new_tensor_4d(ctx, GGML_TYPE_QLUTATTN_KV4_128x128,
                                                      head_dim * PACK_CHUNK_SIZE * n_kv_heads, n_chunks, 1, 1);
    ggml_tensor * v_qlutattn_seg = ggml_new_tensor_4d(ctx, GGML_TYPE_QLUTATTN_KV4_128x128,
                                                      head_dim * PACK_CHUNK_SIZE * n_kv_heads, n_chunks, 1, 1);

    //> Do quantization.
    ggml_tensor * k_qlutattn_seg_quant = ggml_cpy(ctx, k_quant_seg, k_qlutattn_seg);
    ggml_tensor * v_qlutattn_seg_quant = ggml_cpy(ctx, v_quant_seg, v_qlutattn_seg);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, k_qlutattn_seg_quant);
    ggml_build_forward_expand(gf, v_qlutattn_seg_quant);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const int     padded_segment_len = GGML_PAD(kv_segment_len, 64);
    ggml_tensor * mask_fp16_seg      = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_segment_len, padded_seq_len);
    ggml_tensor * mask_quant_seg =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F16, padded_kv_len - kv_segment_len, padded_seq_len);

    ggml_fp16_t * mask_fp16_data  = (ggml_fp16_t *) mask_fp16_seg->data;
    ggml_fp16_t * mask_quant_data = (ggml_fp16_t *) mask_quant_seg->data;
    // // memset(mask_data, 0, ggml_nbytes(mask_fp16_seg));
    // // memset(mask_data, 0, ggml_nbytes(mask_quant_seg));
    // for (int i = 0; i < padded_seq_len; i++) {
    //     for (int j = 0; j < kv_segment_len; j++) {
            
    //         mask_fp16_data[i * kv_segment_len + j] = ggml_fp32_to_fp16(0.0f);

    //         // Causal masking - positions can only see up to their position
    //         // if (i < seq_len && j < kv_len) {
    //         //     // Mask out future positions with -INFINITY
    //         //     mask_fp16_data[i * kv_segment_len + j] = ggml_fp32_to_fp16(0.0f);
    //         // } else {
    //         //     // Mask out future positions with -INFINITY
    //         //     mask_fp16_data[i * kv_segment_len + j] = ggml_fp32_to_fp16(-INFINITY);
    //         // }
    //     }
    // }

    for (int i = 0; i < padded_seq_len; i++) {
        for (int j = 0; j < padded_kv_len; j++) {
            // Causal masking - positions can only see up to their position
            // The actual KV position in the full sequence is j + kv_segment_len
            int actual_kv_pos = j + kv_segment_len;

            if (j < kv_segment_len) {
                mask_fp16_data[i * kv_segment_len + j] = mask_data[i * padded_kv_len + j];
            } else {
                mask_quant_data[i * (padded_kv_len - kv_segment_len) + j] = mask_data[i * padded_kv_len + j];
            }

        }
    }

    print_tensor_summary(q, "Q");
    print_tensor_summary(k_fp16_seg, "K_FP16_SEG");
    print_tensor_summary(v_fp16_seg, "V_FP16_SEG");
    print_tensor_summary(k_quant_seg, "K_QUANT_SEG");
    print_tensor_summary(v_quant_seg, "V_QUANT_SEG");
    print_tensor_summary(mask_fp16_seg, "MASK_FP16_SEG");
    print_tensor_summary(mask_quant_seg, "MASK_QUANT_SEG");
    
    printf(
        "+-----------------------+---------------------------------------+---------------------------------------------"
        "-----+----------+------------+\n");

    ggml_print_mask(mask_fp16_seg, kv_len, seq_len);


    ggml_tensor * result_seg =
        ggml_flash_attn_mixed(ctx, q_fp16, k_fp16_seg, v_fp16_seg, mask_fp16_seg, k_qlutattn_seg, v_qlutattn_seg,
                              mask_quant_seg, 1.0f / std::sqrt(head_dim), 0.0f, 0.0f);
    // ggml_flash_attn_ext_set_prec(result_seg, GGML_PREC_WITH_STATE);
    ggml_flash_attn_ext_set_prec(result_seg, GGML_PREC_MIXED);

    struct ggml_cgraph * graph_seg = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_seg, result_seg);
    enum ggml_status status_seg = ggml_graph_compute_with_ctx(ctx, graph_seg, n_threads);
    if (status_seg != GGML_STATUS_SUCCESS) {
        printf("ERROR: Segmented flash attention failed with status: %d\n", status_seg);
        ggml_free(ctx);
        return 1;
    }

    // ggml_print_mask(mask_quant_seg, kv_len, seq_len);

    //> =====================================================================
    //> Test 3: PyTorch Verification using scaled_dot_product_attention
    //> =====================================================================
    printf("\n--- PyTorch Verification ---\n");

    std::vector<float> torch_result_data;
    bool               torch_success = false;

#ifdef LLAMA_TORCH_AVAILABLE
    try {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);

        auto q_t = torch::zeros({ 1, n_heads, seq_len, head_dim }, options);
        auto k_t = torch::zeros({ 1, n_kv_heads, kv_len, head_dim }, options);
        auto v_t = torch::zeros({ 1, n_kv_heads, kv_len, head_dim }, options);

        float * qd = q_t.data_ptr<float>();
        float * kd = k_t.data_ptr<float>();
        float * vd = v_t.data_ptr<float>();

        for (int h = 0; h < n_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < head_dim; ++d) {
                    int gi = d + s * head_dim + h * head_dim * seq_len;
                    int ti = h * seq_len * head_dim + s * head_dim + d;
                    qd[ti] = ((float *) q->data)[gi];
                }
            }
        }

        for (int h = 0; h < n_kv_heads; ++h) {
            for (int s = 0; s < kv_len; ++s) {
                for (int d = 0; d < head_dim; ++d) {
                    int gi = d + s * head_dim + h * head_dim * kv_len;
                    int ti = h * kv_len * head_dim + s * head_dim + d;
                    kd[ti] = ggml_fp16_to_fp32(((ggml_fp16_t *) k->data)[gi]);
                    vd[ti] = ggml_fp16_to_fp32(((ggml_fp16_t *) v->data)[gi]);
                }
            }
        }

        auto          mask_t = torch::ones({ 1, n_heads, seq_len, kv_len }, torch::TensorOptions().dtype(torch::kBool));
        bool *        mask_td = mask_t.data_ptr<bool>();
        ggml_fp16_t * mask_d  = (ggml_fp16_t *) mask->data;

        for (int h = 0; h < n_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < kv_len; ++d) {
                    int   gi    = d + s * padded_kv_len;
                    float val   = GGML_FP16_TO_FP32(mask_d[gi]);
                    int   ti    = h * seq_len * kv_len + s * kv_len + d;
                    mask_td[ti] = (val == 0.0f);
                }
            }
        }

        if (n_heads > n_kv_heads) {
            k_t = k_t.repeat_interleave(n_heads / n_kv_heads, 1);
            v_t = v_t.repeat_interleave(n_heads / n_kv_heads, 1);
        }

        float scale     = 1.0f / std::sqrt((float) head_dim);
        auto  torch_res = torch::scaled_dot_product_attention(q_t, k_t, v_t, mask_t, 0.0, false, scale);
        torch_res       = torch_res.permute({ 0, 2, 1, 3 }).contiguous();

        float * trd   = torch_res.data_ptr<float>();
        size_t  numel = torch_res.numel();
        torch_result_data.resize(numel);
        for (int h = 0; h < n_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < head_dim; ++d) {
                    int ti                = h * seq_len * head_dim + s * head_dim + d;
                    int ci                = d + s * head_dim + h * head_dim * seq_len;
                    torch_result_data[ci] = trd[ti];
                }
            }
        }
        torch_success = true;
        printf("PyTorch computation successful\n");
    } catch (const std::exception & e) {
        printf("PyTorch verification failed: %s\n", e.what());
        torch_success = false;
    }
#else
    printf("PyTorch verification skipped (PyTorch not available)\n");
#endif

    // ============================================================================
    // Test 3: Compare Results
    // ============================================================================
    printf("\n--- Unified Results Comparison ---\n");

    float * standard_data  = (float *) result_standard->data;
    float * segmented_data = (float *) result_seg->data;
    size_t  n_elems        = ggml_nelements(result_standard);
    if (torch_success) {
        n_elems = std::min(n_elems, torch_result_data.size());
    }

    float max_std_seg = 0.0f, max_std_torch = 0.0f, max_seg_torch = 0.0f;
    for (size_t i = 0; i < n_elems; ++i) {
        float s     = standard_data[i];
        float g     = segmented_data[i];
        max_std_seg = std::max(max_std_seg, std::abs(s - g));
        if (torch_success) {
            float t       = torch_result_data[i];
            max_std_torch = std::max(max_std_torch, std::abs(s - t));
            max_seg_torch = std::max(max_seg_torch, std::abs(g - t));
        }
    }

    printf("Max diff standard vs segmented : %.6e\n", max_std_seg);
    if (torch_success) {
        printf("Max diff standard vs torch     : %.6e\n", max_std_torch);
        printf("Max diff segmented vs torch    : %.6e\n", max_seg_torch);
    }

    printf("\nDetailed Comparison Table (first 128 elements):\n");
    if (torch_success) {
        printf("Idx | Standard    | Segmented   | Torch       | S-G Diff   | S-T Diff   | G-T Diff\n");
        printf("----|-------------|-------------|-------------|-----------|-----------|-----------\n");
    } else {
        printf("Idx | Standard    | Segmented   | S-G Diff\n");
        printf("----|-------------|-------------|-----------\n");
    }

    // ============================================================================
    // Print first 128 elements
    // ============================================================================

    size_t show = std::min((size_t) 128, n_elems);
    for (size_t i = 0; i < show; ++i) {
        float s = standard_data[i];
        float g = segmented_data[i];
        if (torch_success) {
            float t = torch_result_data[i];
            printf("%3zu | %11.6f | %11.6f | %11.6f | %.6e | %.6e | %.6e\n", i, s, g, t, std::abs(s - g),
                   std::abs(s - t), std::abs(g - t));
        } else {
            printf("%3zu | %11.6f | %11.6f | %.6e\n", i, s, g, std::abs(s - g));
        }
    }

    // ============================================================================
    // Print mask
    // ============================================================================

    // print_kqv_mask(mask);
    // print_kqv_mask(mask_fp16_seg);
    // print_kqv_mask(mask_quant_seg);

    // ============================================================================
    // Max error point
    // ============================================================================
    printf("\n=== Max error point ===\n");

    // First find where the large differences occur
    printf("\nScanning for extreme differences...\n");
    size_t max_diff_idx = 0;
    float  max_diff_val = 0.0f;
    for (size_t i = 0; i < n_elems; ++i) {
        float diff = std::abs(standard_data[i] - segmented_data[i]);
        if (diff > max_diff_val) {
            max_diff_val = diff;
            max_diff_idx = i;
        }
    }

    // Print context around max difference
    printf("\nMax difference of %.2e found at index %zu\n", max_diff_val, max_diff_idx);
    printf("Context around max difference (idx %zu):\n", max_diff_idx);
    size_t context_start = (max_diff_idx >= 10) ? max_diff_idx - 10 : 0;
    size_t context_end   = std::min(max_diff_idx + 10, n_elems);

    for (size_t i = context_start; i < context_end; ++i) {
        float s = standard_data[i];
        float g = segmented_data[i];
        printf("%s%3zu | S: %11.6f | G: %11.6f | Diff: %.6e\n", (i == max_diff_idx) ? ">>> " : "    ", i, s, g,
               std::abs(s - g));
    }

    // Also check for patterns of zeros
    printf("\nChecking for zero patterns in segmented result...\n");
    size_t zero_count = 0;
    size_t first_zero = n_elems;
    for (size_t i = 0; i < n_elems; ++i) {
        if (segmented_data[i] == 0.0f) {
            zero_count++;
            if (first_zero == n_elems) {
                first_zero = i;
            }
        }
    }
    printf("Found %zu zeros in segmented result (%.2f%%)\n", zero_count, 100.0f * zero_count / n_elems);
    if (zero_count > 0) {
        printf("First zero at index %zu\n", first_zero);
        // Print pattern around first zeros
        size_t pattern_start = (first_zero >= 5) ? first_zero - 5 : 0;
        size_t pattern_end   = std::min(first_zero + 20, n_elems);
        printf("Pattern around first zeros:\n");
        for (size_t i = pattern_start; i < pattern_end; ++i) {
            printf("%3zu: S=%.6f, G=%.6f %s\n", i, standard_data[i], segmented_data[i],
                   (segmented_data[i] == 0.0f) ? "<-- ZERO" : "");
        }
    }

    // Check for systematic scale differences
    double sum_ratio    = 0.0;
    int    valid_ratios = 0;
    for (size_t i = 0; i < std::min((size_t) 1000, n_elems); ++i) {
        if (std::abs(standard_data[i]) > 1e-6 && std::abs(segmented_data[i]) > 1e-6) {
            double ratio = segmented_data[i] / standard_data[i];
            if (std::abs(ratio) < 1000.0) {  // Ignore extreme outliers
                sum_ratio += ratio;
                valid_ratios++;
            }
        }
    }
    if (valid_ratios > 0) {
        printf("\nAverage ratio (segmented/standard) for first 1000 elements: %.6f\n", sum_ratio / valid_ratios);
    }

    const float tolerance = 1e-3f;
    bool        pass      = max_std_seg < tolerance;
    if (torch_success) {
        pass = pass && max_std_torch < tolerance && max_seg_torch < tolerance;
    }

    // ============================================================================
    // Final Results
    // ============================================================================
    printf("\n=== Final Test Results ===\n");

    if (pass) {
        printf("🎉 ALL TESTS PASSED!\n");
        printf("✅ Segmented flash attention with state produces identical results\n");
        if (torch_success) {
            printf("✅ PyTorch results match GGML outputs\n");
        }
    } else {
        printf("❌ TESTS FAILED!\n");
    }

    printf("\nMax difference S-G: %.2e (tolerance: %.2e)\n", max_std_seg, tolerance);
    if (torch_success) {
        printf("Max difference S-T: %.2e\n", max_std_torch);
        printf("Max difference G-T: %.2e\n", max_seg_torch);
    }

    // Cleanup
    ggml_free(ctx);
    return pass ? 0 : 1;
}
