#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cstdlib>
#include <algorithm>
#include <iostream>

int main() {
    printf("=== Flash Attention Debug Test ===\n");

    // Minimal test parameters
    const int head_dim   = 4;    // Very small for manual verification
    const int n_heads    = 1;    // Single head
    const int n_kv_heads = 1;    // Single KV head
    const int seq_len    = 1;    // Single query
    const int kv_len     = 2;    // Two KV tokens
    const int n_threads  = 1;    // Single thread

    printf("Test Parameters: head_dim=%d, seq_len=%d, kv_len=%d\n", head_dim, seq_len, kv_len);

    // Initialize ggml context
    const size_t ctx_size = 64*1024*1024; // 64MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize ggml context\n");
        return 1;
    }

    // Create tensors
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);
    
    const int padded_kv_len = GGML_PAD(kv_len, 64);
    const int padded_seq_len = GGML_PAD(seq_len, GGML_KQ_MASK_PAD);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, padded_kv_len, padded_seq_len);

    ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, n_heads * seq_len);

    // Set simple, known values for manual verification
    printf("\nSetting test data:\n");
    
    // Q = [1, 0, 0, 0]
    float* q_data = (float*)q->data;
    q_data[0] = 1.0f; q_data[1] = 0.0f; q_data[2] = 0.0f; q_data[3] = 0.0f;
    printf("Q = [%.1f, %.1f, %.1f, %.1f]\n", q_data[0], q_data[1], q_data[2], q_data[3]);

    // K0 = [1, 0, 0, 0], K1 = [0, 1, 0, 0]  
    ggml_fp16_t* k_data = (ggml_fp16_t*)k->data;
    k_data[0] = ggml_fp32_to_fp16(1.0f); k_data[1] = ggml_fp32_to_fp16(0.0f); 
    k_data[2] = ggml_fp32_to_fp16(0.0f); k_data[3] = ggml_fp32_to_fp16(0.0f);
    k_data[4] = ggml_fp32_to_fp16(0.0f); k_data[5] = ggml_fp32_to_fp16(1.0f);
    k_data[6] = ggml_fp32_to_fp16(0.0f); k_data[7] = ggml_fp32_to_fp16(0.0f);
    printf("K0 = [%.1f, %.1f, %.1f, %.1f]\n", 
           GGML_FP16_TO_FP32(k_data[0]), GGML_FP16_TO_FP32(k_data[1]), 
           GGML_FP16_TO_FP32(k_data[2]), GGML_FP16_TO_FP32(k_data[3]));
    printf("K1 = [%.1f, %.1f, %.1f, %.1f]\n", 
           GGML_FP16_TO_FP32(k_data[4]), GGML_FP16_TO_FP32(k_data[5]), 
           GGML_FP16_TO_FP32(k_data[6]), GGML_FP16_TO_FP32(k_data[7]));

    // V0 = [1, 2, 3, 4], V1 = [5, 6, 7, 8]
    ggml_fp16_t* v_data = (ggml_fp16_t*)v->data;
    v_data[0] = ggml_fp32_to_fp16(1.0f); v_data[1] = ggml_fp32_to_fp16(2.0f);
    v_data[2] = ggml_fp32_to_fp16(3.0f); v_data[3] = ggml_fp32_to_fp16(4.0f);
    v_data[4] = ggml_fp32_to_fp16(5.0f); v_data[5] = ggml_fp32_to_fp16(6.0f);
    v_data[6] = ggml_fp32_to_fp16(7.0f); v_data[7] = ggml_fp32_to_fp16(8.0f);
    printf("V0 = [%.1f, %.1f, %.1f, %.1f]\n", 
           GGML_FP16_TO_FP32(v_data[0]), GGML_FP16_TO_FP32(v_data[1]), 
           GGML_FP16_TO_FP32(v_data[2]), GGML_FP16_TO_FP32(v_data[3]));
    printf("V1 = [%.1f, %.1f, %.1f, %.1f]\n", 
           GGML_FP16_TO_FP32(v_data[4]), GGML_FP16_TO_FP32(v_data[5]), 
           GGML_FP16_TO_FP32(v_data[6]), GGML_FP16_TO_FP32(v_data[7]));

    // No mask (all zeros)
    memset(mask->data, 0, ggml_nbytes(mask));

    // Initialize state
    float* state_data = (float*)state->data;
    state_data[0] = -INFINITY; // M
    state_data[1] = 0.0f;      // S

    printf("\nManual calculation:\n");
    printf("QK0 = Q·K0 = 1.0 * 1.0 = 1.0\n");
    printf("QK1 = Q·K1 = 1.0 * 0.0 = 0.0\n");
    printf("Scale = 1/sqrt(%d) = %.6f\n", head_dim, 1.0f/sqrt(head_dim));
    printf("Scaled: QK0_scaled = %.6f, QK1_scaled = %.6f\n", 
           1.0f/sqrt(head_dim), 0.0f/sqrt(head_dim));
    
    float qk0_scaled = 1.0f/sqrt(head_dim);
    float qk1_scaled = 0.0f/sqrt(head_dim);
    float softmax_exp0 = expf(qk0_scaled);
    float softmax_exp1 = expf(qk1_scaled);
    float softmax_sum = softmax_exp0 + softmax_exp1;
    float softmax_0 = softmax_exp0 / softmax_sum;
    float softmax_1 = softmax_exp1 / softmax_sum;
    
    printf("Softmax: exp(%.6f)=%.6f, exp(%.6f)=%.6f\n", 
           qk0_scaled, softmax_exp0, qk1_scaled, softmax_exp1);
    printf("Softmax sum = %.6f\n", softmax_sum);
    printf("Softmax weights: w0=%.6f, w1=%.6f\n", softmax_0, softmax_1);
    
    printf("Expected result = w0*V0 + w1*V1 = %.6f*[1,2,3,4] + %.6f*[5,6,7,8]\n", 
           softmax_0, softmax_1);
    printf("                = [%.6f, %.6f, %.6f, %.6f]\n", 
           softmax_0*1.0f + softmax_1*5.0f,
           softmax_0*2.0f + softmax_1*6.0f,
           softmax_0*3.0f + softmax_1*7.0f,
           softmax_0*4.0f + softmax_1*8.0f);

    // Test 1: Standard Flash Attention
    printf("\n--- Test 1: Standard Flash Attention ---\n");
    
    ggml_tensor * result_standard = ggml_flash_attn_ext(
        ctx, q, k, v, mask,
        1.0f / std::sqrt(head_dim),  // scale
        0.0f,  // max_bias
        0.0f   // logit_softcap
    );
    ggml_flash_attn_ext_set_prec(result_standard, GGML_PREC_F32);

    struct ggml_cgraph * graph_standard = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_standard, result_standard);

    enum ggml_status status_standard = ggml_graph_compute_with_ctx(ctx, graph_standard, n_threads);
    if (status_standard != GGML_STATUS_SUCCESS) {
        printf("ERROR: Standard flash attention failed\n");
        return 1;
    }

    float* result_std_data = (float*)result_standard->data;
    printf("Standard result: [%.6f, %.6f, %.6f, %.6f]\n", 
           result_std_data[0], result_std_data[1], result_std_data[2], result_std_data[3]);

    // Test 2: Flash Attention with State (Full KV)
    printf("\n--- Test 2: Flash Attention with State (Full KV) ---\n");
    
    // Reset state
    state_data[0] = -INFINITY; // M
    state_data[1] = 0.0f;      // S
    
    ggml_tensor * result_state_full = ggml_flash_attn_ext_with_state(
        ctx, q, k, v, mask, state,
        1.0f / std::sqrt(head_dim), 0.0f, 0.0f
    );
    ggml_flash_attn_ext_set_prec(result_state_full, GGML_PREC_F32);
    
    struct ggml_cgraph * graph_state_full = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_state_full, result_state_full);
    ggml_graph_compute_with_ctx(ctx, graph_state_full, n_threads);
    
    float* result_state_full_data = (float*)result_state_full->data;
    printf("State result (full - unnormalized): [%.6f, %.6f, %.6f, %.6f]\n", 
           result_state_full_data[0], result_state_full_data[1], 
           result_state_full_data[2], result_state_full_data[3]);
    printf("Final state: M=%.6f, S=%.6f\n", state_data[0], state_data[1]);
    
    // Normalize the result for comparison
    float final_S_full = state_data[1];
    for (int i = 0; i < head_dim; i++) {
        result_state_full_data[i] /= final_S_full;
    }
    printf("State result (full - normalized): [%.6f, %.6f, %.6f, %.6f]\n", 
           result_state_full_data[0], result_state_full_data[1], 
           result_state_full_data[2], result_state_full_data[3]);

    // Test 3: Segmented Processing
    printf("\n--- Test 3: Segmented Processing ---\n");
    
    // Reset state tensor for segmented test
    memset(state->data, 0, ggml_nbytes(state));
    for (int i = 0; i < n_heads * seq_len; i++) {
        ((float*)state->data)[i * 2 + 0] = -INFINITY; // M values
        ((float*)state->data)[i * 2 + 1] = 0.0f;      // S values  
    }
    
    int kv_segments = 2;
    int kv_segment_len = kv_len / kv_segments;
    printf("Processing %d segments of KV cache (segment_len=%d)...\n", kv_segments, kv_segment_len);

    ggml_tensor * result_segmented = nullptr;
    
    for (int seg = 0; seg < kv_segments; seg++) {
        printf("\nSegment %d: Processing K%d/V%d\n", seg + 1, seg, seg);
        
        // Print state before this segment
        float* state_data = (float*)state->data;
        printf("State before: M=%.6f, S=%.6f\n", 
               state_data[0 * 2 + 0], state_data[0 * 2 + 1]);

        // Create views of K and V for this segment using ggml_view_4d
        ggml_tensor * k_segment = ggml_view_4d(ctx, k, 
            head_dim, kv_segment_len, n_kv_heads, 1,  // ne
            k->nb[1], k->nb[2], k->nb[3],             // nb (strides)
            seg * kv_segment_len * k->nb[1]);         // offset

        ggml_tensor * v_segment = ggml_view_4d(ctx, v,
            head_dim, kv_segment_len, n_kv_heads, 1,  // ne
            v->nb[1], v->nb[2], v->nb[3],             // nb (strides)
            seg * kv_segment_len * v->nb[1]);         // offset

        // Create mask for this segment
        int padded_seq_len = ((seq_len + 64 - 1) / 64) * 64; // Pad to GGML_KQ_MASK_PAD = 64
        ggml_tensor * mask_segment = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 
            kv_segment_len, padded_seq_len);
        
        // Fill segment mask (no masking for simplicity)
        ggml_fp16_t* mask_seg_data = (ggml_fp16_t*)mask_segment->data;
        memset(mask_seg_data, 0, ggml_nbytes(mask_segment));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_segment_len; j++) {
                mask_seg_data[i * kv_segment_len + j] = ggml_fp32_to_fp16(0.0f);
            }
        }

        // Call flash attention with state for this segment
        result_segmented = ggml_flash_attn_ext_with_state(ctx, q, k_segment, v_segment, mask_segment, state,
                                                          1.0f / std::sqrt(head_dim), 0.0f, 0.0f);

        // Set precision
        ggml_flash_attn_ext_set_prec(result_segmented, GGML_PREC_F32);

        // Compute the result
        struct ggml_cgraph * graph_seg = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph_seg, result_segmented);
        ggml_graph_compute_with_ctx(ctx, graph_seg, n_threads);

        // Print state after this segment
        printf("State after seg%d: M=%.6f, S=%.6f\n", 
               seg + 1, state_data[0 * 2 + 0], state_data[0 * 2 + 1]);
        
        // Print segment result (should be unnormalized except for last segment)
        printf("Segment %d result: [%.6f, %.6f, %.6f, %.6f]\n", 
               seg + 1, ((float*)result_segmented->data)[0], ((float*)result_segmented->data)[1], 
               ((float*)result_segmented->data)[2], ((float*)result_segmented->data)[3]);
    }
    
    // IMPORTANT: Final normalization step
    // The result from the last segment contains unnormalized values, we need to normalize by final S
    float final_S = ((float*)state->data)[0 * 2 + 1];
    printf("\nFinal normalization with S=%.6f\n", final_S);
    
    float* result_data = (float*)result_segmented->data;
    for (int i = 0; i < head_dim; i++) {
        result_data[i] /= final_S;
    }
    
    printf("Normalized segmented result: [%.6f, %.6f, %.6f, %.6f]\n",
           result_data[0], result_data[1], result_data[2], result_data[3]);

    // Compare results
    printf("\n--- Results Comparison ---\n");
    printf("Expected:       [%.6f, %.6f, %.6f, %.6f]\n", 
           softmax_0*1.0f + softmax_1*5.0f,
           softmax_0*2.0f + softmax_1*6.0f,
           softmax_0*3.0f + softmax_1*7.0f,
           softmax_0*4.0f + softmax_1*8.0f);
    printf("Standard:       [%.6f, %.6f, %.6f, %.6f]\n", 
           result_std_data[0], result_std_data[1], result_std_data[2], result_std_data[3]);
    printf("State (full):   [%.6f, %.6f, %.6f, %.6f]\n", 
           result_state_full_data[0], result_state_full_data[1], 
           result_state_full_data[2], result_state_full_data[3]);
    printf("Segmented:      [%.6f, %.6f, %.6f, %.6f]\n", 
           result_data[0], result_data[1], result_data[2], result_data[3]);

    float max_diff_standard = 0.0f;
    float max_diff_segmented = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        max_diff_standard = std::max(max_diff_standard, 
                                   std::abs(result_std_data[i] - result_state_full_data[i]));
        max_diff_segmented = std::max(max_diff_segmented, 
                                    std::abs(result_std_data[i] - result_data[i]));
    }
    
    printf("\nMax differences:\n");
    printf("Standard vs State(full): %.2e\n", max_diff_standard);
    printf("Standard vs Segmented:   %.2e\n", max_diff_segmented);

    // Cleanup
    ggml_free(ctx);
    
    if (max_diff_standard < 1e-5 && max_diff_segmented < 1e-5) {
        printf("\n✅ SUCCESS: All results match!\n");
        return 0;
    } else {
        printf("\n❌ FAIL: Results don't match!\n");
        return 1;
    }
}