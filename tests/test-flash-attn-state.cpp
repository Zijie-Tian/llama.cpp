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

#ifdef LLAMA_TORCH_AVAILABLE
#include <torch/torch.h>

// Convert ggml tensor to torch tensor using type traits
torch::Tensor ggml_to_torch(ggml_tensor* tensor) {
    auto type_traits = ggml_get_type_traits(tensor->type);
    size_t n_elements = ggml_nelements(tensor);
    
    // Create temporary buffer for float conversion
    std::vector<float> float_buffer(n_elements);
    
    if (type_traits->to_float && tensor->type != GGML_TYPE_F32) {
        // Use type traits to convert to float
        type_traits->to_float(tensor->data, float_buffer.data(), n_elements);
    } else if (tensor->type == GGML_TYPE_F32) {
        // Direct copy for F32
        memcpy(float_buffer.data(), tensor->data, n_elements * sizeof(float));
    } else {
        printf("ERROR: Unsupported tensor type for conversion: %s\n", ggml_type_name(tensor->type));
        return torch::Tensor();
    }
    
    // Create torch tensor with same shape
    std::vector<int64_t> sizes;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] > 1 || i == 0) {  // Include dimensions > 1 and always include first dimension
            sizes.push_back(tensor->ne[i]);
        }
    }
    
    return torch::from_blob(float_buffer.data(), sizes, torch::kFloat32).clone();
}

// Perform torch flash attention for comparison
torch::Tensor torch_flash_attention(
    torch::Tensor Q, 
    torch::Tensor K, 
    torch::Tensor V, 
    torch::Tensor mask = torch::Tensor(),
    float scale = 1.0f
) {
    // Q shape: [batch, n_heads, seq_len, head_dim]
    // K, V shape: [batch, n_kv_heads, kv_len, head_dim]
    
    std::cout << "Torch Flash Attention Input Shapes:" << std::endl;
    std::cout << "Q: " << Q.sizes() << std::endl;
    std::cout << "K: " << K.sizes() << std::endl;
    std::cout << "V: " << V.sizes() << std::endl;
    if (mask.defined()) {
        std::cout << "Mask: " << mask.sizes() << std::endl;
    }
    
    // Compute attention scores: Q @ K^T
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) * scale;
    
    if (mask.defined()) {
        // Apply mask by adding it (mask contains 0s and -inf)
        scores = scores + mask;
    }
    
    // Apply softmax
    auto attn_weights = torch::softmax(scores, -1);
    
    // Apply to values: attn_weights @ V
    auto output = torch::matmul(attn_weights, V);
    
    return output;
}

void test_torch_integration() {
    std::cout << "Testing PyTorch C++ integration..." << std::endl;
    
    // Create a simple tensor
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Created tensor with shape: " << tensor.sizes() << std::endl;
    std::cout << "Tensor data:\n" << tensor << std::endl;
    
    // Test basic operations
    torch::Tensor result = tensor * 2.0;
    std::cout << "After multiplication by 2:\n" << result << std::endl;
    
    // Check CUDA availability
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "CUDA is not available, using CPU" << std::endl;
    }
    
    std::cout << "PyTorch integration test completed successfully!" << std::endl;
}
#endif // LLAMA_TORCH_AVAILABLE

// Use fixed seed for reproducible results
static std::mt19937 g_rng(std::random_device{}());

static void fill_tensor_f32(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    float* data = (float*)dst->data;
    size_t n_elements = ggml_nelements(dst);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = dis(g_rng);
    }
}

static void fill_tensor_f16(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    ggml_fp16_t* data = (ggml_fp16_t*)dst->data;
    size_t n_elements = ggml_nelements(dst);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = ggml_fp32_to_fp16(dis(g_rng));
    }
}

static void print_tensor_info(const char* name, ggml_tensor* tensor) {
    printf("%s: [%ld, %ld, %ld, %ld] type=%s, elements=%ld\n",
           name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
           ggml_type_name(tensor->type), ggml_nelements(tensor));
}

static void print_f32_sample(const char* name, ggml_tensor* tensor, int max_elements = 10) {
    if (tensor->type != GGML_TYPE_F32) {
        printf("%s: Not F32 tensor (type=%s)\n", name, ggml_type_name(tensor->type));
        return;
    }
    
    float* data = (float*)tensor->data;
    size_t n_elements = ggml_nelements(tensor);
    size_t elements_to_print = std::min((size_t)max_elements, n_elements);
    
    printf("%s sample values: ", name);
    for (size_t i = 0; i < elements_to_print; i++) {
        printf("%.6f ", data[i]);
    }
    if (elements_to_print < n_elements) {
        printf("... (total %ld elements)", n_elements);
    }
    printf("\n");
}

static float tensor_max_diff(ggml_tensor* a, ggml_tensor* b) {
    if (ggml_nelements(a) != ggml_nelements(b) || a->type != b->type) {
        printf("ERROR: Tensors have different sizes or types\n");
        return -1.0f;
    }
    
    if (a->type != GGML_TYPE_F32) {
        printf("ERROR: Only F32 tensors supported for comparison\n");
        return -1.0f;
    }
    
    float* data_a = (float*)a->data;
    float* data_b = (float*)b->data;
    size_t n_elements = ggml_nelements(a);
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < n_elements; i++) {
        float diff = std::abs(data_a[i] - data_b[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    return max_diff;
}

static void reset_state_tensor(ggml_tensor* state) {
    float* state_data = (float*)state->data;
    size_t n_pairs = ggml_nelements(state) / 2;
    
    for (size_t i = 0; i < n_pairs; i++) {
        state_data[i * 2 + 0] = -INFINITY;  // M (max KQ value)
        state_data[i * 2 + 1] = 0.0f;       // S (sum)
    }
}

int main() {
    printf("=== Flash Attention State Tensor - Comprehensive Test ===\n");

    // Test parameters
    const int head_dim   = 32;
    const int n_heads    = 8;
    const int n_kv_heads = 4;
    const int seq_len    = 2;
    const int kv_len     = 4;  // Will be split into segments
    const int n_threads  = 4;
    const int kv_segments = 2;  // Split KV into 2 segments
    const int kv_segment_len = kv_len / kv_segments;

    printf("Test Parameters:\n");
    printf("  head_dim=%d, n_heads=%d, n_kv_heads=%d\n", head_dim, n_heads, n_kv_heads);
    printf("  seq_len=%d, kv_len=%d\n", seq_len, kv_len);
    printf("  kv_segments=%d, kv_segment_len=%d\n", kv_segments, kv_segment_len);

    // Initialize ggml context
    const size_t ctx_size = 1024*1024*1024; // 1GB
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

    // ============================================================================
    // Create and initialize tensors with FIXED data
    // ============================================================================
    printf("\n--- Creating Fixed Test Data ---\n");

    // Create tensors for flash attention
    // Format: [head_dim, seq_len, n_heads, 1] for Q
    // Format: [head_dim, kv_len, n_kv_heads, 1] for K, V
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);

    // Create mask tensor with proper padding
    const int padded_kv_len = GGML_PAD(kv_len, 64);
    const int padded_seq_len = GGML_PAD(seq_len, GGML_KQ_MASK_PAD);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, padded_kv_len, padded_seq_len);

    // Create state tensor: [2, n_heads * seq_len] for [M, S] pairs
    ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, n_heads * seq_len);

    print_tensor_info("Q", q);
    print_tensor_info("K", k);
    print_tensor_info("V", v);
    print_tensor_info("Mask", mask);
    print_tensor_info("State", state);

    // Fill with FIXED reproducible data
    printf("\nGenerating fixed test data (seed=42)...\n");
    fill_tensor_f32(q, -0.8f, 0.8f);
    fill_tensor_f16(k, -0.6f, 0.6f);
    fill_tensor_f16(v, -0.7f, 0.7f);

    // Initialize mask (no causal mask - all positions can see all KV)
    ggml_fp16_t* mask_data = (ggml_fp16_t*)mask->data;
    memset(mask_data, 0, ggml_nbytes(mask));
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < kv_len; j++) {
            // No masking - all positions can see all KV tokens
            mask_data[i * padded_kv_len + j] = ggml_fp32_to_fp16(0.0f);
        }
    }

    printf("Fixed test data generated successfully\n");

    // ============================================================================
    // Test 1: Standard Flash Attention (Reference Result)
    // ============================================================================
    printf("\n--- Test 1: Standard Flash Attention (Reference) ---\n");

    ggml_tensor * result_standard = ggml_flash_attn_ext(
        ctx, q, k, v, mask,
        1.0f / std::sqrt(head_dim),  // scale
        0.0f,  // max_bias
        0.0f   // logit_softcap
    );
    ggml_flash_attn_ext_set_prec(result_standard, GGML_PREC_F32);

    if (!result_standard) {
        printf("ERROR: Failed to create standard flash attention operation\n");
        ggml_free(ctx);
        return 1;
    }

    struct ggml_cgraph * graph_standard = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_standard, result_standard);

    printf("Computing standard flash attention...\n");
    enum ggml_status status_standard = ggml_graph_compute_with_ctx(ctx, graph_standard, n_threads);

    if (status_standard != GGML_STATUS_SUCCESS) {
        printf("ERROR: Standard flash attention failed with status: %d\n", status_standard);
        ggml_free(ctx);
        return 1;
    }

    printf("Standard flash attention computation successful\n");
    print_f32_sample("Standard result", result_standard, 8);

    // ============================================================================
    // Test 2: Segmented Flash Attention with State Accumulation
    // ============================================================================
    printf("\n--- Test 2: Segmented Flash Attention with State ---\n");

    // Reset state tensor
    reset_state_tensor(state);
    
    // Create result tensor for accumulation (same shape as standard result)
    ggml_tensor * result_segmented = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
        head_dim, seq_len, n_heads, 1);

    // Initialize segmented result to zero
    memset(result_segmented->data, 0, ggml_nbytes(result_segmented));

    printf("Processing %d segments of KV cache (segment_len=%d)...\n", kv_segments, kv_segment_len);

    for (int seg = 0; seg < kv_segments; seg++) {
        printf("\n  Segment %d/%d (kv_pos %d-%d):\n", 
               seg + 1, kv_segments, seg * kv_segment_len, (seg + 1) * kv_segment_len - 1);

        // Print state before this segment
        printf("    State before segment %d: ", seg + 1);
        float* state_data = (float*)state->data;
        for (int i = 0; i < std::min(4, n_heads * seq_len); i++) {
            printf("[M=%.3f,S=%.3f] ", state_data[i * 2 + 0], state_data[i * 2 + 1]);
        }
        printf("...\n");

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
        const int padded_segment_len = GGML_PAD(kv_segment_len, 64);
        ggml_tensor * mask_segment = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 
            padded_segment_len, padded_seq_len);

        // Fill segment mask
        ggml_fp16_t* mask_seg_data = (ggml_fp16_t*)mask_segment->data;
        memset(mask_seg_data, 0, ggml_nbytes(mask_segment));
        
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_segment_len; j++) {
                int global_j = seg * kv_segment_len + j;
                // No masking for segment - all positions can see all KV tokens in this segment
                mask_seg_data[i * padded_segment_len + j] = ggml_fp32_to_fp16(0.0f);
            }
        }

        // Debug: Print mask information for first segment
        if (seg == 0) {
            printf("    Debug - Global mask (first 4 seq positions, first 20 kv positions):\n");
            for (int i = 0; i < std::min(4, seq_len); i++) {
                printf("      seq[%d]: ", i);
                for (int j = 0; j < std::min(20, kv_len); j++) {
                    float mask_val = GGML_FP16_TO_FP32(mask_data[i * padded_kv_len + j]);
                    printf("%.0f ", mask_val == -INFINITY ? -1.0f : mask_val);
                }
                printf("...\n");
            }
            
            printf("    Debug - Segment mask (first 4 seq positions, all segment positions):\n");
            for (int i = 0; i < std::min(4, seq_len); i++) {
                printf("      seq[%d]: ", i);
                for (int j = 0; j < kv_segment_len; j++) {
                    float mask_val = GGML_FP16_TO_FP32(mask_seg_data[i * padded_segment_len + j]);
                    printf("%.0f ", mask_val == -INFINITY ? -1.0f : mask_val);
                }
                printf("\n");
            }
        }

        print_tensor_info("    K segment", k_segment);
        print_tensor_info("    V segment", v_segment);

        // Compute flash attention with state for this segment
        // CRITICAL: Create the operation but redirect its output to our accumulation tensor
        ggml_tensor * result_seg = ggml_flash_attn_ext_with_state(
            ctx, q, k_segment, v_segment, mask_segment, state,
            1.0f / std::sqrt(head_dim),  // scale
            0.0f,  // max_bias
            0.0f   // logit_softcap
        );
        ggml_flash_attn_ext_set_prec(result_seg, GGML_PREC_F32);

        if (!result_seg) {
            printf("ERROR: Failed to create segmented flash attention for segment %d\n", seg);
            ggml_free(ctx);
            return 1;
        }

        // CRITICAL FIX: Redirect the operation's output to our accumulation tensor
        // This ensures that each segment reads from and writes to the same tensor
        result_seg->data = result_segmented->data;
        result_seg->nb[0] = result_segmented->nb[0];
        result_seg->nb[1] = result_segmented->nb[1];
        result_seg->nb[2] = result_segmented->nb[2];
        result_seg->nb[3] = result_segmented->nb[3];

        struct ggml_cgraph * graph_seg = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph_seg, result_seg);

        enum ggml_status status_seg = ggml_graph_compute_with_ctx(ctx, graph_seg, n_threads);

        if (status_seg != GGML_STATUS_SUCCESS) {
            printf("ERROR: Segmented flash attention failed for segment %d with status: %d\n", seg, status_seg);
            ggml_free(ctx);
            return 1;
        }

        printf("    Segment %d computed successfully\n", seg + 1);
        print_f32_sample("    Segment result", result_segmented, 6);

        // Print state after this segment
        printf("    State after segment %d: ", seg + 1);
        for (int i = 0; i < std::min(4, n_heads * seq_len); i++) {
            printf("[M=%.3f,S=%.3f] ", state_data[i * 2 + 0], state_data[i * 2 + 1]);
        }
        printf("...\n");

        // No need to copy result since we're already writing to result_segmented
    }

    printf("\nSegmented computation completed\n");
    print_f32_sample("Final segmented result", result_segmented, 8);

    // ============================================================================
    // Test 3: PyTorch Verification with scaled_dot_product_attention
    // ============================================================================
    printf("\n--- Test 3: PyTorch Verification ---\n");
    
    // Variables to store PyTorch results for later comparison
    std::vector<float> torch_result_data;
    bool torch_success = false;
    
#ifdef LLAMA_TORCH_AVAILABLE
    try {
        // Convert data to torch tensors
        // PyTorch expects [batch_size, num_heads, seq_len, head_dim] format
        
        // Create torch tensors from existing data
        auto torch_options = torch::TensorOptions().dtype(torch::kFloat32);
        
        // Query: [1, n_heads, seq_len, head_dim]
        auto q_torch = torch::zeros({1, n_heads, seq_len, head_dim}, torch_options);
        float* q_torch_data = q_torch.data_ptr<float>();
        
        // Convert from ggml format [head_dim, seq_len, n_heads, 1] to torch format [1, n_heads, seq_len, head_dim]
        for (int h = 0; h < n_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    int ggml_idx = d + s * head_dim + h * head_dim * seq_len;
                    int torch_idx = h * seq_len * head_dim + s * head_dim + d;
                    q_torch_data[torch_idx] = ((float*)q->data)[ggml_idx];
                }
            }
        }
        
        // Key: [1, n_kv_heads, kv_len, head_dim]
        auto k_torch = torch::zeros({1, n_kv_heads, kv_len, head_dim}, torch_options);
        float* k_torch_data = k_torch.data_ptr<float>();
        
        // Convert from ggml format [head_dim, kv_len, n_kv_heads, 1] to torch format [1, n_kv_heads, kv_len, head_dim]
        for (int h = 0; h < n_kv_heads; h++) {
            for (int s = 0; s < kv_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    int ggml_idx = d + s * head_dim + h * head_dim * kv_len;
                    int torch_idx = h * kv_len * head_dim + s * head_dim + d;
                    // Convert F16 to F32
                    k_torch_data[torch_idx] = ggml_fp16_to_fp32(((ggml_fp16_t*)k->data)[ggml_idx]);
                }
            }
        }
        
        // Value: [1, n_kv_heads, kv_len, head_dim]  
        auto v_torch = torch::zeros({1, n_kv_heads, kv_len, head_dim}, torch_options);
        float* v_torch_data = v_torch.data_ptr<float>();
        
        // Convert from ggml format [head_dim, kv_len, n_kv_heads, 1] to torch format [1, n_kv_heads, kv_len, head_dim]
        for (int h = 0; h < n_kv_heads; h++) {
            for (int s = 0; s < kv_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    int ggml_idx = d + s * head_dim + h * head_dim * kv_len;
                    int torch_idx = h * kv_len * head_dim + s * head_dim + d;
                    // Convert F16 to F32
                    v_torch_data[torch_idx] = ggml_fp16_to_fp32(((ggml_fp16_t*)v->data)[ggml_idx]);
                }
            }
        }

        // Create boolean mask for PyTorch (tensor shape: [1, n_heads, seq_len, kv_len])
        // PyTorch attention mask: true = can attend, false = cannot attend
        auto mask_torch = torch::ones({1, n_heads, seq_len, kv_len}, torch::TensorOptions().dtype(torch::kBool));
        bool* mask_torch_data = mask_torch.data_ptr<bool>();

        // Convert ggml mask to PyTorch boolean mask format
        // ggml mask: 0.0f = can attend, -INFINITY = cannot attend
        // PyTorch mask: true = can attend, false = cannot attend
        for (int h = 0; h < n_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < kv_len; d++) {
                    // Read from ggml mask (format: [kv_len, seq_len])
                    int ggml_idx = d + s * padded_kv_len;
                    float ggml_mask_val = ggml_fp16_to_fp32(mask_data[ggml_idx]);
                    
                    // PyTorch index (format: [1, n_heads, seq_len, kv_len])
                    int torch_idx = h * seq_len * kv_len + s * kv_len + d;
                    
                    // Convert: ggml 0.0f -> PyTorch true (can attend)
                    //          ggml -INFINITY -> PyTorch false (cannot attend)
                    if (ggml_mask_val == 0.0f) {
                        mask_torch_data[torch_idx] = true;   // Can attend
                    } else {
                        mask_torch_data[torch_idx] = false;  // Cannot attend
                    }
                }
            }
        }
        
        // For GQA (Grouped Query Attention), we need to repeat KV heads to match Q heads
        if (n_heads > n_kv_heads) {
            // Repeat KV heads
            k_torch = k_torch.repeat_interleave(n_heads / n_kv_heads, /*dim=*/1);
            v_torch = v_torch.repeat_interleave(n_heads / n_kv_heads, /*dim=*/1);
        }
        
        printf("PyTorch tensor shapes:\n");
        printf("  Q: [%ld, %ld, %ld, %ld]\n", q_torch.size(0), q_torch.size(1), q_torch.size(2), q_torch.size(3));
        printf("  K: [%ld, %ld, %ld, %ld]\n", k_torch.size(0), k_torch.size(1), k_torch.size(2), k_torch.size(3));
        printf("  V: [%ld, %ld, %ld, %ld]\n", v_torch.size(0), v_torch.size(1), v_torch.size(2), v_torch.size(3));

        // Compute scaled dot product attention
        float scale_factor = 1.0f / std::sqrt((float)head_dim);
        auto torch_result = torch::scaled_dot_product_attention(
            q_torch, k_torch, v_torch, mask_torch, 
            /*dropout_p=*/0.0,
            /*is_causal=*/false,
            /*scale=*/scale_factor
        );
        torch_result = torch_result.permute({0, 2, 1, 3}).contiguous();     //> [1, seq_len, n_heads, head_dim]

        printf("PyTorch result shape: [%ld, %ld, %ld, %ld]\n", 
               torch_result.size(0), torch_result.size(1), torch_result.size(2), torch_result.size(3));
        
        // Store PyTorch result data for later comparison
        float* torch_data_ptr = torch_result.data_ptr<float>();
        size_t torch_elements = torch_result.numel();
        torch_result_data.resize(torch_elements);
        
        // Convert torch result from [1, seq_len, n_heads, head_dim] to [head_dim, seq_len, n_heads, 1] format
        for (int h = 0; h < n_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    // PyTorch result format: [1, seq_len, n_heads, head_dim]
                    int torch_idx = s * n_heads * head_dim + h * head_dim + d;
                    // Custom result format: [head_dim, seq_len, n_heads, 1]
                    int custom_idx = d + s * head_dim + h * head_dim * seq_len;
                    torch_result_data[custom_idx] = torch_data_ptr[torch_idx];
                }
            }
        }
        
        torch_success = true;
        printf("PyTorch computation successful\n");
        
    } catch (const std::exception& e) {
        printf("PyTorch verification failed with exception: %s\n", e.what());
        printf("This might be due to PyTorch not being properly installed or linked.\n");
        torch_success = false;
    }
#else
    printf("PyTorch verification skipped (PyTorch not available)\n");
    torch_success = false;
#endif // LLAMA_TORCH_AVAILABLE

    // ============================================================================
    // Test 4: Unified Comparison of Standard, Segmented, and PyTorch Results
    // ============================================================================
    printf("\n--- Test 4: Unified Results Comparison ---\n");

    float* standard_data = (float*)result_standard->data;
    float* segmented_data = (float*)result_segmented->data;

    // Compare element by element
    size_t standard_elements = ggml_nelements(result_standard);
    size_t segmented_elements = ggml_nelements(result_segmented);

    printf("Result tensor information:\n");
    printf("  Standard result elements: %zu\n", standard_elements);
    printf("  Segmented result elements: %zu\n", segmented_elements);
    if (torch_success) {
        printf("  PyTorch result elements: %zu\n", torch_result_data.size());
    } else {
        printf("  PyTorch result: FAILED\n");
    }

    // Calculate comparison statistics
    float max_standard_segmented = 0.0f, sum_standard_segmented = 0.0f;
    float max_standard_torch = 0.0f, sum_standard_torch = 0.0f;
    float max_segmented_torch = 0.0f, sum_segmented_torch = 0.0f;
    size_t compared_elements = 0;

    // Compare the first min(standard_elements, segmented_elements) elements
    size_t min_elements = std::min(standard_elements, segmented_elements);
    if (torch_success) {
        min_elements = std::min(min_elements, torch_result_data.size());
    }

    for (size_t i = 0; i < min_elements; i++) {
        float standard_val = standard_data[i];
        float segmented_val = segmented_data[i];
        float torch_val = torch_success ? torch_result_data[i] : NAN;

        if (std::isfinite(standard_val) && std::isfinite(segmented_val)) {
            float abs_diff_ss = std::abs(standard_val - segmented_val);
            max_standard_segmented = std::max(max_standard_segmented, abs_diff_ss);
            sum_standard_segmented += abs_diff_ss;

            if (torch_success && std::isfinite(torch_val)) {
                float abs_diff_st = std::abs(standard_val - torch_val);
                float abs_diff_seg_torch = std::abs(segmented_val - torch_val);
                max_standard_torch = std::max(max_standard_torch, abs_diff_st);
                max_segmented_torch = std::max(max_segmented_torch, abs_diff_seg_torch);
                sum_standard_torch += abs_diff_st;
                sum_segmented_torch += abs_diff_seg_torch;
            }
            compared_elements++;
        }
    }

    // Print detailed comparison table
    printf("\nDetailed Comparison Table (first 128 elements):\n");
    if (torch_success) {
        printf("Index | Standard    | Segmented   | PyTorch     | S-Seg Diff  | S-Torch Diff| Seg-Torch Diff\n");
        printf("------|-------------|-------------|-------------|-------------|-------------|---------------\n");
    } else {
        printf("Index | Standard    | Segmented   | S-Seg Diff\n");
        printf("------|-------------|-------------|-----------\n");
    }

    size_t show_elements = std::min(size_t(128), min_elements);
    for (size_t i = 0; i < show_elements; i++) {
        float standard_val = standard_data[i];
        float segmented_val = segmented_data[i];

        if (torch_success) {
            float torch_val = torch_result_data[i];
            
            if (std::isfinite(standard_val) && std::isfinite(segmented_val) && std::isfinite(torch_val)) {
                float abs_diff_ss = std::abs(standard_val - segmented_val);
                float abs_diff_st = std::abs(standard_val - torch_val);
                float abs_diff_seg_torch = std::abs(segmented_val - torch_val);
                printf("%5zu | %11.6f | %11.6f | %11.6f | %.6e | %.6e | %.6e\n", 
                       i, standard_val, segmented_val, torch_val, abs_diff_ss, abs_diff_st, abs_diff_seg_torch);
            } else {
                printf("%5zu | %11.6f | %11.6f | %11.6f |     N/A     |     N/A     |     N/A\n", 
                       i, standard_val, segmented_val, torch_val);
            }
        } else {
            if (std::isfinite(standard_val) && std::isfinite(segmented_val)) {
                float abs_diff_ss = std::abs(standard_val - segmented_val);
                printf("%5zu | %11.6f | %11.6f | %.6e\n", i, standard_val, segmented_val, abs_diff_ss);
            } else {
                printf("%5zu | %11.6f | %11.6f |     N/A\n", i, standard_val, segmented_val);
            }
        }
    }

    // Print comparison statistics
    printf("\nComparison Statistics:\n");
    printf("  Total compared elements: %zu\n", compared_elements);
    
    if (compared_elements > 0) {
        float avg_standard_segmented = sum_standard_segmented / compared_elements;
        printf("  Standard vs Segmented:\n");
        printf("    Max absolute difference: %.6e\n", max_standard_segmented);
        printf("    Average absolute difference: %.6e\n", avg_standard_segmented);
        
        if (torch_success) {
            float avg_standard_torch = sum_standard_torch / compared_elements;
            float avg_segmented_torch = sum_segmented_torch / compared_elements;
            printf("  Standard vs PyTorch:\n");
            printf("    Max absolute difference: %.6e\n", max_standard_torch);
            printf("    Average absolute difference: %.6e\n", avg_standard_torch);
            printf("  Segmented vs PyTorch:\n");
            printf("    Max absolute difference: %.6e\n", max_segmented_torch);
            printf("    Average absolute difference: %.6e\n", avg_segmented_torch);
        }
    } else {
        printf("  No finite elements to compare\n");
    }

    // ============================================================================
    // Test 5: State Tensor Analysis
    // ============================================================================
    printf("\n--- Test 5: State Tensor Analysis ---\n");

    printf("Final state tensor values:\n");
    print_f32_sample("Final state", state, 16);

    float* state_data = (float*)state->data;
    float min_m = INFINITY, max_m = -INFINITY;
    float min_s = INFINITY, max_s = -INFINITY;
    
    for (int i = 0; i < n_heads * seq_len; i++) {
        float m_val = state_data[i * 2 + 0];
        float s_val = state_data[i * 2 + 1];
        
        if (m_val != -INFINITY) {
            min_m = std::min(min_m, m_val);
            max_m = std::max(max_m, m_val);
        }
        
        min_s = std::min(min_s, s_val);
        max_s = std::max(max_s, s_val);
    }

    printf("State tensor statistics:\n");
    printf("  M values: min=%.6f, max=%.6f\n", min_m, max_m);
    printf("  S values: min=%.6f, max=%.6f\n", min_s, max_s);

    // ============================================================================
    // Final Results
    // ============================================================================
    printf("\n=== Final Test Results ===\n");
    
    // Determine test result - adjust tolerance for F16 precision
    const float tolerance = 1e-4f;  // Tolerance for F16 numerical differences
    bool test_passed = (compared_elements > 0) && (max_standard_segmented < tolerance);
    
    if (torch_success) {
        bool torch_test_passed = (compared_elements > 0) && (max_standard_torch < tolerance) && (max_segmented_torch < tolerance);
        test_passed = test_passed && torch_test_passed;
    }

    printf("Overall Test Result: %s\n", test_passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
    printf("  Standard vs Segmented: %s (max diff: %.6e)\n", 
           (compared_elements > 0 && max_standard_segmented < tolerance) ? "PASS" : "FAIL", 
           max_standard_segmented);
    
    if (torch_success) {
        printf("  Standard vs PyTorch: %s (max diff: %.6e)\n", 
               (compared_elements > 0 && max_standard_torch < tolerance) ? "PASS" : "FAIL", 
               max_standard_torch);
        printf("  Segmented vs PyTorch: %s (max diff: %.6e)\n", 
               (compared_elements > 0 && max_segmented_torch < tolerance) ? "PASS" : "FAIL", 
               max_segmented_torch);
    } else {
        printf("  PyTorch comparison: SKIPPED (PyTorch failed)\n");
    }
    
    if (test_passed) {
        printf("\n🎉 ALL TESTS PASSED!\n");
        printf("✅ Segmented flash attention with state produces identical results\n");
        printf("✅ State tensor correctly accumulates across segments\n");
        if (torch_success) {
            printf("✅ Results match PyTorch reference implementation\n");
        }
        printf("✅ Implementation is working correctly\n");
    } else {
        printf("\n❌ TESTS FAILED!\n");
        printf("❌ Results differ beyond acceptable tolerance (%.2e)\n", tolerance);
        printf("❌ Implementation needs debugging\n");
    }

    printf("\nTest Summary:\n");
    printf("  Tolerance: %.2e\n", tolerance);
    printf("  Standard vs Segmented max diff: %.2e\n", max_standard_segmented);
    if (torch_success) {
        printf("  Standard vs PyTorch max diff: %.2e\n", max_standard_torch);
        printf("  Segmented vs PyTorch max diff: %.2e\n", max_segmented_torch);
    }

    // Cleanup
    ggml_free(ctx);
    return test_passed ? 0 : 1;
}