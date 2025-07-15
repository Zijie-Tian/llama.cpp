#include <iostream>
#include <vector>
#include <chrono>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#ifdef GGML_USE_TMAC
#include "tmac.h"
#include "lut_mul_mat.h"
#endif

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
    
//     // Test with different quantization types
//     struct {
//         ggml_type type;
//         const char* name;
//     } test_types[] = {
//         {GGML_TYPE_Q4_0,    "Q4_0"},
//         {GGML_TYPE_F32,     "F32"},
//         {GGML_TYPE_F16,     "F16"},
// #ifdef GGML_USE_TMAC
//         {GGML_TYPE_TMAC_W4G64_0,    "TMAC_W4G64_0"},
//         {GGML_TYPE_TMAC_W4G128_0,   "TMAC_W4G128_0"},
//         {GGML_TYPE_TMAC_W2G64_0,    "TMAC_W2G64_0"},
// #endif
//     };
    
//     for (size_t i = 0; i < sizeof(test_types) / sizeof(test_types[0]); i++) {
//         ggml_type type = test_types[i].type;
//         const char* name = test_types[i].name;
        
//         bool is_supported = is_type_supported(type);
//         printf("  %s: %s\n", name, is_supported ? "✓ supported" : "✗ not supported");
        
//         if (is_supported) {
//             // Get type info
//             size_t type_size    = ggml_type_size(type);
//             size_t block_size   = ggml_blck_size(type);
//             printf("    - Type size: %zu bytes\n", type_size);
//             printf("    - Block size: %zu elements\n", block_size);
//         }
//     }

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
        .mem_size   = 1024 * 1024,  // 1MB
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
    //> Create Tensors
    //> ===================================================================================================

    // Create a simple 2D tensor
    const int M = 64, N = 64;
    ggml_tensor* tensor = ggml_new_tensor_2d(tmac_ctx, GGML_TYPE_TMAC_W4G64_0, N, M);

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

    // Test buffer allocation for tensor
    size_t alloc_size = ggml_backend_buft_get_alloc_size(tmac_buft, tensor);
    printf("  - Required allocation size: %zu bytes\n", alloc_size);

    // // Create a dummy mul_mat operation to test T-MAC capability
    // printf("\n=== T-MAC Capability Test ===\n");
    
    // ggml_tensor* input = ggml_new_tensor_2d(tmac_ctx, GGML_TYPE_F32, N, 1);
    // ggml_tensor* mul_op = ggml_mul_mat(tmac_ctx, tensor, input);
    
    // if (mul_op) {
    //     bool can_handle = ggml_tmac_can_mul_mat(mul_op);
    //     printf("✓ Mul_mat operation created\n");
    //     printf("  - T-MAC can handle: %s\n", can_handle ? "YES" : "NO");
        
    //     if (!can_handle) {
    //         printf("  - Note: This is expected for tensors without specific names\n");
    //         printf("  - T-MAC typically targets model weights like 'blk.0.attn_k.weight'\n");
    //     }
    // } else {
    //     printf("✗ Failed to create mul_mat operation\n");
    // }

    // // Clean up
    // ggml_free(tmac_ctx);
    // ggml_backend_buffer_free(test_buf);

    // printf("\n=== Test Summary ===\n");
    // printf("✓ T-MAC initialization: SUCCESS\n");
    // printf("✓ Buffer type creation: SUCCESS\n");
    // printf("✓ Buffer allocation: SUCCESS\n");
    // printf("✓ Tensor creation: SUCCESS\n");
    // printf("✓ Type support check: SUCCESS\n");
    // printf("✓ Capability check: SUCCESS\n");

    // printf("\n=== Simple T-MAC Test Completed Successfully ===\n");



    return 0;
} 