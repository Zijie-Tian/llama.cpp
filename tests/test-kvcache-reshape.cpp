#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

int main() {
    printf("Testing KV cache tensor reshape with GGML\n");
    
    // Initialize GGML
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize GGML context\n");
        return 1;
    }
    
    // Create a KV cache-like tensor
    // Typical KV cache dimensions: [n_embd, n_head, n_ctx, n_batch]
    // Let's use smaller dimensions for testing
    const int64_t n_embd = 128;   // embedding dimension
    const int64_t n_head = 8;     // number of heads
    const int64_t n_ctx = 64;     // context length
    const int64_t n_batch = 4;    // batch size
    
    printf("Creating KV cache tensor with shape [%lld, %lld, %lld, %lld]\n", 
           n_embd, n_head, n_ctx, n_batch);
    
    // Create the original tensor
    struct ggml_tensor * kv_cache = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                                        n_embd, n_head, n_ctx, n_batch);
    
    // Fill with some test data
    float * data = (float *)kv_cache->data;
    for (int i = 0; i < ggml_nelements(kv_cache); i++) {
        data[i] = (float)(i % 100) / 100.0f;
    }
    
    printf("Original tensor shape: [%lld, %lld, %lld, %lld]\n", 
           kv_cache->ne[0], kv_cache->ne[1], kv_cache->ne[2], kv_cache->ne[3]);
    printf("Total elements: %lld\n", ggml_nelements(kv_cache));
    
    // Test reshape operation 1: Merge head and embedding dimensions
    // New shape: [n_embd * n_head, n_ctx, n_batch]
    struct ggml_tensor * reshaped1 = ggml_reshape_3d(ctx, kv_cache, 
                                                      n_embd * n_head, n_ctx, n_batch);
    
    printf("\nReshape 1 - Merged head and embedding dimensions:\n");
    printf("New shape: [%lld, %lld, %lld]\n", 
           reshaped1->ne[0], reshaped1->ne[1], reshaped1->ne[2]);
    printf("Total elements: %lld\n", ggml_nelements(reshaped1));
    
    // Verify the data is still accessible correctly
    float * reshaped1_data = (float *)reshaped1->data;
    bool reshape1_valid = true;
    for (int i = 0; i < 10; i++) {
        if (data[i] != reshaped1_data[i]) {
            reshape1_valid = false;
            break;
        }
    }
    printf("Reshape 1 data integrity: %s\n", reshape1_valid ? "PASS" : "FAIL");
    
    // Test reshape operation 2: Flatten to 2D
    // New shape: [n_embd * n_head, n_ctx * n_batch]
    struct ggml_tensor * reshaped2 = ggml_reshape_2d(ctx, kv_cache, 
                                                      n_embd * n_head, n_ctx * n_batch);
    
    printf("\nReshape 2 - Flattened to 2D:\n");
    printf("New shape: [%lld, %lld]\n", reshaped2->ne[0], reshaped2->ne[1]);
    printf("Total elements: %lld\n", ggml_nelements(reshaped2));
    
    // Verify the data
    float * reshaped2_data = (float *)reshaped2->data;
    bool reshape2_valid = true;
    for (int i = 0; i < 10; i++) {
        if (data[i] != reshaped2_data[i]) {
            reshape2_valid = false;
            break;
        }
    }
    printf("Reshape 2 data integrity: %s\n", reshape2_valid ? "PASS" : "FAIL");
    
    // Test reshape operation 3: Complete flatten to 1D
    struct ggml_tensor * reshaped3 = ggml_reshape_1d(ctx, kv_cache, 
                                                      n_embd * n_head * n_ctx * n_batch);
    
    printf("\nReshape 3 - Flattened to 1D:\n");
    printf("New shape: [%lld]\n", reshaped3->ne[0]);
    printf("Total elements: %lld\n", ggml_nelements(reshaped3));
    
    // Verify the data
    float * reshaped3_data = (float *)reshaped3->data;
    bool reshape3_valid = true;
    for (int i = 0; i < 10; i++) {
        if (data[i] != reshaped3_data[i]) {
            reshape3_valid = false;
            break;
        }
    }
    printf("Reshape 3 data integrity: %s\n", reshape3_valid ? "PASS" : "FAIL");
    
    // Test reshape back to original dimensions
    struct ggml_tensor * reshaped_back = ggml_reshape_4d(ctx, reshaped3,
                                                          n_embd, n_head, n_ctx, n_batch);
    
    printf("\nReshape back to original 4D:\n");
    printf("New shape: [%lld, %lld, %lld, %lld]\n", 
           reshaped_back->ne[0], reshaped_back->ne[1], 
           reshaped_back->ne[2], reshaped_back->ne[3]);
    
    // Verify dimensions match original
    bool dims_match = (reshaped_back->ne[0] == kv_cache->ne[0] &&
                       reshaped_back->ne[1] == kv_cache->ne[1] &&
                       reshaped_back->ne[2] == kv_cache->ne[2] &&
                       reshaped_back->ne[3] == kv_cache->ne[3]);
    
    printf("Dimensions match original: %s\n", dims_match ? "PASS" : "FAIL");
    
    // Clean up
    ggml_free(ctx);
    
    // Summary
    printf("\n=== Test Summary ===\n");
    printf("All tests %s\n", 
           (reshape1_valid && reshape2_valid && reshape3_valid && dims_match) ? "PASSED" : "FAILED");
    
    return 0;
}