#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <random>
#include <chrono>

#include "ggml.h"
#include "ggml-backend.h"
#include "qlutattn/pack_weights.h"

//> ===================================================================================================
//> Test utilities
//> ===================================================================================================

static void generate_random_data(uint8_t* data, size_t size, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(rng);
    }
}

static void generate_random_floats(float* data, size_t size, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(rng);
    }
}

static bool compare_buffers(const uint8_t* a, const uint8_t* b, size_t size, float tolerance = 0.0f) {
    for (size_t i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            printf("Mismatch at index %zu: %u != %u\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static bool compare_fp16_buffers(const ggml_fp16_t* a, const ggml_fp16_t* b, size_t size, float tolerance = 1e-3f) {
    for (size_t i = 0; i < size; i++) {
        float fa = GGML_FP16_TO_FP32(a[i]);
        float fb = GGML_FP16_TO_FP32(b[i]);
        if (fabsf(fa - fb) > tolerance) {
            printf("Mismatch at index %zu: %.6f != %.6f (diff: %.6f)\n", 
                   i, fa, fb, fabsf(fa - fb));
            return false;
        }
    }
    return true;
}

//> ===================================================================================================
//> Test cases
//> ===================================================================================================

static bool test_pack_config_init() {
    printf("Testing pack_config_init...\n");
    
    struct pack_config cfg;
    
    // Test 1-bit configuration
    pack_config_init(&cfg, 1, 256, 256, 256, 16, false);
    assert(cfg.bits == 1);
    assert(cfg.g == 4);
    assert(cfg.bm == 256);
    assert(cfg.kfactor == 16);
    assert(cfg.nelem_per_byte == 8);
    assert(cfg.simd_width == 16);
    
    // Test 2-bit configuration
    pack_config_init(&cfg, 2, 512, 512, 512, 8, false);
    assert(cfg.bits == 2);
    assert(cfg.nelem_per_byte == 4);
    
    // Test 4-bit configuration
    pack_config_init(&cfg, 4, 1024, 1024, 1024, 16, false);
    assert(cfg.bits == 4);
    assert(cfg.nelem_per_byte == 2);
    
    printf("  ✓ Configuration initialization works correctly\n");
    return true;
}

static bool test_pack_dimensions_valid() {
    printf("Testing pack_dimensions_valid...\n");
    
    // Valid dimensions
    assert(pack_dimensions_valid(128, 128, 1, 4) == true);
    assert(pack_dimensions_valid(256, 256, 2, 4) == true);
    assert(pack_dimensions_valid(512, 512, 4, 4) == true);
    
    // Invalid dimensions - m not divisible by bits
    assert(pack_dimensions_valid(127, 128, 2, 4) == false);
    
    // Invalid dimensions - k not divisible by g
    assert(pack_dimensions_valid(128, 127, 1, 4) == false);
    
    // Invalid dimensions - not 128x128 aligned
    assert(pack_dimensions_valid(64, 64, 1, 4) == false);
    
    printf("  ✓ Dimension validation works correctly\n");
    return true;
}

static bool test_pack_weights_1bit() {
    printf("Testing 1-bit pack_weights...\n");
    
    const int m = 128;
    const int k = 128;
    const int bits = 1;
    const int g = 4;
    
    // Allocate buffers
    size_t src_size = (m * k) / 8;  // 1 bit per element
    size_t dst_size = (m * k) / g / 2;  // After packing
    size_t workspace_size = m * k / g;
    
    uint8_t* src = (uint8_t*)malloc(src_size);
    uint8_t* dst = (uint8_t*)malloc(dst_size);
    uint8_t* workspace = (uint8_t*)malloc(workspace_size);
    
    // Generate random data
    std::mt19937 rng(42);
    generate_random_data(src, src_size, rng);
    
    // Initialize config
    struct pack_config cfg;
    pack_config_init(&cfg, bits, m, k, 256, 16, false);
    
    // Pack weights
    pack_weights_optimized(src, dst, workspace, m, k, bits, g, &cfg);
    
    // Basic validation - check output is non-zero
    bool has_nonzero = false;
    for (size_t i = 0; i < dst_size; i++) {
        if (dst[i] != 0) {
            has_nonzero = true;
            break;
        }
    }
    assert(has_nonzero);
    
    free(src);
    free(dst);
    free(workspace);
    
    printf("  ✓ 1-bit packing works\n");
    return true;
}

static bool test_pack_weights_2bit() {
    printf("Testing 2-bit pack_weights...\n");
    
    const int m = 256;
    const int k = 256;
    const int bits = 2;
    const int g = 4;
    
    // Allocate buffers
    size_t src_size = (m * k) / 4;  // 2 bits per element
    size_t dst_size = (m * k) / g / 2;  // After packing
    size_t workspace_size = m * k / g * 2;
    
    uint8_t* src = (uint8_t*)malloc(src_size);
    uint8_t* dst = (uint8_t*)malloc(dst_size);
    uint8_t* workspace = (uint8_t*)malloc(workspace_size);
    
    // Generate random data
    std::mt19937 rng(42);
    generate_random_data(src, src_size, rng);
    
    // Initialize config
    struct pack_config cfg;
    pack_config_init(&cfg, bits, m, k, 512, 16, false);
    
    // Pack weights
    pack_weights_optimized(src, dst, workspace, m, k, bits, g, &cfg);
    
    // Basic validation
    bool has_nonzero = false;
    for (size_t i = 0; i < dst_size; i++) {
        if (dst[i] != 0) {
            has_nonzero = true;
            break;
        }
    }
    assert(has_nonzero);
    
    free(src);
    free(dst);
    free(workspace);
    
    printf("  ✓ 2-bit packing works\n");
    return true;
}

static bool test_pack_weights_4bit() {
    printf("Testing 4-bit pack_weights...\n");
    
    const int m = 512;
    const int k = 512;
    const int bits = 4;
    const int g = 4;
    
    // Allocate buffers
    size_t src_size = (m * k) / 2;  // 4 bits per element
    size_t dst_size = (m * k) / g / 2;  // After packing
    size_t workspace_size = m * k / g * 4;
    
    uint8_t* src = (uint8_t*)malloc(src_size);
    uint8_t* dst = (uint8_t*)malloc(dst_size);
    uint8_t* workspace = (uint8_t*)malloc(workspace_size);
    
    // Generate random data
    std::mt19937 rng(42);
    generate_random_data(src, src_size, rng);
    
    // Initialize config
    struct pack_config cfg;
    pack_config_init(&cfg, bits, m, k, 1024, 16, false);
    
    // Pack weights
    pack_weights_optimized(src, dst, workspace, m, k, bits, g, &cfg);
    
    // Basic validation
    bool has_nonzero = false;
    for (size_t i = 0; i < dst_size; i++) {
        if (dst[i] != 0) {
            has_nonzero = true;
            break;
        }
    }
    assert(has_nonzero);
    
    free(src);
    free(dst);
    free(workspace);
    
    printf("  ✓ 4-bit packing works\n");
    return true;
}

static bool test_pack_scales() {
    printf("Testing pack_scales...\n");
    
    const int m = 256;
    const int k = 256;
    const int bits = 2;
    const int group_size = 64;
    
    // Calculate sizes
    int num_groups = (m / bits) * (k / group_size);
    size_t scales_out_size = num_groups * 2;  // Scale + zero point
    
    // Allocate buffers
    float* scale_ptr = (float*)malloc(num_groups * sizeof(float));
    float* zero_ptr = (float*)malloc(num_groups * sizeof(float));
    ggml_fp16_t* scales_out = (ggml_fp16_t*)malloc(scales_out_size * sizeof(ggml_fp16_t));
    
    // Generate random scales
    std::mt19937 rng(42);
    generate_random_floats(scale_ptr, num_groups, rng);
    generate_random_floats(zero_ptr, num_groups, rng);
    
    // Initialize config
    struct pack_config cfg;
    pack_config_init(&cfg, bits, m, k, 512, 16, false);
    
    // Pack scales
    pack_scales_optimized(scale_ptr, zero_ptr, scales_out, m, k, bits,
                         group_size, num_groups, &cfg);
    
    // Basic validation - check some values are converted properly
    bool has_nonzero = false;
    for (size_t i = 0; i < scales_out_size; i++) {
        float val = GGML_FP16_TO_FP32(scales_out[i]);
        if (val != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    assert(has_nonzero);
    
    free(scale_ptr);
    free(zero_ptr);
    free(scales_out);
    
    printf("  ✓ Scale packing works\n");
    return true;
}

static bool test_memory_pool() {
    printf("Testing memory pool...\n");
    
    const size_t pool_size = 1024 * 1024;  // 1MB
    const size_t alignment = 64;
    
    uint8_t* buffer = (uint8_t*)aligned_alloc(alignment, pool_size);
    assert(buffer != NULL);
    
    struct pack_memory_pool pool;
    pack_memory_pool_init(&pool, buffer, pool_size, alignment);
    
    // Test allocation
    void* ptr1 = pack_memory_pool_alloc(&pool, 1024);
    assert(ptr1 != NULL);
    assert(((uintptr_t)ptr1 & (alignment - 1)) == 0);  // Check alignment
    
    void* ptr2 = pack_memory_pool_alloc(&pool, 2048);
    assert(ptr2 != NULL);
    assert(ptr2 > ptr1);
    
    // Test reset
    pack_memory_pool_reset(&pool);
    void* ptr3 = pack_memory_pool_alloc(&pool, 512);
    assert(ptr3 == ptr1);  // Should reuse same memory after reset
    
    // Test exhaustion
    pack_memory_pool_alloc(&pool, pool_size - 512);
    void* ptr4 = pack_memory_pool_alloc(&pool, 1024);
    assert(ptr4 == NULL);  // Should fail - pool exhausted
    
    free(buffer);
    
    printf("  ✓ Memory pool works correctly\n");
    return true;
}

//> ===================================================================================================
//> Main test runner
//> ===================================================================================================

int main(int argc, char** argv) {
    printf("=== Pack Weights Basic Functionality Tests ===\n\n");
    
    int passed = 0;
    int failed = 0;
    
    // Run tests
    if (test_pack_config_init()) passed++; else failed++;
    if (test_pack_dimensions_valid()) passed++; else failed++;
    if (test_pack_weights_1bit()) passed++; else failed++;
    if (test_pack_weights_2bit()) passed++; else failed++;
    if (test_pack_weights_4bit()) passed++; else failed++;
    if (test_pack_scales()) passed++; else failed++;
    if (test_memory_pool()) passed++; else failed++;
    
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);
    
    if (failed == 0) {
        printf("\n✅ ALL TESTS PASSED!\n");
        return 0;
    } else {
        printf("\n❌ SOME TESTS FAILED!\n");
        return 1;
    }
}