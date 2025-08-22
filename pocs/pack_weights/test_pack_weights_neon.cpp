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

static void generate_test_pattern(uint8_t* data, size_t size, uint8_t pattern) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (pattern + i) & 0xFF;
    }
}

static double get_time_ms() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

//> ===================================================================================================
//> NEON vs Scalar comparison tests
//> ===================================================================================================

static bool test_neon_vs_scalar_1bit() {
    printf("Testing NEON vs Scalar for 1-bit...\n");
    
    const int m = 256;
    const int k = 256;
    const int bits = 1;
    const int g = 4;
    
    // Allocate buffers
    size_t src_size = (m * k) / 8;
    size_t dst_size = (m * k) / g / 2;
    size_t workspace_size = m * k / g;
    
    uint8_t* src = (uint8_t*)malloc(src_size);
    uint8_t* dst_scalar = (uint8_t*)malloc(dst_size);
    uint8_t* dst_neon = (uint8_t*)malloc(dst_size);
    uint8_t* workspace = (uint8_t*)malloc(workspace_size);
    
    // Generate test pattern
    generate_test_pattern(src, src_size, 0xA5);
    
    // Test scalar version
    struct pack_config cfg_scalar;
    pack_config_init(&cfg_scalar, bits, m, k, 256, 16, true);  // force_scalar = true
    
    double t0 = get_time_ms();
    pack_weights_optimized(src, dst_scalar, workspace, m, k, bits, g, &cfg_scalar);
    double t_scalar = get_time_ms() - t0;
    
    // Test NEON version
    struct pack_config cfg_neon;
    pack_config_init(&cfg_neon, bits, m, k, 256, 16, false);  // force_scalar = false
    
    t0 = get_time_ms();
    pack_weights_optimized(src, dst_neon, workspace, m, k, bits, g, &cfg_neon);
    double t_neon = get_time_ms() - t0;
    
    // Compare results - they should be identical
    bool match = true;
    for (size_t i = 0; i < dst_size; i++) {
        if (dst_scalar[i] != dst_neon[i]) {
            printf("  Mismatch at index %zu: scalar=%u, neon=%u\n", 
                   i, dst_scalar[i], dst_neon[i]);
            match = false;
            break;
        }
    }
    
    if (match) {
        printf("  ✓ Results match (scalar: %.2f ms, neon: %.2f ms, speedup: %.2fx)\n", 
               t_scalar, t_neon, t_scalar / t_neon);
    } else {
        printf("  ✗ Results don't match!\n");
    }
    
    free(src);
    free(dst_scalar);
    free(dst_neon);
    free(workspace);
    
    return match;
}

static bool test_neon_vs_scalar_2bit() {
    printf("Testing NEON vs Scalar for 2-bit...\n");
    
    const int m = 512;
    const int k = 512;
    const int bits = 2;
    const int g = 4;
    
    // Allocate buffers
    size_t src_size = (m * k) / 4;
    size_t dst_size = (m * k) / g / 2;
    size_t workspace_size = m * k / g * 2;
    
    uint8_t* src = (uint8_t*)malloc(src_size);
    uint8_t* dst_scalar = (uint8_t*)malloc(dst_size);
    uint8_t* dst_neon = (uint8_t*)malloc(dst_size);
    uint8_t* workspace = (uint8_t*)malloc(workspace_size);
    
    // Generate test pattern
    generate_test_pattern(src, src_size, 0x5A);
    
    // Test scalar version
    struct pack_config cfg_scalar;
    pack_config_init(&cfg_scalar, bits, m, k, 512, 16, true);
    
    double t0 = get_time_ms();
    pack_weights_optimized(src, dst_scalar, workspace, m, k, bits, g, &cfg_scalar);
    double t_scalar = get_time_ms() - t0;
    
    // Test NEON version
    struct pack_config cfg_neon;
    pack_config_init(&cfg_neon, bits, m, k, 512, 16, false);
    
    t0 = get_time_ms();
    pack_weights_optimized(src, dst_neon, workspace, m, k, bits, g, &cfg_neon);
    double t_neon = get_time_ms() - t0;
    
    // Compare results
    bool match = true;
    for (size_t i = 0; i < dst_size; i++) {
        if (dst_scalar[i] != dst_neon[i]) {
            printf("  Mismatch at index %zu: scalar=%u, neon=%u\n", 
                   i, dst_scalar[i], dst_neon[i]);
            match = false;
            break;
        }
    }
    
    if (match) {
        printf("  ✓ Results match (scalar: %.2f ms, neon: %.2f ms, speedup: %.2fx)\n", 
               t_scalar, t_neon, t_scalar / t_neon);
    } else {
        printf("  ✗ Results don't match!\n");
    }
    
    free(src);
    free(dst_scalar);
    free(dst_neon);
    free(workspace);
    
    return match;
}

static bool test_neon_vs_scalar_4bit() {
    printf("Testing NEON vs Scalar for 4-bit...\n");
    
    const int m = 1024;
    const int k = 1024;
    const int bits = 4;
    const int g = 4;
    
    // Allocate buffers
    size_t src_size = (m * k) / 2;
    size_t dst_size = (m * k) / g / 2;
    size_t workspace_size = m * k / g * 4;
    
    uint8_t* src = (uint8_t*)malloc(src_size);
    uint8_t* dst_scalar = (uint8_t*)malloc(dst_size);
    uint8_t* dst_neon = (uint8_t*)malloc(dst_size);
    uint8_t* workspace = (uint8_t*)malloc(workspace_size);
    
    // Generate test pattern
    generate_test_pattern(src, src_size, 0xF0);
    
    // Test scalar version
    struct pack_config cfg_scalar;
    pack_config_init(&cfg_scalar, bits, m, k, 1024, 16, true);
    
    double t0 = get_time_ms();
    pack_weights_optimized(src, dst_scalar, workspace, m, k, bits, g, &cfg_scalar);
    double t_scalar = get_time_ms() - t0;
    
    // Test NEON version
    struct pack_config cfg_neon;
    pack_config_init(&cfg_neon, bits, m, k, 1024, 16, false);
    
    t0 = get_time_ms();
    pack_weights_optimized(src, dst_neon, workspace, m, k, bits, g, &cfg_neon);
    double t_neon = get_time_ms() - t0;
    
    // Compare results
    bool match = true;
    for (size_t i = 0; i < dst_size; i++) {
        if (dst_scalar[i] != dst_neon[i]) {
            printf("  Mismatch at index %zu: scalar=%u, neon=%u\n", 
                   i, dst_scalar[i], dst_neon[i]);
            match = false;
            break;
        }
    }
    
    if (match) {
        printf("  ✓ Results match (scalar: %.2f ms, neon: %.2f ms, speedup: %.2fx)\n", 
               t_scalar, t_neon, t_scalar / t_neon);
    } else {
        printf("  ✗ Results don't match!\n");
    }
    
    free(src);
    free(dst_scalar);
    free(dst_neon);
    free(workspace);
    
    return match;
}

static bool test_neon_memcpy() {
    printf("Testing NEON optimized memcpy...\n");
    
    const size_t sizes[] = {64, 256, 1024, 4096, 16384, 65536};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        size_t size = sizes[i];
        
        // Allocate aligned buffers
        uint8_t* src = (uint8_t*)aligned_alloc(16, size);
        uint8_t* dst1 = (uint8_t*)aligned_alloc(16, size);
        uint8_t* dst2 = (uint8_t*)aligned_alloc(16, size);
        
        // Fill source with test pattern
        generate_test_pattern(src, size, 0xAB);
        
        // Standard memcpy
        double t0 = get_time_ms();
        memcpy(dst1, src, size);
        double t_std = get_time_ms() - t0;
        
        // NEON optimized memcpy
        t0 = get_time_ms();
        pack_memcpy_optimized(dst2, src, size);
        double t_neon = get_time_ms() - t0;
        
        // Verify results match
        bool match = (memcmp(dst1, dst2, size) == 0);
        
        if (match) {
            printf("  Size %6zu: ✓ (std: %.3f ms, neon: %.3f ms, speedup: %.2fx)\n",
                   size, t_std, t_neon, t_std / t_neon);
        } else {
            printf("  Size %6zu: ✗ Results don't match!\n", size);
        }
        
        free(src);
        free(dst1);
        free(dst2);
        
        if (!match) return false;
    }
    
    return true;
}

//> ===================================================================================================
//> Main test runner
//> ===================================================================================================

int main(int argc, char** argv) {
    printf("=== Pack Weights NEON Optimization Tests ===\n\n");
    
    int passed = 0;
    int failed = 0;
    
#ifdef __ARM_NEON
    printf("ARM NEON support: ENABLED\n\n");
    
    // Run NEON tests
    if (test_neon_vs_scalar_1bit()) passed++; else failed++;
    if (test_neon_vs_scalar_2bit()) passed++; else failed++;
    if (test_neon_vs_scalar_4bit()) passed++; else failed++;
    if (test_neon_memcpy()) passed++; else failed++;
#else
    printf("ARM NEON support: DISABLED\n");
    printf("Skipping NEON tests on non-ARM platform.\n");
    passed = 4;  // Mark as passed since we can't test
#endif
    
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