#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//> ===================================================================================================
//> Pack configuration and function types
//> ===================================================================================================

// Configuration for pack operations
struct pack_config {
    int bits;               // Quantization bit width (1, 2, or 4)
    int g;                  // Group size for LUT (typically 4)
    int bm;                 // Block size (256/512/1024/2048)
    int kfactor;           // K-dimension unrolling factor
    int simd_width;        // SIMD vector width
    int simd_n_in;         // SIMD input width
    int simd_n_out;        // SIMD output width
    int ngroups_per_elem;  // Groups per element
    int mgroup;            // M group size
    int nelem_per_byte;    // Elements per byte
    bool use_neon;         // Enable NEON optimization
};

// Function pointer types for pack operations
typedef void (*pack_weights_fn)(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                                int m, int k, const struct pack_config* cfg);

typedef void (*pack_scales_fn)(const float* scale_ptr, const float* zero_ptr,
                               ggml_fp16_t* scales_out, int m, int k, 
                               int group_size, int scales_size,
                               const struct pack_config* cfg);

//> ===================================================================================================
//> Main pack functions
//> ===================================================================================================

// Optimized pack weights function with runtime dispatch
void pack_weights_optimized(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                           int m, int k, int bits, int g,
                           const struct pack_config* cfg);

// Optimized pack scales function with runtime dispatch
void pack_scales_optimized(const float* scale_ptr, const float* zero_ptr,
                          ggml_fp16_t* scales_out, int m, int k, int bits,
                          int group_size, int scales_size,
                          const struct pack_config* cfg);

//> ===================================================================================================
//> Scalar implementations (fallback)
//> ===================================================================================================

void pack_weights_1bit_scalar(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                              int m, int k, const struct pack_config* cfg);

void pack_weights_2bit_scalar(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                              int m, int k, const struct pack_config* cfg);

void pack_weights_4bit_scalar(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                              int m, int k, const struct pack_config* cfg);

void pack_scales_scalar(const float* scale_ptr, const float* zero_ptr,
                        ggml_fp16_t* scales_out, int m, int k, int bits,
                        int group_size, int scales_size,
                        const struct pack_config* cfg);

//> ===================================================================================================
//> NEON optimized implementations
//> ===================================================================================================

#ifdef __ARM_NEON

void pack_weights_1bit_neon(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                            int m, int k, const struct pack_config* cfg);

void pack_weights_2bit_neon(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                            int m, int k, const struct pack_config* cfg);

void pack_weights_4bit_neon(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                            int m, int k, const struct pack_config* cfg);

void pack_scales_neon(const float* scale_ptr, const float* zero_ptr,
                      ggml_fp16_t* scales_out, int m, int k, int bits,
                      int group_size, int scales_size,
                      const struct pack_config* cfg);

#endif // __ARM_NEON

//> ===================================================================================================
//> Batch processing structures and functions
//> ===================================================================================================

// Batch processing context for multiple tensors
struct pack_batch_context {
    int num_tensors;           // Number of tensors in batch
    int total_workspace_size;  // Total workspace needed
    int cache_line_size;       // Cache line size for alignment
    bool use_parallel;         // Enable parallel processing
    int num_threads;           // Number of threads to use
};

// Batch operation descriptor
struct pack_batch_op {
    const uint8_t* src;        // Source data pointer
    uint8_t* dst;              // Destination data pointer
    const float* scale_ptr;    // Scale pointer (for scales)
    const float* zero_ptr;     // Zero point pointer (for scales)
    ggml_fp16_t* scales_out;   // Output scales (for scales)
    int m;                     // M dimension
    int k;                     // K dimension
    int bits;                  // Bit width
    int group_size;            // Group size for scales
    int scales_size;           // Scales size
    bool is_weights;           // True for weights, false for scales
};

// Batch processing functions
void pack_batch_init(struct pack_batch_context* ctx, int num_tensors,
                     int cache_line_size, bool use_parallel);

void pack_batch_process_weights(struct pack_batch_context* ctx,
                                struct pack_batch_op* ops, int num_ops,
                                uint8_t* shared_workspace,
                                const struct pack_config* cfg);

void pack_batch_process_scales(struct pack_batch_context* ctx,
                               struct pack_batch_op* ops, int num_ops,
                               const struct pack_config* cfg);

//> ===================================================================================================
//> Memory optimization functions
//> ===================================================================================================

// Optimized memory copy with prefetching
void pack_memcpy_optimized(void* dst, const void* src, size_t size);

// Cache-aware data reordering
void pack_reorder_for_cache(uint8_t* data, int m, int k, int bits,
                            int cache_line_size);

// Memory pool for workspace allocation
struct pack_memory_pool {
    uint8_t* base;             // Base pointer
    size_t size;               // Total size
    size_t used;               // Used size
    size_t alignment;          // Alignment requirement
};

void pack_memory_pool_init(struct pack_memory_pool* pool,
                           uint8_t* buffer, size_t size, size_t alignment);

void* pack_memory_pool_alloc(struct pack_memory_pool* pool, size_t size);

void pack_memory_pool_reset(struct pack_memory_pool* pool);

//> ===================================================================================================
//> Utility functions
//> ===================================================================================================

// Initialize pack configuration with optimal settings
void pack_config_init(struct pack_config* cfg, int bits, int m, int k,
                     int bm, int kfactor, bool force_scalar);

// Calculate destination offset for packed data
static inline int calculate_pack_dst_offset(int im, int ik, int k,
                                           const struct pack_config* cfg) {
    // Simplified offset calculation for better performance
    const int k_g = ik / cfg->g;
    const int im_block = im / cfg->bm;
    const int im_in_block = im % cfg->bm;
    
    return im_block * (cfg->bm * k / cfg->g) + 
           im_in_block * (k / cfg->g) + 
           k_g;
}

// Check if dimensions are compatible with pack requirements
static inline bool pack_dimensions_valid(int m, int k, int bits, int g) {
    return (m % bits == 0) && 
           (k % g == 0) && 
           (m * k % (128 * 128) == 0);
}

// Get optimal chunk size for cache
static inline int pack_get_optimal_chunk_size(int total_size, int cache_size) {
    // Use 1/4 of L2 cache for optimal performance
    int optimal = cache_size / 4;
    
    // Align to cache line boundary (64 bytes)
    optimal = (optimal / 64) * 64;
    
    // Clamp to reasonable range
    if (optimal < 4096) optimal = 4096;
    if (optimal > 65536) optimal = 65536;
    
    return optimal;
}

#ifdef __cplusplus
}
#endif