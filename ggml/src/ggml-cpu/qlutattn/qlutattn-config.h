#ifndef GGML_QLUTATTN_CONFIG_H
#define GGML_QLUTATTN_CONFIG_H

//> ===================================================================================================
//> Unified PACK dimensions for QLUTATTN
//> ===================================================================================================
#define QLUTATTN_PACK_SIZE       128  // Fixed block size for QLUTATTN processing
#define QLUTATTN_PACK_CHUNK_SIZE 128  // Fixed chunk size for QLUTATTN processing

// Compile-time check to ensure consistency
#if QLUTATTN_PACK_SIZE != 128 || QLUTATTN_PACK_CHUNK_SIZE != 128
    #error "QLUTATTN currently requires PACK_SIZE and PACK_CHUNK_SIZE to be 128"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

//> ===================================================================================================
//> Simple global config access for QLUTATTN
//> ===================================================================================================

// Kernel configuration structure (moved from qlut_ctor.h to avoid circular dependency)
struct qlutattn_kernel_config {
    int32_t g;
    int32_t ngroups_per_elem;
    int32_t q_group_size;
    int32_t act_group_size;
    
    bool has_scale;
    int  kfactor;
    int  bits;
    int  actk;  // should be equal to (act_group_size / g)
    bool has_zero_point;
    bool one_scale;
    
    int32_t  bm;
    uint32_t simd_n_in;
    uint32_t simd_n_out;
    
    int32_t chunk_n;  // For compatibility
};

// Initialize the global config system (call once at startup)
void ggml_qlutattn_config_init(void);

// Get kernel config - returns NULL if not found
// Thread-safe with internal locking
const struct qlutattn_kernel_config * ggml_qlutattn_get_config(int M, int K, int bits);

// Register a config for specific M, K, bits combination
// Returns false if config already exists
bool ggml_qlutattn_register_config(int M, int K, int bits, const struct qlutattn_kernel_config * config);

// Check if config system is initialized
bool ggml_qlutattn_config_is_initialized(void);

// Cleanup (optional, for clean shutdown)
void ggml_qlutattn_config_cleanup(void);

//> ===================================================================================================
//> Unified kernel config interface for QLUTATTN
//> ===================================================================================================

// Get kernel config from type info - main unified interface
// This function determines M, K, bits from the type and dimensions
// Returns NULL if type is not supported or config not found
// type_bits: quantization bits (1, 2, or 4)
// is_v_type: true for V-types (transposed), false for K-types
const struct qlutattn_kernel_config * ggml_qlutattn_get_unified_config(
    int32_t type_bits,   // Quantization bits (1, 2, or 4)
    int32_t k_size,      // Key/Value dimension size
    int32_t v_size       // Value dimension size (may differ from k_size)
);

// Validate that two configs are identical (for consistency checking)
bool ggml_qlutattn_configs_equal(
    const struct qlutattn_kernel_config * config1,
    const struct qlutattn_kernel_config * config2
);

#ifdef __cplusplus
}
#endif

#endif // GGML_QLUTATTN_CONFIG_H