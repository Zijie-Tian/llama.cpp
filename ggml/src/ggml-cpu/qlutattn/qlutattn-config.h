#ifndef GGML_QLUTATTN_CONFIG_H
#define GGML_QLUTATTN_CONFIG_H

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

#ifdef __cplusplus
}
#endif

#endif // GGML_QLUTATTN_CONFIG_H