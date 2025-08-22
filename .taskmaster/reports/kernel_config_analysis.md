# QLUTATTN Kernel Config Analysis Report

## 1. Kernel Config Structure Definition

### Core Structure (`qlutattn_kernel_config`)
Located in: `ggml/src/ggml-cpu/qlutattn/qlutattn-config.h:16-34`

```cpp
struct qlutattn_kernel_config {
    // Group processing parameters
    int32_t g;                  // Group size for LUT (typically 4)
    int32_t ngroups_per_elem;   // Groups per byte element (8/g = 2)
    
    // Quantization parameters
    int32_t q_group_size;       // Weight quantization group size (64/128)
    int32_t act_group_size;     // Activation quantization group size
    
    // Scale configuration
    bool has_scale;             // Whether weights have scales
    bool has_zero_point;        // Whether weights have zero points
    bool one_scale;             // Single scale for entire tensor
    
    // Performance tuning
    int kfactor;                // K-dimension unrolling factor (8/16)
    int bits;                   // Quantization bit width (1/2/4)
    int actk;                   // act_group_size / g
    
    // SIMD and block parameters
    int32_t bm;                 // Block size (256/512/1024/2048)
    uint32_t simd_n_in;         // SIMD input width (16)
    uint32_t simd_n_out;        // SIMD output width (8/16)
    
    int32_t chunk_n;            // For compatibility (8)
};
```

## 2. Config Management System

### Singleton Pattern Implementation
- **Location**: `qlutattn-config.cpp:13-155`
- **Class**: `QlutattnConfigManager`
- **Thread-safe**: Uses mutex locking for all operations
- **Key generation**: `"M{M}_K{K}_b{bits}"` format

### C Interface Functions
- `ggml_qlutattn_config_init()` - Initialize system
- `ggml_qlutattn_get_config(M, K, bits)` - Retrieve config
- `ggml_qlutattn_register_config(M, K, bits, config)` - Add new config
- `ggml_qlutattn_config_is_initialized()` - Check status
- `ggml_qlutattn_config_cleanup()` - Clean up

## 3. Kernel Config Call Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     ops.cpp (Entry Points)                   │
├─────────────────────────────────────────────────────────────┤
│ ggml_compute_forward_pack_mixed_kv_cache()                   │
│ ggml_compute_forward_flash_attn_ext_mixed_kv()               │
│ ggml_compute_forward_flash_attn_ext_qlutattn()               │
│                           ▼                                   │
│      ggml_qlutattn_get_config(M, K, bits)                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              qlutattn-config.cpp (Config Manager)            │
├─────────────────────────────────────────────────────────────┤
│ QlutattnConfigManager::get_config()                          │
│   - Lookup by key "M{M}_K{K}_b{bits}"                       │
│   - Fallback to any config with matching bits                │
│   - Return nullptr if not found                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  qlutattn.cpp (Processing)                   │
├─────────────────────────────────────────────────────────────┤
│ vec_dot functions:                                           │
│   - ggml_vec_dot_qlutattn_k1_128x128                        │
│   - ggml_vec_dot_qlutattn_k2_128x128                        │
│   - ggml_vec_dot_qlutattn_k4_128x128                        │
│                           ▼                                   │
│ qlutattn_lut_ctor_int8_g4() - LUT construction              │
│ qgemm_lut_int8_g4() - Matrix multiplication                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    tbl.cpp (Computation)                     │
├─────────────────────────────────────────────────────────────┤
│ qgemm_lut_int8_g4()                                          │
│   - Extract all kernel_config fields                         │
│   - Use config for LUT-based matrix multiplication           │
│   - Parameters control:                                      │
│     * Group processing (g, ngroups_per_elem)                 │
│     * Quantization (bits, q_group_size, act_group_size)      │
│     * Scales (has_scale, has_zero_point, one_scale)          │
│     * Performance (kfactor, bm, simd_n_in/out)               │
└─────────────────────────────────────────────────────────────┘
```

## 4. Hardcoded Configuration Values

### Default Configs in qlutattn-config.cpp:27-83
```cpp
// 4-bit config
{
    .g                = 4,
    .ngroups_per_elem = 2,    // 8/4 = 2
    .q_group_size     = 128,
    .act_group_size   = 64,
    .has_scale        = true,
    .kfactor          = 16,
    .bits             = 4,
    .actk             = 16,   // 64/4 = 16
    .has_zero_point   = false,
    .one_scale        = true,
    .bm               = 256,
    .simd_n_in        = 16,
    .simd_n_out       = 16,
    .chunk_n          = 8
}

// Similar configs for 2-bit and 1-bit
// Registered for keys:
// - "M1_K1_b4", "M128_K128_b4"
// - "M128_K128_b2"
// - "M128_K128_b1"
```

### Hardcoded Values in ops.cpp
```cpp
const int64_t PACK_SIZE       = 128;  // head_dim (ops.cpp:682,983,7987,8006)
const int64_t PACK_CHUNK_SIZE = 128;  // tokens per chunk (ops.cpp:683,984,7988)
```

### Test Configs in ops.cpp
```cpp
// Test lookups at ops.cpp:7759, 7970, 8586
ggml_qlutattn_get_config(1, 1, 4)      // Test config
ggml_qlutattn_get_config(128, 128, 4)  // 128x128 config
```

## 5. PACK_SIZE and PACK_CHUNK_SIZE Usage

### Definition
- **PACK_SIZE**: 128 - Represents head dimension
- **PACK_CHUNK_SIZE**: 128 - Tokens processed per chunk

### Usage Patterns
1. **KV Cache Packing** (ops.cpp:682-924)
   - Determines chunk boundaries for quantized KV cache
   - Calculate number of KV heads: `n_kv_heads = ne00 / (PACK_SIZE * PACK_CHUNK_SIZE)`
   - Buffer sizes based on PACK_SIZE * PACK_CHUNK_SIZE

2. **Flash Attention** (ops.cpp:7983-8176)
   - Q dimensions: `[head_dim, q_len, n_q_head, n_q_batch]`
   - K/V dimensions: `[head_dim * PACK_CHUNK_SIZE, kv_len / PACK_CHUNK_SIZE, n_q_head, n_q_batch]`
   - Processing loops iterate over PACK_CHUNK_SIZE tokens

3. **Memory Allocation**
   ```cpp
   PSEUDO_QUANT_SIZE = PACK_CHUNK_SIZE * PACK_SIZE / nelem_per_byte
   SRC0_F32_SIZE = PACK_CHUNK_SIZE * PACK_SIZE * sizeof(float)
   REPACK_SIZE = PACK_CHUNK_SIZE * PACK_SIZE * 4 * sizeof(uint8_t)
   ```

## 6. Key Findings

### Critical Constraints
1. **Fixed 128x128 blocks**: Current implementation hardcoded for 128x128 processing
2. **Group size g=4**: LUT processes 4 elements atomically
3. **Config lookup fallback**: If exact (M,K,bits) not found, uses any config with matching bits
4. **Thread safety**: All config operations protected by mutex

### Config Dependencies
- `actk = act_group_size / g` (must be integer)
- `ngroups_per_elem = 8 / g` (for byte alignment)
- `q_group_size` must be multiple of 32 or 64
- `act_group_size` must be 32 or 64

### Performance Critical Parameters
- **bm**: Block size for chunking (256/512/1024/2048)
- **kfactor**: K-dimension unrolling (8/16)
- **simd_n_in/out**: SIMD vector widths (16/8)

## 7. Recommendations

1. **Generalize PACK_SIZE/PACK_CHUNK_SIZE**: Make these configurable rather than hardcoded
2. **Dynamic config generation**: Add auto-tuning for missing (M,K,bits) combinations
3. **Config validation**: Add checks for constraint violations
4. **Performance profiling**: Track which configs are most frequently accessed
5. **Config persistence**: Save auto-tuned configs to disk for reuse