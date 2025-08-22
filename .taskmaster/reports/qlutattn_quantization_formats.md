# QLUTATTN Quantization Formats Analysis

## Overview
QLUTATTN supports 6 quantization formats for KV cache compression, divided into two categories:
- **K-types (Key cache)**: K1, K2, K4 - Standard packing
- **V-types (Value cache)**: V1, V2, V4 - Transposed packing

## 1. Block Structure Definitions

### K-Type Formats (Key Cache)

#### QLUTATTN_K1_128x128 (1-bit quantization)
```cpp
// ggml-common.h:287-294
#define QKLUTATTN_KV1_128x128 (128 * 128)  // 16384 elements

typedef struct {
    uint8_t qs[QKLUTATTN_KV1_128x128 / 8 + 128 * sizeof(float) * 2];
    // = [16384/8 + 128*4*2] = [2048 + 1024] = 3072 bytes
} block_qlutattn_kv1_128x128;
```
- **Storage**: 1 bit per element → 8 elements per byte
- **Scales**: 128 float scales + 128 float zero points
- **Total size**: 3072 bytes for 16384 values
- **Compression ratio**: ~21.3:1 (vs FP32)

#### QLUTATTN_K2_128x128 (2-bit quantization)
```cpp
// ggml-common.h:296-303
#define QKLUTATTN_KV2_128x128 (128 * 128)

typedef struct {
    uint8_t qs[QKLUTATTN_KV2_128x128 / 4 + 128 * sizeof(float) * 2];
    // = [16384/4 + 128*4*2] = [4096 + 1024] = 5120 bytes
} block_qlutattn_kv2_128x128;
```
- **Storage**: 2 bits per element → 4 elements per byte
- **Scales**: 128 float scales + 128 float zero points
- **Total size**: 5120 bytes for 16384 values
- **Compression ratio**: ~12.8:1

#### QLUTATTN_K4_128x128 (4-bit quantization)
```cpp
// ggml-common.h:305-312
#define QKLUTATTN_KV4_128x128 (128 * 128)

typedef struct {
    uint8_t qs[QKLUTATTN_KV4_128x128 / 2 + 128 * sizeof(float) * 2];
    // = [16384/2 + 128*4*2] = [8192 + 1024] = 9216 bytes
} block_qlutattn_kv4_128x128;
```
- **Storage**: 4 bits per element → 2 elements per byte
- **Scales**: 128 float scales + 128 float zero points
- **Total size**: 9216 bytes for 16384 values
- **Compression ratio**: ~7.1:1

### V-Type Formats (Value Cache)

V-types use the same bit widths (1, 2, 4) but with **transposed storage layout** for better memory access patterns during attention computation.

## 2. Usage in ops.cpp

### K-Type Processing (ops.cpp:706-824)
```cpp
case GGML_TYPE_QLUTATTN_K4_128x128:
{
    // Standard row-major packing
    // Process in chunks of PACK_CHUNK_SIZE (128)
    // Direct quantization without transpose
    for (int i = 0; i < PACK_CHUNK_SIZE * PACK_SIZE; i++) {
        src0_f32[i] = GGML_FP16_TO_FP32(src0_ptr[i]);
    }
    qlutattn_quantize_block(quantize_block_q, src0_f32, qweight_ptr, 
                            PACK_CHUNK_SIZE * PACK_SIZE);
}
```

### V-Type Processing (ops.cpp:825-924)
```cpp
case GGML_TYPE_QLUTATTN_V4_128x128:
{
    // Transposed packing for value cache
    // NOTICE: We needs to do transpose
    GGML_ASSERT(PACK_SIZE == PACK_CHUNK_SIZE);
    for (int i = 0; i < PACK_CHUNK_SIZE; i++) {
        for (int j = 0; j < PACK_SIZE; j++) {
            int src_idx = j * PACK_CHUNK_SIZE + i;  // Transpose
            int dst_idx = i * PACK_SIZE + j;
            src0_f32[dst_idx] = GGML_FP16_TO_FP32(src0_ptr[src_idx]);
        }
    }
}
```

## 3. Key Differences Between K and V Types

| Aspect | K-Types (Key) | V-Types (Value) |
|--------|---------------|-----------------|
| **Layout** | Row-major | Column-major (transposed) |
| **Memory Access** | Sequential write | Transposed for better cache locality |
| **Use Case** | Query-Key dot products | Weighted value aggregation |
| **Processing** | Direct quantization | Transpose → Quantize |
| **vec_dot Function** | `ggml_vec_dot_qlutattn_kv{1,2,4}_128x128` | `pv_vec_dot_fp16` (different API) |

## 4. Quantization Process Flow

### Common Steps
1. **Group size**: 128 elements per quantization group
2. **Scale computation**: Per-group min/max normalization
3. **Bit packing**: Pack multiple elements per byte
4. **LUT preparation**: Pre-compute lookup tables for fast inference

### K-Type Flow
```
Input FP16 → Convert to FP32 → Quantize → Pack bits → Store with scales
```

### V-Type Flow
```
Input FP16 → Convert to FP32 → Transpose → Quantize → Pack bits → Store with scales
```

## 5. Performance Characteristics

### Memory Footprint
- **1-bit**: 3KB per 128×128 block (highest compression)
- **2-bit**: 5KB per 128×128 block (balanced)
- **4-bit**: 9KB per 128×128 block (best quality)

### Computation Trade-offs
- **K-types**: Optimized for sequential access in Q×K computation
- **V-types**: Optimized for strided access in attention-weighted aggregation

### LUT Processing
All formats use:
- **Group size g=4**: Process 4 elements atomically
- **16-entry LUT**: Pre-computed dot products for all 2^4 bit patterns
- **SIMD width**: 16 elements for vectorized operations

## 6. Selection Guidelines

### Use K-Types When:
- Computing query-key attention scores
- Sequential memory access is beneficial
- Cache line utilization is critical

### Use V-Types When:
- Aggregating values with attention weights
- Transposed access pattern improves performance
- Working with column-major data layouts

### Bit Width Selection:
- **1-bit**: Maximum compression, suitable for binary features
- **2-bit**: Good compression with reasonable quality
- **4-bit**: Best quality-compression balance for most use cases

## 7. Implementation Notes

### Hardcoded Constraints
- Fixed 128×128 block size
- Group size must be 128
- PACK_SIZE == PACK_CHUNK_SIZE == 128
- All formats use same kernel_config lookup

### Memory Layout
```
[Quantized Data][Scale Buffer][Zero Point Buffer]
|<-- bits/8 -->|<-- 128*4 -->|<-- 128*4 ------->|
```

### Future Improvements
1. Support variable block sizes beyond 128×128
2. Adaptive bit width selection based on data distribution
3. Mixed-precision within same tensor
4. Dynamic K/V type selection based on access patterns