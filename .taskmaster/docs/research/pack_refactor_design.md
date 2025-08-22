# Pack Function Refactor Design Document

## Overview
Refactoring the `qlutattn_pack_weights` and `qlutattn_pack_scales` functions to optimize performance through NEON SIMD instructions and improved memory access patterns.

## Current Implementation Analysis

### Key Components
1. **Bit-plane Separation**: Extracts individual bits from quantized values and groups them for LUT processing (g=4)
2. **SIMD Layout Permutation**: Complex 3-stage reshape operations for SIMD optimization
3. **Scale Packing**: Interleaves scales with zero points for cache locality

### Performance Issues
- No NEON SIMD optimization
- Complex nested loops with poor locality
- Inefficient memory access patterns
- Excessive temporary buffer usage

## New Architecture Design

### Design Principles
1. **SIMD-First Design**: Leverage ARM NEON instructions for parallel processing
2. **Cache-Friendly Access**: Optimize memory access patterns for cache efficiency
3. **Modular Implementation**: Separate bit-width specific optimizations
4. **Minimal Overhead**: Reduce temporary buffer usage and data movement

### Implementation Strategy

#### 1. Optimized Pack Weights Function

```cpp
// New structure for optimized pack functions
typedef void (*pack_weights_fn)(const uint8_t* src, uint8_t* dst, 
                                int m, int k, const pack_config* cfg);

struct pack_config {
    int bits;           // 1, 2, or 4
    int g;              // Group size (4)
    int bm;             // Block size (256/512/1024)
    int kfactor;        // K unrolling factor
    int simd_width;     // NEON vector width
    bool use_neon;      // NEON availability flag
};

// Function pointer table for different bit widths
static const pack_weights_fn pack_functions[3][2] = {
    {pack_weights_1bit_scalar, pack_weights_1bit_neon},
    {pack_weights_2bit_scalar, pack_weights_2bit_neon},
    {pack_weights_4bit_scalar, pack_weights_4bit_neon}
};
```

#### 2. NEON Optimized Implementation

```cpp
// Example: 2-bit NEON pack implementation
void pack_weights_2bit_neon(const uint8_t* src, uint8_t* dst,
                            int m, int k, const pack_config* cfg) {
    const int g = cfg->g;
    const int chunk_size = 256;  // Process in 256-byte chunks
    
    // Process 16 elements at a time using NEON
    for (int im = 0; im < m/2; im += 16) {
        for (int ik = 0; ik < k; ik += chunk_size) {
            // Prefetch next chunk
            __builtin_prefetch(src + im*k + ik + chunk_size, 0, 1);
            
            // Load 16 2-bit values
            uint8x16_t q_vals = vld1q_u8(src + (im*k + ik)/4);
            
            // Extract bit planes using NEON
            uint8x16_t plane0 = vandq_u8(q_vals, vdupq_n_u8(0x55));  // bits 0,2,4,6
            uint8x16_t plane1 = vandq_u8(vshrq_n_u8(q_vals, 1), vdupq_n_u8(0x55));  // bits 1,3,5,7
            
            // Group for LUT (g=4)
            uint8x16x2_t grouped = vzipq_u8(plane0, plane1);
            
            // Store with optimized layout
            vst2q_u8(dst + calculate_dst_offset(im, ik, cfg), grouped);
        }
    }
}
```

#### 3. Optimized Scale Packing

```cpp
void pack_scales_neon(const float* scale_ptr, const float* zero_ptr,
                     ggml_fp16_t* scales_out, int m, int k, 
                     int group_size, const pack_config* cfg) {
    if (cfg->use_neon) {
        // Process 8 scales at a time
        for (int i = 0; i < m * k / group_size; i += 8) {
            // Load 8 float scales
            float32x4_t scales_low = vld1q_f32(scale_ptr + i);
            float32x4_t scales_high = vld1q_f32(scale_ptr + i + 4);
            
            // Convert to fp16
            float16x4_t fp16_low = vcvt_f16_f32(scales_low);
            float16x4_t fp16_high = vcvt_f16_f32(scales_high);
            
            // Interleave with zero points if needed
            if (zero_ptr) {
                float32x4_t zeros_low = vld1q_f32(zero_ptr + i);
                float32x4_t zeros_high = vld1q_f32(zero_ptr + i + 4);
                
                float16x4_t fp16_zeros_low = vcvt_f16_f32(zeros_low);
                float16x4_t fp16_zeros_high = vcvt_f16_f32(zeros_high);
                
                // Interleave scales and zeros
                float16x4x2_t interleaved_low = vzip_f16(fp16_low, fp16_zeros_low);
                float16x4x2_t interleaved_high = vzip_f16(fp16_high, fp16_zeros_high);
                
                vst2_f16((float16_t*)scales_out + i*2, interleaved_low);
                vst2_f16((float16_t*)scales_out + i*2 + 8, interleaved_high);
            } else {
                vst1_f16((float16_t*)scales_out + i, fp16_low);
                vst1_f16((float16_t*)scales_out + i + 4, fp16_high);
            }
        }
    } else {
        // Scalar fallback
        pack_scales_scalar(scale_ptr, zero_ptr, scales_out, m, k, group_size, cfg);
    }
}
```

### Memory Layout Optimization

#### Current Layout Issues
- Complex 3-stage permutation causes cache misses
- Poor spatial locality during LUT lookup
- Excessive indirection in index calculation

#### New Layout Design
```
Original: [M][K] -> Complex 3-stage permutation
New:      [M/BM][BM/SIMD][K/G][G] -> Direct SIMD-friendly layout

Benefits:
- Direct mapping for SIMD loads
- Better cache line utilization
- Simplified index calculation
- Reduced memory bandwidth
```

### Performance Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Pack throughput | ~2 GB/s | 10 GB/s | 5x |
| Cache miss rate | ~15% | <5% | 3x |
| SIMD utilization | 0% | 80% | N/A |
| Memory bandwidth | ~8 GB/s | ~4 GB/s | 2x reduction |

### Testing Strategy

1. **Correctness Tests**
   - Bit-exact comparison with reference implementation
   - Edge cases (non-aligned dimensions)
   - All quantization types (1/2/4-bit)

2. **Performance Benchmarks**
   - Throughput measurement (GB/s)
   - Cache miss analysis
   - SIMD instruction count
   - Comparison across different tensor sizes

3. **Integration Tests**
   - End-to-end LUT multiplication
   - Model inference accuracy
   - Memory consumption

## Implementation Phases

### Phase 1: Basic Refactor (Task 5.2)
- Implement modular pack functions
- Separate bit-width specific code
- Create scalar reference implementation

### Phase 2: NEON Optimization (Task 5.3)
- Add NEON intrinsics for each bit width
- Optimize bit-plane separation
- Implement vectorized permutation

### Phase 3: Memory Optimization (Task 5.5)
- Implement prefetching
- Optimize chunk size for cache
- Reduce temporary buffer usage

### Phase 4: Testing & Validation (Task 5.6-5.7)
- Create comprehensive test suite
- Performance benchmarking
- Integration testing

## Code Organization

```
ggml/src/ggml-cpu/
├── pack_weights.h        # New header for pack functions
├── pack_weights.cpp      # Main implementation
├── pack_weights_neon.cpp # NEON optimized versions
├── pack_weights_test.cpp # Unit tests
└── ops.cpp              # Integration point
```

## Migration Plan

1. Keep existing functions as fallback
2. Add feature flag for new implementation
3. Gradual rollout with performance monitoring
4. Remove old implementation after validation

## Risk Mitigation

- **Risk**: Bit-exact differences affecting model accuracy
  - **Mitigation**: Extensive validation against reference implementation
  
- **Risk**: Performance regression on specific hardware
  - **Mitigation**: Runtime detection and fallback mechanism
  
- **Risk**: Increased code complexity
  - **Mitigation**: Modular design with clear interfaces

## Success Criteria

1. 4-6x speedup in pack operations
2. Cache miss rate below 5%
3. No accuracy degradation
4. Maintainable and documented code
5. Full test coverage