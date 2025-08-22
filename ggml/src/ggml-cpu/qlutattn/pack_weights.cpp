#include "pack_weights.h"
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

//> ===================================================================================================
//> Type accessors for different bit widths
//> ===================================================================================================

struct QlutattnI1TypeAccessor {
    static constexpr int BITS   = 1;
    static constexpr int n_elem = 8 / BITS;

    static uint8_t get_q(const void * data, int idx) {
        const uint8_t * qs       = (const uint8_t *) data;
        int             elem_idx = idx % n_elem;
        return (qs[idx / n_elem] >> ((n_elem - 1 - elem_idx) * BITS)) & 0x1;
    }
};

struct QlutattnI2TypeAccessor {
    static constexpr int BITS   = 2;
    static constexpr int n_elem = 8 / BITS;

    static uint8_t get_q(const void * data, int idx) {
        const uint8_t * qs       = (const uint8_t *) data;
        int             elem_idx = idx % n_elem;
        return (qs[idx / n_elem] >> ((n_elem - 1 - elem_idx) * BITS)) & 0x3;
    }
};

struct QlutattnI4TypeAccessor {
    static constexpr int BITS   = 4;
    static constexpr int n_elem = 8 / BITS;

    static uint8_t get_q(const void * data, int idx) {
        const uint8_t * qs       = (const uint8_t *) data;
        int             elem_idx = idx % n_elem;
        return (qs[idx / n_elem] >> ((n_elem - 1 - elem_idx) * BITS)) & 0xF;
    }
};

//> ===================================================================================================
//> Configuration initialization
//> ===================================================================================================

void pack_config_init(struct pack_config* cfg, int bits, int m, int k,
                     int bm, int kfactor, bool force_scalar) {
    assert(cfg != NULL);
    
    cfg->bits = bits;
    cfg->g = 4;  // Fixed group size for LUT
    cfg->bm = bm;
    cfg->kfactor = kfactor;
    cfg->nelem_per_byte = 8 / bits;
    
    // SIMD configuration
    cfg->simd_width = 16;  // NEON 128-bit vectors
    cfg->simd_n_in = cfg->simd_width / sizeof(uint8_t);
    cfg->simd_n_out = cfg->simd_width / sizeof(uint8_t);
    cfg->ngroups_per_elem = 8 / cfg->g;
    cfg->mgroup = cfg->ngroups_per_elem * cfg->simd_n_in;
    
#ifdef __ARM_NEON
    cfg->use_neon = !force_scalar;
#else
    cfg->use_neon = false;
#endif
}

//> ===================================================================================================
//> Main dispatch functions
//> ===================================================================================================

void pack_weights_optimized(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                           int m, int k, int bits, int g,
                           const struct pack_config* cfg) {
    assert(src != NULL && dst != NULL && workspace != NULL && cfg != NULL);
    assert(bits == 1 || bits == 2 || bits == 4);
    assert(pack_dimensions_valid(m, k, bits, g));
    
    // Select appropriate implementation based on bit width and NEON availability
    if (cfg->use_neon) {
#ifdef __ARM_NEON
        switch (bits) {
            case 1:
                pack_weights_1bit_neon(src, dst, workspace, m, k, cfg);
                break;
            case 2:
                pack_weights_2bit_neon(src, dst, workspace, m, k, cfg);
                break;
            case 4:
                pack_weights_4bit_neon(src, dst, workspace, m, k, cfg);
                break;
        }
#else
        // Fallback to scalar if NEON not available
        switch (bits) {
            case 1:
                pack_weights_1bit_scalar(src, dst, workspace, m, k, cfg);
                break;
            case 2:
                pack_weights_2bit_scalar(src, dst, workspace, m, k, cfg);
                break;
            case 4:
                pack_weights_4bit_scalar(src, dst, workspace, m, k, cfg);
                break;
        }
#endif
    } else {
        // Use scalar implementation
        switch (bits) {
            case 1:
                pack_weights_1bit_scalar(src, dst, workspace, m, k, cfg);
                break;
            case 2:
                pack_weights_2bit_scalar(src, dst, workspace, m, k, cfg);
                break;
            case 4:
                pack_weights_4bit_scalar(src, dst, workspace, m, k, cfg);
                break;
        }
    }
}

void pack_scales_optimized(const float* scale_ptr, const float* zero_ptr,
                          ggml_fp16_t* scales_out, int m, int k, int bits,
                          int group_size, int scales_size,
                          const struct pack_config* cfg) {
    assert(scale_ptr != NULL && scales_out != NULL && cfg != NULL);
    
    if (cfg->use_neon) {
#ifdef __ARM_NEON
        pack_scales_neon(scale_ptr, zero_ptr, scales_out, m, k, bits,
                        group_size, scales_size, cfg);
#else
        pack_scales_scalar(scale_ptr, zero_ptr, scales_out, m, k, bits,
                          group_size, scales_size, cfg);
#endif
    } else {
        pack_scales_scalar(scale_ptr, zero_ptr, scales_out, m, k, bits,
                          group_size, scales_size, cfg);
    }
}

//> ===================================================================================================
//> Scalar implementations (reference/fallback)
//> ===================================================================================================

// Helper function for bit-plane separation (scalar)
static void bitplane_separation_scalar(const uint8_t* qweight_ptr, uint8_t* repack_ws,
                                      int m, int k, int bits, int g) {
    memset(repack_ws, 0, m * k / g);
    
    for (int im = 0; im < m / bits; im++) {
        for (int ik = 0; ik < k; ik++) {
            uint8_t v;
            
            // Extract quantized value based on bit width
            if (bits == 1) {
                v = QlutattnI1TypeAccessor::get_q(qweight_ptr, im * k + ik);
            } else if (bits == 2) {
                v = QlutattnI2TypeAccessor::get_q(qweight_ptr, im * k + ik);
            } else if (bits == 4) {
                v = QlutattnI4TypeAccessor::get_q(qweight_ptr, im * k + ik);
            } else {
                assert(false && "Invalid bits");
                v = 0;
            }
            
            // Separate bits and pack into groups for LUT
            for (int ib = 0; ib < bits; ib++) {
                int new_ik    = ik / g;  // Group index
                int shft_left = ik % g;  // Position within group
                repack_ws[im * bits * k / g + ib * k / g + new_ik] += ((v >> ib) & 1) << shft_left;
            }
        }
    }
}

// Helper function for SIMD layout permutation (scalar)
static void simd_permutation_scalar(const uint8_t* repack_ws, uint8_t* qweights_out,
                                   int m, int k, int bits, int g,
                                   const struct pack_config* cfg) {
    const int ngroups_per_elem = cfg->ngroups_per_elem;
    const int bm = cfg->bm;
    const int simd_n_in = cfg->simd_n_in;
    const int simd_n_out = cfg->simd_n_out;
    const int kfactor = cfg->kfactor;
    const int mgroup = cfg->mgroup;
    const int nelem_per_byte = cfg->nelem_per_byte;
    
    memset(qweights_out, 0, m * k / g / nelem_per_byte);
    
    for (int im = 0; im < m / bits; im++) {
        for (int ib = 0; ib < bits; ib++) {
            for (int ik = 0; ik < k / g; ik++) {
                // Stage 1 - Reshape for SIMD width
                int new_im   = im / simd_n_out;
                int new_isno = im % simd_n_out;
                int new_idx  = new_im * bits * simd_n_out * k / g + 
                              ib * simd_n_out * k / g + 
                              new_isno * k / g + ik;
                
                // Stage 2 - Group-wise reshape
                int nb2      = k / g;
                int nb1      = simd_n_in * nb2;
                int nb0      = ngroups_per_elem * nb1;
                new_im       = new_idx / nb0;
                int new_ing  = (new_idx % nb0) / nb1;
                int new_isni = (new_idx % nb1) / nb2;
                int new_ik   = (new_idx % nb2);
                new_idx      = new_im * ngroups_per_elem * simd_n_in * k / g + 
                              new_isni * ngroups_per_elem * k / g +
                              new_ing * k / g + new_ik;
                
                // Stage 3 - Block-wise reshape for cache optimization
                int nb4     = kfactor;
                int nb3     = k / g / kfactor * nb4;
                nb2         = ngroups_per_elem * nb3;
                nb1         = simd_n_in * nb2;
                nb0         = bm / mgroup * nb1;
                new_im      = new_idx / nb0;
                int new_ibm = (new_idx % nb0) / nb1;
                new_isni    = (new_idx % nb1) / nb2;
                new_ing     = (new_idx % nb2) / nb3;
                new_ik      = (new_idx % nb3) / nb4;
                int new_ikf = (new_idx % nb4);
                new_idx     = new_im * k / g / kfactor * bm / mgroup * kfactor * simd_n_in * ngroups_per_elem +
                             new_ik * bm / mgroup * kfactor * simd_n_in * ngroups_per_elem +
                             new_ibm * kfactor * simd_n_in * ngroups_per_elem + 
                             new_ikf * simd_n_in * ngroups_per_elem +
                             new_isni * ngroups_per_elem + new_ing;
                new_idx = new_idx / ngroups_per_elem;
                
                // Accumulate the permuted bits
                qweights_out[new_idx] += repack_ws[im * bits * k / g + ib * k / g + ik] << (new_ing * g);
            }
        }
    }
}

void pack_weights_1bit_scalar(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                              int m, int k, const struct pack_config* cfg) {
    const int bits = 1;
    const int g = cfg->g;
    
    // Stage 1: Bit-plane separation
    bitplane_separation_scalar(src, workspace, m, k, bits, g);
    
    // Stage 2: SIMD layout permutation
    simd_permutation_scalar(workspace, dst, m, k, bits, g, cfg);
}

void pack_weights_2bit_scalar(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                              int m, int k, const struct pack_config* cfg) {
    const int bits = 2;
    const int g = cfg->g;
    
    // Stage 1: Bit-plane separation
    bitplane_separation_scalar(src, workspace, m, k, bits, g);
    
    // Stage 2: SIMD layout permutation
    simd_permutation_scalar(workspace, dst, m, k, bits, g, cfg);
}

void pack_weights_4bit_scalar(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                              int m, int k, const struct pack_config* cfg) {
    const int bits = 4;
    const int g = cfg->g;
    
    // Stage 1: Bit-plane separation
    bitplane_separation_scalar(src, workspace, m, k, bits, g);
    
    // Stage 2: SIMD layout permutation
    simd_permutation_scalar(workspace, dst, m, k, bits, g, cfg);
}

void pack_scales_scalar(const float* scale_ptr, const float* zero_ptr,
                        ggml_fp16_t* scales_out, int m, int k, int bits,
                        int group_size, int scales_size,
                        const struct pack_config* cfg) {
    const int bm = cfg->bm;
    const int simd_n_out = cfg->simd_n_out;
    
    if (scales_size < m / bits) {
        // BitNet-like scale (single scale for all groups)
        for (int i = 0; i < scales_size; i++) {
            scales_out[i] = GGML_FP32_TO_FP16(scale_ptr[i]);
        }
    } else {
        // Per-group scales with layout transformation for SIMD access
        for (int im = 0; im < m / bits; im++) {
            for (int ik = 0; ik < k; ik += group_size) {
                int idx = im * k + ik;
                
                // Extract scale and zero point
                ggml_fp16_t scale = GGML_FP32_TO_FP16(scale_ptr[idx / group_size]);
                ggml_fp16_t zero_point = zero_ptr ? 
                    GGML_FP32_TO_FP16(zero_ptr[idx / group_size]) : 
                    GGML_FP32_TO_FP16(0.0f);
                
                // Transform scale layout for SIMD-friendly access pattern
                idx         = idx / group_size;
                int nb1     = k / group_size;
                int nb0     = bm / bits * nb1;
                int new_im  = idx / nb0;
                int new_ibm = (idx % nb0) / nb1;
                int new_ik  = (idx % nb1);
                
                int new_isimd     = new_ibm % simd_n_out;
                int new_idx_outer = new_im * bm / bits * k / group_size / simd_n_out + 
                                   new_ik * bm / bits / simd_n_out +
                                   new_ibm / simd_n_out;
                int new_idx_scale = new_idx_outer * (simd_n_out * 2) + new_isimd;
                int new_idx_zero  = new_idx_outer * (simd_n_out * 2) + simd_n_out + new_isimd;
                
                scales_out[new_idx_scale] = scale;
                scales_out[new_idx_zero]  = zero_point;
            }
        }
    }
}

//> ===================================================================================================
//> NEON optimized implementations
//> ===================================================================================================

#ifdef __ARM_NEON

//> ===================================================================================================
//> NEON helper functions for bit-plane separation
//> ===================================================================================================

// Helper for 1-bit NEON bit-plane separation
static void bitplane_separation_1bit_neon(const uint8_t* qweight_ptr, uint8_t* repack_ws,
                                          int m, int k, int g) {
    memset(repack_ws, 0, m * k / g);
    
    const int simd_width = 16;  // Process 16 bytes at a time
    const int k_blocks = k / (simd_width * 8);  // Each byte has 8 1-bit values
    
    for (int im = 0; im < m; im++) {
        int ik = 0;
        
        // SIMD processing for aligned data
        for (int kb = 0; kb < k_blocks; kb++) {
            // Load 16 bytes = 128 1-bit values
            uint8x16_t qvals = vld1q_u8(qweight_ptr + (im * k) / 8 + kb * simd_width);
            
            // Store to temporary buffer for processing
            uint8_t temp_buf[16];
            vst1q_u8(temp_buf, qvals);
            
            // Process each byte
            for (int i = 0; i < simd_width; i++) {
                uint8_t byte_val = temp_buf[i];
                
                // Extract bits and group for LUT (g=4)
                for (int bit = 0; bit < 8; bit++) {
                    uint8_t v = (byte_val >> (7 - bit)) & 1;
                    int cur_ik = ik + i * 8 + bit;
                    int new_ik = cur_ik / g;
                    int shft_left = cur_ik % g;
                    repack_ws[im * k / g + new_ik] |= v << shft_left;
                }
            }
            ik += simd_width * 8;
        }
        
        // Handle remaining elements with scalar code
        for (; ik < k; ik++) {
            uint8_t v = QlutattnI1TypeAccessor::get_q(qweight_ptr, im * k + ik);
            int new_ik = ik / g;
            int shft_left = ik % g;
            repack_ws[im * k / g + new_ik] |= v << shft_left;
        }
    }
}

// Helper for 2-bit NEON bit-plane separation
static void bitplane_separation_2bit_neon(const uint8_t* qweight_ptr, uint8_t* repack_ws,
                                          int m, int k, int g) {
    memset(repack_ws, 0, m * k / g * 2);  // 2 bit-planes for 2-bit
    
    const int simd_width = 16;
    const int k_blocks = k / (simd_width * 4);  // Each byte has 4 2-bit values
    
    for (int im = 0; im < m / 2; im++) {  // 2-bit quantization
        int ik = 0;
        
        // SIMD processing
        for (int kb = 0; kb < k_blocks; kb++) {
            // Prefetch next block
            __builtin_prefetch(qweight_ptr + (im * k) / 4 + (kb + 1) * simd_width, 0, 1);
            
            // Load 16 bytes = 64 2-bit values
            uint8x16_t qvals = vld1q_u8(qweight_ptr + (im * k) / 4 + kb * simd_width);
            
            // Extract bit planes using NEON
            uint8x16_t plane0_mask = vdupq_n_u8(0x55);  // 01010101 - bits 0,2,4,6
            uint8x16_t plane1_mask = vdupq_n_u8(0xAA);  // 10101010 - bits 1,3,5,7
            
            uint8x16_t plane0_raw = vandq_u8(qvals, plane0_mask);
            uint8x16_t plane1_raw = vandq_u8(qvals, plane1_mask);
            
            // Store to temporary buffers for processing
            uint8_t p0_buf[16], p1_buf[16];
            vst1q_u8(p0_buf, plane0_raw);
            vst1q_u8(p1_buf, plane1_raw);
            
            // Process extracted values
            for (int i = 0; i < simd_width; i++) {
                uint8_t p0_byte = p0_buf[i];
                uint8_t p1_byte = p1_buf[i];
                
                // Process 4 2-bit values from this byte
                for (int j = 0; j < 4; j++) {
                    int cur_ik = ik + i * 4 + j;
                    int new_ik = cur_ik / g;
                    int shft_left = cur_ik % g;
                    
                    // Extract bit 0 from appropriate position
                    uint8_t bit0 = (p0_byte >> (j * 2)) & 1;
                    uint8_t bit1 = (p1_byte >> (j * 2 + 1)) & 1;
                    
                    repack_ws[im * 2 * k / g + 0 * k / g + new_ik] |= bit0 << shft_left;
                    repack_ws[im * 2 * k / g + 1 * k / g + new_ik] |= bit1 << shft_left;
                }
            }
            ik += simd_width * 4;
        }
        
        // Handle remaining elements
        for (; ik < k; ik++) {
            uint8_t v = QlutattnI2TypeAccessor::get_q(qweight_ptr, im * k + ik);
            int new_ik = ik / g;
            int shft_left = ik % g;
            
            for (int ib = 0; ib < 2; ib++) {
                repack_ws[im * 2 * k / g + ib * k / g + new_ik] |= ((v >> ib) & 1) << shft_left;
            }
        }
    }
}

// Helper for 4-bit NEON bit-plane separation
static void bitplane_separation_4bit_neon(const uint8_t* qweight_ptr, uint8_t* repack_ws,
                                          int m, int k, int g) {
    memset(repack_ws, 0, m * k / g * 4);  // 4 bit-planes for 4-bit
    
    const int simd_width = 16;
    const int k_blocks = k / (simd_width * 2);  // Each byte has 2 4-bit values
    
    for (int im = 0; im < m / 4; im++) {
        int ik = 0;
        
        // SIMD processing
        for (int kb = 0; kb < k_blocks; kb++) {
            // Prefetch
            __builtin_prefetch(qweight_ptr + (im * k) / 2 + (kb + 1) * simd_width, 0, 1);
            
            // Load 16 bytes = 32 4-bit values
            uint8x16_t qvals = vld1q_u8(qweight_ptr + (im * k) / 2 + kb * simd_width);
            
            // Extract nibbles
            uint8x16_t low_nibbles = vandq_u8(qvals, vdupq_n_u8(0x0F));
            uint8x16_t high_nibbles = vshrq_n_u8(qvals, 4);
            
            // Store to temporary buffers for processing
            uint8_t low_buf[16], high_buf[16];
            vst1q_u8(low_buf, low_nibbles);
            vst1q_u8(high_buf, high_nibbles);
            
            // Process nibbles
            for (int i = 0; i < simd_width; i++) {
                uint8_t low = low_buf[i];
                uint8_t high = high_buf[i];
                
                // Process low nibble
                int cur_ik = ik + i * 2;
                int new_ik = cur_ik / g;
                int shft_left = cur_ik % g;
                
                for (int ib = 0; ib < 4; ib++) {
                    repack_ws[im * 4 * k / g + ib * k / g + new_ik] |= ((high >> ib) & 1) << shft_left;
                }
                
                // Process high nibble
                cur_ik++;
                new_ik = cur_ik / g;
                shft_left = cur_ik % g;
                
                for (int ib = 0; ib < 4; ib++) {
                    repack_ws[im * 4 * k / g + ib * k / g + new_ik] |= ((low >> ib) & 1) << shft_left;
                }
            }
            ik += simd_width * 2;
        }
        
        // Handle remaining elements
        for (; ik < k; ik++) {
            uint8_t v = QlutattnI4TypeAccessor::get_q(qweight_ptr, im * k + ik);
            int new_ik = ik / g;
            int shft_left = ik % g;
            
            for (int ib = 0; ib < 4; ib++) {
                repack_ws[im * 4 * k / g + ib * k / g + new_ik] |= ((v >> ib) & 1) << shft_left;
            }
        }
    }
}

// NEON optimized SIMD permutation (shared by all bit widths)
static void simd_permutation_neon(const uint8_t* repack_ws, uint8_t* qweights_out,
                                  int m, int k, int bits, int g,
                                  const struct pack_config* cfg) {
    const int ngroups_per_elem = cfg->ngroups_per_elem;
    const int bm = cfg->bm;
    const int simd_n_in = cfg->simd_n_in;
    const int simd_n_out = cfg->simd_n_out;
    const int kfactor = cfg->kfactor;
    const int mgroup = cfg->mgroup;
    const int nelem_per_byte = cfg->nelem_per_byte;
    
    memset(qweights_out, 0, m * k / g / nelem_per_byte);
    
    // Process in blocks for better cache utilization
    const int block_size = 64;  // Process 64 elements at a time
    
    for (int im_block = 0; im_block < m / bits; im_block += block_size) {
        int im_end = (im_block + block_size < m / bits) ? im_block + block_size : m / bits;
        
        for (int im = im_block; im < im_end; im++) {
            for (int ib = 0; ib < bits; ib++) {
                // Prefetch next data
                if (ib < bits - 1) {
                    __builtin_prefetch(repack_ws + im * bits * k / g + (ib + 1) * k / g, 0, 1);
                }
                
                for (int ik = 0; ik < k / g; ik++) {
                    // Same permutation logic but with prefetching
                    int new_im = im / simd_n_out;
                    int new_isno = im % simd_n_out;
                    int new_idx = new_im * bits * simd_n_out * k / g + 
                                 ib * simd_n_out * k / g + 
                                 new_isno * k / g + ik;
                    
                    // Stage 2 & 3 permutation (same as scalar)
                    int nb2 = k / g;
                    int nb1 = simd_n_in * nb2;
                    int nb0 = ngroups_per_elem * nb1;
                    new_im = new_idx / nb0;
                    int new_ing = (new_idx % nb0) / nb1;
                    int new_isni = (new_idx % nb1) / nb2;
                    int new_ik = (new_idx % nb2);
                    new_idx = new_im * ngroups_per_elem * simd_n_in * k / g + 
                             new_isni * ngroups_per_elem * k / g +
                             new_ing * k / g + new_ik;
                    
                    int nb4 = kfactor;
                    int nb3 = k / g / kfactor * nb4;
                    nb2 = ngroups_per_elem * nb3;
                    nb1 = simd_n_in * nb2;
                    nb0 = bm / mgroup * nb1;
                    new_im = new_idx / nb0;
                    int new_ibm = (new_idx % nb0) / nb1;
                    new_isni = (new_idx % nb1) / nb2;
                    new_ing = (new_idx % nb2) / nb3;
                    new_ik = (new_idx % nb3) / nb4;
                    int new_ikf = (new_idx % nb4);
                    new_idx = new_im * k / g / kfactor * bm / mgroup * kfactor * simd_n_in * ngroups_per_elem +
                             new_ik * bm / mgroup * kfactor * simd_n_in * ngroups_per_elem +
                             new_ibm * kfactor * simd_n_in * ngroups_per_elem + 
                             new_ikf * simd_n_in * ngroups_per_elem +
                             new_isni * ngroups_per_elem + new_ing;
                    new_idx = new_idx / ngroups_per_elem;
                    
                    qweights_out[new_idx] += repack_ws[im * bits * k / g + ib * k / g + ik] << (new_ing * g);
                }
            }
        }
    }
}

void pack_weights_1bit_neon(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                            int m, int k, const struct pack_config* cfg) {
    const int bits = 1;
    const int g = cfg->g;
    
    // Stage 1: NEON optimized bit-plane separation
    bitplane_separation_1bit_neon(src, workspace, m, k, g);
    
    // Stage 2: NEON optimized SIMD layout permutation
    simd_permutation_neon(workspace, dst, m, k, bits, g, cfg);
}

void pack_weights_2bit_neon(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                            int m, int k, const struct pack_config* cfg) {
    // For now, fallback to scalar version for 2-bit due to complex bit extraction
    // TODO: Optimize 2-bit NEON implementation
    pack_weights_2bit_scalar(src, dst, workspace, m, k, cfg);
}

void pack_weights_4bit_neon(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                            int m, int k, const struct pack_config* cfg) {
    const int bits = 4;
    const int g = cfg->g;
    
    // Stage 1: NEON optimized bit-plane separation
    bitplane_separation_4bit_neon(src, workspace, m, k, g);
    
    // Stage 2: NEON optimized SIMD layout permutation
    simd_permutation_neon(workspace, dst, m, k, bits, g, cfg);
}

void pack_scales_neon(const float* scale_ptr, const float* zero_ptr,
                      ggml_fp16_t* scales_out, int m, int k, int bits,
                      int group_size, int scales_size,
                      const struct pack_config* cfg) {
    const int bm = cfg->bm;
    const int simd_n_out = cfg->simd_n_out;
    
    if (scales_size < m / bits) {
        // BitNet-like scale - just convert to fp16
        for (int i = 0; i < scales_size; i++) {
            scales_out[i] = GGML_FP32_TO_FP16(scale_ptr[i]);
        }
    } else {
        // Per-group scales with NEON optimization
        const int total_groups = (m / bits) * (k / group_size);
        int i = 0;
        
        // Process 8 scales at a time using NEON
        for (; i + 8 <= total_groups; i += 8) {
            // Load 8 float scales
            float32x4_t scales_low = vld1q_f32(scale_ptr + i);
            float32x4_t scales_high = vld1q_f32(scale_ptr + i + 4);
            
            // Load 8 float zero points if available
            float32x4_t zeros_low, zeros_high;
            if (zero_ptr) {
                zeros_low = vld1q_f32(zero_ptr + i);
                zeros_high = vld1q_f32(zero_ptr + i + 4);
            } else {
                zeros_low = vdupq_n_f32(0.0f);
                zeros_high = vdupq_n_f32(0.0f);
            }
            
            // Store to temporary buffers for processing
            float scales_low_buf[4], scales_high_buf[4];
            float zeros_low_buf[4], zeros_high_buf[4];
            vst1q_f32(scales_low_buf, scales_low);
            vst1q_f32(scales_high_buf, scales_high);
            if (zero_ptr) {
                vst1q_f32(zeros_low_buf, zeros_low);
                vst1q_f32(zeros_high_buf, zeros_high);
            }
            
            // Convert to fp16 and transform layout
            for (int j = 0; j < 4; j++) {
                int idx = i + j;
                int im = idx / (k / group_size);
                int ik_group = idx % (k / group_size);
                
                // Calculate transformed indices
                int nb1 = k / group_size;
                int nb0 = bm / bits * nb1;
                int new_im = im / nb0;
                int new_ibm = (im % nb0) / nb1;
                int new_ik = ik_group;
                
                int new_isimd = new_ibm % simd_n_out;
                int new_idx_outer = new_im * bm / bits * k / group_size / simd_n_out + 
                                   new_ik * bm / bits / simd_n_out +
                                   new_ibm / simd_n_out;
                int new_idx_scale = new_idx_outer * (simd_n_out * 2) + new_isimd;
                int new_idx_zero = new_idx_outer * (simd_n_out * 2) + simd_n_out + new_isimd;
                
                scales_out[new_idx_scale] = GGML_FP32_TO_FP16(scales_low_buf[j]);
                scales_out[new_idx_zero] = zero_ptr ? 
                    GGML_FP32_TO_FP16(zeros_low_buf[j]) : 
                    GGML_FP32_TO_FP16(0.0f);
            }
            
            for (int j = 0; j < 4; j++) {
                int idx = i + 4 + j;
                int im = idx / (k / group_size);
                int ik_group = idx % (k / group_size);
                
                // Calculate transformed indices
                int nb1 = k / group_size;
                int nb0 = bm / bits * nb1;
                int new_im = im / nb0;
                int new_ibm = (im % nb0) / nb1;
                int new_ik = ik_group;
                
                int new_isimd = new_ibm % simd_n_out;
                int new_idx_outer = new_im * bm / bits * k / group_size / simd_n_out + 
                                   new_ik * bm / bits / simd_n_out +
                                   new_ibm / simd_n_out;
                int new_idx_scale = new_idx_outer * (simd_n_out * 2) + new_isimd;
                int new_idx_zero = new_idx_outer * (simd_n_out * 2) + simd_n_out + new_isimd;
                
                scales_out[new_idx_scale] = GGML_FP32_TO_FP16(scales_high_buf[j]);
                scales_out[new_idx_zero] = zero_ptr ? 
                    GGML_FP32_TO_FP16(zeros_high_buf[j]) : 
                    GGML_FP32_TO_FP16(0.0f);
            }
        }
        
        // Handle remaining scales with scalar code
        for (; i < total_groups; i++) {
            int im = i / (k / group_size);
            int ik_group = i % (k / group_size);
            
            ggml_fp16_t scale = GGML_FP32_TO_FP16(scale_ptr[i]);
            ggml_fp16_t zero_point = zero_ptr ? 
                GGML_FP32_TO_FP16(zero_ptr[i]) : 
                GGML_FP32_TO_FP16(0.0f);
            
            // Transform layout
            int nb1 = k / group_size;
            int nb0 = bm / bits * nb1;
            int new_im = im / nb0;
            int new_ibm = (im % nb0) / nb1;
            int new_ik = ik_group;
            
            int new_isimd = new_ibm % simd_n_out;
            int new_idx_outer = new_im * bm / bits * k / group_size / simd_n_out + 
                               new_ik * bm / bits / simd_n_out +
                               new_ibm / simd_n_out;
            int new_idx_scale = new_idx_outer * (simd_n_out * 2) + new_isimd;
            int new_idx_zero = new_idx_outer * (simd_n_out * 2) + simd_n_out + new_isimd;
            
            scales_out[new_idx_scale] = scale;
            scales_out[new_idx_zero] = zero_point;
        }
    }
}

#endif // __ARM_NEON

//> ===================================================================================================
//> Batch processing - Reserved for future implementation
//> ===================================================================================================

// TODO: Batch processing implementation
// The batch processing functions are temporarily disabled due to integration 
// complexity with llama.cpp's thread pool. 
//
// Key implementation notes for future:
// 1. Use llama.cpp's native thread pool (ggml_graph_compute_thread) instead of OpenMP
// 2. Implement lock-free memory pool for thread-safe workspace allocation
// 3. Integrate with ggml_backend buffer management for memory efficiency
// 4. Add proper synchronization with existing parallel graph execution
//
// Expected optimizations:
// - Memory pooling to reduce allocation overhead by 30-50%
// - Cache-aware chunking for 20-30% better data locality
// - Parallel processing potential for 2-4x speedup on multi-core systems
//
// Current workaround:
// Process tensors individually with separate workspace allocations.
// This is less efficient but avoids thread coordination issues.

//> ===================================================================================================
//> Memory optimization implementations
//> ===================================================================================================

void pack_memcpy_optimized(void* dst, const void* src, size_t size) {
    assert(dst != NULL && src != NULL);
    
    const uint8_t* src8 = (const uint8_t*)src;
    uint8_t* dst8 = (uint8_t*)dst;
    
    // Use NEON for aligned copies
#ifdef __ARM_NEON
    if (((uintptr_t)src8 & 15) == 0 && ((uintptr_t)dst8 & 15) == 0) {
        size_t simd_size = size & ~15;  // Round down to multiple of 16
        
        for (size_t i = 0; i < simd_size; i += 64) {
            // Prefetch next cache line
            __builtin_prefetch(src8 + i + 64, 0, 1);
            
            // Load and store 64 bytes at a time
            uint8x16_t v0 = vld1q_u8(src8 + i);
            uint8x16_t v1 = vld1q_u8(src8 + i + 16);
            uint8x16_t v2 = vld1q_u8(src8 + i + 32);
            uint8x16_t v3 = vld1q_u8(src8 + i + 48);
            
            vst1q_u8(dst8 + i, v0);
            vst1q_u8(dst8 + i + 16, v1);
            vst1q_u8(dst8 + i + 32, v2);
            vst1q_u8(dst8 + i + 48, v3);
        }
        
        // Copy remaining bytes
        for (size_t i = simd_size; i < size; i++) {
            dst8[i] = src8[i];
        }
    } else
#endif
    {
        // Fallback to standard memcpy
        memcpy(dst, src, size);
    }
}

void pack_reorder_for_cache(uint8_t* data, int m, int k, int bits,
                            int cache_line_size) {
    assert(data != NULL);
    
    const int bytes_per_line = cache_line_size;
    const int elements_per_byte = 8 / bits;
    const int elements_per_line = bytes_per_line * elements_per_byte;
    
    // Temporary buffer for reordering
    size_t data_size = (m * k) / elements_per_byte;
    uint8_t* temp = (uint8_t*)malloc(data_size);
    if (temp == NULL) return;  // Skip reordering if allocation fails
    
    memcpy(temp, data, data_size);
    
    // Reorder data in cache-line-sized blocks
    int idx = 0;
    for (int block_m = 0; block_m < m; block_m += elements_per_line) {
        for (int block_k = 0; block_k < k; block_k += elements_per_line) {
            // Copy block from temp to data
            for (int im = block_m; im < block_m + elements_per_line && im < m; im++) {
                for (int ik = block_k; ik < block_k + elements_per_line && ik < k; ik++) {
                    int src_idx = (im * k + ik) / elements_per_byte;
                    data[idx++] = temp[src_idx];
                }
            }
        }
    }
    
    free(temp);
}

//> ===================================================================================================
//> Memory pool implementations
//> ===================================================================================================

void pack_memory_pool_init(struct pack_memory_pool* pool,
                           uint8_t* buffer, size_t size, size_t alignment) {
    assert(pool != NULL && buffer != NULL);
    
    pool->base = buffer;
    pool->size = size;
    pool->used = 0;
    pool->alignment = alignment > 0 ? alignment : 16;
}

void* pack_memory_pool_alloc(struct pack_memory_pool* pool, size_t size) {
    assert(pool != NULL);
    
    // Align size to pool alignment
    size = (size + pool->alignment - 1) & ~(pool->alignment - 1);
    
    if (pool->used + size > pool->size) {
        return NULL;  // Pool exhausted
    }
    
    void* ptr = pool->base + pool->used;
    pool->used += size;
    
    return ptr;
}

void pack_memory_pool_reset(struct pack_memory_pool* pool) {
    assert(pool != NULL);
    pool->used = 0;
}