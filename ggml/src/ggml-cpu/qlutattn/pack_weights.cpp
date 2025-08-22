#include "pack_weights.h"
#include <string.h>
#include <assert.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
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

void pack_weights_1bit_neon(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                            int m, int k, const struct pack_config* cfg) {
    // TODO: Implement NEON optimized version
    // For now, fall back to scalar implementation
    pack_weights_1bit_scalar(src, dst, workspace, m, k, cfg);
}

void pack_weights_2bit_neon(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                            int m, int k, const struct pack_config* cfg) {
    // TODO: Implement NEON optimized version
    // For now, fall back to scalar implementation
    pack_weights_2bit_scalar(src, dst, workspace, m, k, cfg);
}

void pack_weights_4bit_neon(const uint8_t* src, uint8_t* dst, uint8_t* workspace,
                            int m, int k, const struct pack_config* cfg) {
    // TODO: Implement NEON optimized version
    // For now, fall back to scalar implementation
    pack_weights_4bit_scalar(src, dst, workspace, m, k, cfg);
}

void pack_scales_neon(const float* scale_ptr, const float* zero_ptr,
                      ggml_fp16_t* scales_out, int m, int k, int bits,
                      int group_size, int scales_size,
                      const struct pack_config* cfg) {
    // TODO: Implement NEON optimized version
    // For now, fall back to scalar implementation
    pack_scales_scalar(scale_ptr, zero_ptr, scales_out, m, k, bits,
                      group_size, scales_size, cfg);
}

#endif // __ARM_NEON