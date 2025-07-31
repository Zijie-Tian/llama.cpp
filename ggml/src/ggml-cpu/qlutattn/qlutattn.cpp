#include "qlutattn.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "qlut_ctor.h"
#include "tbl.h"

#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"

//> ===================================================================================================
//> Helper functions for QLUTATTN
//> ===================================================================================================
static inline bool is_qlutattn_2bit_type(enum ggml_type type) {
    return (type == GGML_TYPE_QLUTATTN_K2_128x128);
}

static inline bool is_qlutattn_4bit_type(enum ggml_type type) {
    return (type == GGML_TYPE_QLUTATTN_K4_128x128);
}

bool is_qlutattn_type(enum ggml_type type) {
    return (type == GGML_TYPE_QLUTATTN_K1_128x128 || type == GGML_TYPE_QLUTATTN_K2_128x128 ||
            type == GGML_TYPE_QLUTATTN_K4_128x128);
}

bool is_type_supported(enum ggml_type type) {
    return is_qlutattn_type(type);
}

bool ggml_qlutattn_can_mul_mat(const struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    if (dst->op == GGML_OP_MUL_MAT && (is_type_supported(src0->type)) && src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 && strcmp(src0->name, "token_embd.weight") &&  // means not equal
        strcmp(src0->name, "output.weight")) {
        return true;
    }
    return false;
}

static inline int get_type_bits(enum ggml_type type) {
    if (is_qlutattn_2bit_type(type) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) {
        return 2;
    } else if (is_qlutattn_4bit_type(type) || type == GGML_TYPE_Q4_0) {
        return 4;
    } else {
        return 0;
    }
}

static inline int get_type_group_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_QLUTATTN_K1_128x128:
        case GGML_TYPE_QLUTATTN_K2_128x128:
        case GGML_TYPE_QLUTATTN_K4_128x128:
            return 128;
        default:
            return -1;  // Unsupported type
    }
}

static inline int ggml_qlutattn_get_scales_size(const struct qlutattn_kernel_config * kernel_config, int m, int k) {
    int scales_size;
    if (kernel_config->one_scale) {
        scales_size = 1;
    } else if (kernel_config->has_zero_point) {
        scales_size = m * k / kernel_config->q_group_size * 2;
    } else {
        scales_size = m * k / kernel_config->q_group_size;
    }
    return scales_size;
}

static inline bool get_type_has_zero_point(enum ggml_type type) {
    // Currently, all QLUTATTN types do not have zero point.
    return false;
}

static inline bool get_type_is_one_scale(enum ggml_type type) {
    // Currently, all QLUTATTN types have one scale.
    return true;
}

//> ===================================================================================================
//> T-MAC meta model info
//> ===================================================================================================

// static void init_tmac_kernel_config_from_tensor_type(enum ggml_type type, int M,
//                                                      struct qlutattn_kernel_config * kernel_config) {
//     kernel_config->bits           = get_type_bits(type);
//     kernel_config->q_group_size   = get_type_group_size(type);
//     kernel_config->has_zero_point = get_type_has_zero_point(type);
//     kernel_config->one_scale      = get_type_is_one_scale(type);
//
//     // Fixed features
//     kernel_config->has_scale        = true;
//     kernel_config->g                = 4;
//     kernel_config->ngroups_per_elem = 8 / kernel_config->g;
//
//     // Decide q_group_size for BN_0
//     if (kernel_config->q_group_size == -1) {
//         if (M % 256 == 0) {
//             kernel_config->q_group_size = 64;
//         } else if (M % 128 == 0) {
//             kernel_config->q_group_size = 64;
//         } else if (M % 64 == 0) {
//             kernel_config->q_group_size = 64;
//         } else if (M % 32 == 0) {
//             kernel_config->q_group_size = 32;
//         } else {
//             GGML_LOG_ERROR(
//                 "Unsupported M value. Expected multiple of 32, got %d. Please check all of the model weight shapes.\n",
//                 M);
//         }
//     }
//
//     if (kernel_config->q_group_size % 64 == 0) {
//         kernel_config->act_group_size = 64;
//     } else if (kernel_config->q_group_size % 32 == 0) {
//         kernel_config->act_group_size = 32;
//     } else {
//         GGML_LOG_ERROR("Unsupported activation group size: %d\n", kernel_config->q_group_size);
//     }
//     kernel_config->actk = kernel_config->act_group_size / kernel_config->g;
//
//     // kfactor to be tuned
//     // bm to be tuned
//     kernel_config->simd_n_in  = 16;
//     kernel_config->simd_n_out = 8;
//
//     kernel_config->chunk_n = 8;
// }
//
//> ===================================================================================================
//> QLUTATTN kernel config management
//> ===================================================================================================

static std::unordered_map<std::string, struct qlutattn_kernel_config> qlutattn_kernel_config;

static std::string ggml_vec_dot_qlutattn_kv4_128x128_kernel_config_key(int M, int K, int bits) {
    return "M" + std::to_string(M) + "_K" + std::to_string(K) + "_b" + std::to_string(bits);
}

// struct tmac_kernel_config * find_qlutattn_128x128_kernel_config(int M, int K, int bits) {
//     std::string key = get_tmac_kernel_config_key(M, K, bits);
//     if (final_tmac_kernel_config.count(key) == 0) {
//         return nullptr;
//     }
//     return &final_tmac_kernel_config[key];
// }

struct qlutattn_kernel_config * find_qlutattn_128x128_kernel_config(int M, int K, int bits) {
    if (qlutattn_kernel_config.count("test") == 0) {
        struct qlutattn_kernel_config kernel_config{
            .g                = 4,
            .ngroups_per_elem = 2,  // NOTE: Must 8 // g, in tmac Kernel g is set to 4.
            .q_group_size     = 128,
            .act_group_size   = 64,
            .has_scale        = true,
            .kfactor          = 16,
            .bits             = bits,
            .actk             = 16,  // should be equal to (act_group_size / g).
            .has_zero_point   = true,
            .one_scale        = false,
            .bm               = 128 * bits,
            .simd_n_in        = 16,
            .simd_n_out       = 8,
            .chunk_n          = 8  // useless for QLUTATTN.
        };

        qlutattn_kernel_config["test"] = kernel_config;
    }

    return &qlutattn_kernel_config["test"];
}

static void insert_or_assign_qlutattn_kv4_128x128_kernel_config(int M, int K, int bits,
                                                                struct qlutattn_kernel_config kernel_config) {
    std::string key = ggml_vec_dot_qlutattn_kv4_128x128_kernel_config_key(M, K, bits);
    qlutattn_kernel_config.insert_or_assign(key, kernel_config);
}

//
// static inline void ggml_tmac_forward_mul_mat(void * A, void * B, void * C, void * QLUT, void * LUT_Scales,
//                                              void * LUT_Biases, void * Scales, int M, int N, int K,
//                                              const struct tmac_kernel_config * kernel_config) {
//     // Currently, scale is a must.
//     assert(kernel_config->has_scale);
//     // Currently, one_scale and has_zero_point are mutually exclusive.
//     assert(!(kernel_config->one_scale && kernel_config->has_zero_point));
//
//     int bits           = kernel_config->bits;
//     int bm             = kernel_config->bm;
//     int act_group_size = kernel_config->act_group_size;
//
//     lut_ctor_int8_g4(B, LUT_Scales, LUT_Biases, QLUT, K, kernel_config);
//
//     const int     m           = bm / bits;
//     const int64_t chunk_size0 = m;
//
//     for (int32_t chunk_outer = 0; chunk_outer < M / m; chunk_outer++) {
//         /* One Block */
//         const int64_t w_offset = chunk_outer * m * K * bits / 8;
//         const int64_t scales_offset =
//             kernel_config->one_scale ? 0 : ggml_tmac_get_scales_size(kernel_config, m, K) * chunk_outer;
//
//         for (int32_t n_outer = 0; n_outer < N; n_outer++) {
//             const int64_t qlut_offset       = K * n_outer * 4;
//             const int64_t lut_scales_offset = K / act_group_size * n_outer;
//             const int64_t dst_offset        = M * n_outer + chunk_outer * chunk_size0;
//
//             int8_t *          lut        = (int8_t *) QLUT + qlut_offset;
//             uint8_t *         a          = (uint8_t *) A + w_offset;
//             tmac_float_type * scales     = (tmac_float_type *) Scales + scales_offset;
//             tmac_float_type * lut_scales = (tmac_float_type *) LUT_Scales + lut_scales_offset;
//             tmac_float_type * lut_biases = (tmac_float_type *) LUT_Biases + lut_scales_offset;
//             tmac_float_type * act_output = (tmac_float_type *) C + dst_offset;
//
//             qgemm_lut_int8_g4(a, lut, scales, lut_scales, lut_biases, act_output, bm, K, N, kernel_config);
//         }
//         /* One Block */
//     }
// }

//> ===================================================================================================
//> QLUTATTN MUL_MAT task init and compute
//> ===================================================================================================

// m = batch_size
// n = output_dim
// t-mac llama.cpp n and m swapped
void ggml_qlutattn_mul_mat_task_init(void * src1, void * qlut, void * lut_scales, void * lut_biases, int n, int k,
                                     int m, int bits) {
    struct qlutattn_kernel_config * kernel_config = find_qlutattn_128x128_kernel_config(n, k, bits);
    if (kernel_config == nullptr) {
        throw std::runtime_error("ggml_tmac_mul_mat_task_init: Failed to find kernel config for m" + std::to_string(n) +
                                 "_k" + std::to_string(k) + "_b" + std::to_string(bits));
    }
    ggml::cpu::qlutattn::qlutattn_lut_ctor_int8_g4(src1, lut_scales, lut_biases, qlut, k, kernel_config);
}

void ggml_qlutattn_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases,
                                        void * dst, int n, int k, int m, int bits) {
    struct qlutattn_kernel_config * kernel_config = find_qlutattn_128x128_kernel_config(n, k, bits);
    if (kernel_config == nullptr) {
        GGML_LOG_INFO("Failed to find kernel config for m%d_k%d_b%d\n", n, k, bits);
        throw std::runtime_error("ggml_qlutattn_mul_mat_task_compute: Failed to find kernel config for m" +
                                 std::to_string(n) + "_k" + std::to_string(k) + "_b" + std::to_string(bits));
    }
    ggml::cpu::qlutattn::qgemm_lut_int8_g4(src0, qlut, scales, lut_scales, lut_biases, dst, kernel_config->bm, k, m,
                                           kernel_config);
}

void ggml_vec_dot_qlutattn_kv4_128x128(int n, ggml_fp16_t * GGML_RESTRICT C, size_t bs, const uint8_t * GGML_RESTRICT x,
                                       size_t bx, const ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
    GGML_ASSERT(bx % 128 == 0 && by % 128 == 0 && "Must be multiple of 128 for QLUTATTN KV4 128x128");
    struct qlutattn_kernel_config * kernel_config = find_qlutattn_128x128_kernel_config(n, nrc, 4);

    int bits           = kernel_config->bits;
    int bm             = kernel_config->bm;
    int act_group_size = kernel_config->act_group_size;
    int K              = 128;
    int M              = 128;
    int N              = 1;

    // TODO: Add vec_dot of LUT inside this function.
    assert(kernel_config->has_scale);
    assert(!(kernel_config->one_scale && kernel_config->has_zero_point));

    // NOTE: Following are extract the pointers from the y tensor.
    int8_t *          QLUT       = (int8_t *) y;                                       // Quantized  LUT
    tmac_float_type * LUT_Scales = (tmac_float_type *) ((uint8_t *) y + by / 4 * 16);  // Scales for LUT_Scales
    tmac_float_type * LUT_Biases =
        (tmac_float_type *) ((uint8_t *) y + by / 4 * 16 +
                             by / act_group_size * sizeof(tmac_float_type));  // Biases for LUT_Biases

    // NOTE: Extract from x.
    int8_t *          qweights = (int8_t *) x;  // Packed KV cache
    tmac_float_type * Scales   = (tmac_float_type *) ((uint8_t *) x + 128 * 128 / 2 * sizeof(uint8_t));  // Scales for A

    assert(QLUT != nullptr && LUT_Scales != nullptr && LUT_Biases != nullptr);
    assert(C != nullptr && "Output tensor s must not be null");

    const int     m           = bm / bits;
    const int64_t chunk_size0 = m;

    for (int32_t chunk_outer = 0; chunk_outer < M / m; chunk_outer++) {
        /* One Block */
        const int64_t w_offset = chunk_outer * m * K * bits / 8;
        const int64_t scales_offset =
            kernel_config->one_scale ? 0 : ggml_qlutattn_get_scales_size(kernel_config, m, K) * chunk_outer;

        for (int32_t n_outer = 0; n_outer < N; n_outer++) {
            const int64_t qlut_offset       = K * n_outer * 4;
            const int64_t lut_scales_offset = K / act_group_size * n_outer;
            const int64_t dst_offset        = M * n_outer + chunk_outer * chunk_size0;

            int8_t *          lut        = (int8_t *) QLUT + qlut_offset;
            uint8_t *         a          = (uint8_t *) qweights + w_offset;
            tmac_float_type * scales     = (tmac_float_type *) Scales + scales_offset;
            tmac_float_type * lut_scales = (tmac_float_type *) LUT_Scales + lut_scales_offset;
            tmac_float_type * lut_biases = (tmac_float_type *) LUT_Biases + lut_scales_offset;
            tmac_float_type * act_output = (tmac_float_type *) C + dst_offset;

            ggml::cpu::qlutattn::qgemm_lut_int8_g4(a, lut, scales, lut_scales, lut_biases, act_output, bm, K, N,
                                                   kernel_config);
        }
        /* One Block */
    }
}
