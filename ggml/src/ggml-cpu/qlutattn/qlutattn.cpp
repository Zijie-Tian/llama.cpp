#include "qlutattn.h"
#include "qlutattn-config.h"

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
    // Check for 1-bit types
    if (type == GGML_TYPE_QLUTATTN_K1_128x128 || type == GGML_TYPE_QLUTATTN_V1_128x128) {
        return 1;
    }
    // Check for 2-bit types
    else if (is_qlutattn_2bit_type(type) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) {
        return 2;
    }
    // Check for 4-bit types
    else if (is_qlutattn_4bit_type(type) || type == GGML_TYPE_Q4_0) {
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
//> QLUTATTN kernel config management
//> ===================================================================================================


//> ===================================================================================================
//> QLUTATTN MUL_MAT task init and compute
//> ===================================================================================================

// m = batch_size
// n = output_dim
// t-mac llama.cpp n and m swapped
void ggml_qlutattn_mul_mat_task_init(void * src1, void * qlut, void * lut_scales, void * lut_biases, int n, int k,
                                     int m, int bits) {
    // Initialize config system if needed
    if (!ggml_qlutattn_config_is_initialized()) {
        ggml_qlutattn_config_init();
    }
    
    const struct qlutattn_kernel_config * kernel_config = ggml_qlutattn_get_config(n, k, bits);
    if (kernel_config == nullptr) {
        throw std::runtime_error("ggml_tmac_mul_mat_task_init: Failed to find kernel config for m" + std::to_string(n) +
                                 "_k" + std::to_string(k) + "_b" + std::to_string(bits));
    }
    ggml::cpu::qlutattn::qlutattn_lut_ctor_int8_g4(src1, lut_scales, lut_biases, qlut, k, kernel_config);
}

void ggml_qlutattn_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases,
                                        void * dst, int n, int k, int m, int bits) {
    // Initialize config system if needed
    if (!ggml_qlutattn_config_is_initialized()) {
        ggml_qlutattn_config_init();
    }
    
    const struct qlutattn_kernel_config * kernel_config = ggml_qlutattn_get_config(n, k, bits);
    if (kernel_config == nullptr) {
        throw std::runtime_error("ggml_qlutattn_mul_mat_task_compute: Failed to find kernel config for m" +
                                 std::to_string(n) + "_k" + std::to_string(k) + "_b" + std::to_string(bits));
    }
    ggml::cpu::qlutattn::qgemm_lut_int8_g4(src0, qlut, scales, lut_scales, lut_biases, dst, kernel_config->bm, k, m,
                                           kernel_config);
}

void ggml_vec_dot_qlutattn_kv1_128x128(int n, ggml_fp16_t * GGML_RESTRICT C, size_t bs, const uint8_t * GGML_RESTRICT x,
                                       size_t bx, const ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
    GGML_ASSERT(bx % 128 == 0 && by % 128 == 0 && "Must be multiple of 128 for QLUTATTN KV1 128x128");

    // NOTE: This function is for 1-bit QLUTATTN KV.
    // Initialize config system if needed
    if (!ggml_qlutattn_config_is_initialized()) {
        ggml_qlutattn_config_init();
    }
    const struct qlutattn_kernel_config * kernel_config = ggml_qlutattn_get_config(n, nrc, 1);
    if (kernel_config == nullptr) {
        GGML_LOG_ERROR("Failed to find kernel config for KV1 n=%d nrc=%d\n", n, nrc);
        return;
    }

    int bits           = kernel_config->bits;
    int bm             = kernel_config->bm;
    int act_group_size = kernel_config->act_group_size;
    int K              = QLUTATTN_PACK_SIZE;
    int M              = QLUTATTN_PACK_CHUNK_SIZE;
    int N              = 1;

    int nelems_per_byte = 8 / bits;  // > 1-bit QLUTATTN KV, so 8 / 1 = 8.

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
    int8_t *          qweights = (int8_t *) x;                                                // Packed KV cache
    tmac_float_type * Scales =
        (tmac_float_type *) ((uint8_t *) x + 128 * 128 / nelems_per_byte * sizeof(uint8_t));  // Scales for A

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

void ggml_vec_dot_qlutattn_kv2_128x128(int n, ggml_fp16_t * GGML_RESTRICT C, size_t bs, const uint8_t * GGML_RESTRICT x,
                                       size_t bx, const ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
    GGML_ASSERT(bx % 128 == 0 && by % 128 == 0 && "Must be multiple of 128 for QLUTATTN KV2 128x128");

    // NOTE: This function is for 2-bit QLUTATTN KV.
    // Initialize config system if needed
    if (!ggml_qlutattn_config_is_initialized()) {
        ggml_qlutattn_config_init();
    }
    const struct qlutattn_kernel_config * kernel_config = ggml_qlutattn_get_config(n, nrc, 2);
    if (kernel_config == nullptr) {
        GGML_LOG_ERROR("Failed to find kernel config for KV2 n=%d nrc=%d\n", n, nrc);
        return;
    }

    int bits           = kernel_config->bits;
    int bm             = kernel_config->bm;
    int act_group_size = kernel_config->act_group_size;
    int K              = QLUTATTN_PACK_SIZE;
    int M              = QLUTATTN_PACK_CHUNK_SIZE;
    int N              = 1;

    int nelems_per_byte = 8 / bits;  // 2-bit quantization means 4 elements per byte.

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
    int8_t *          qweights = (int8_t *) x;                                                // Packed KV cache
    tmac_float_type * Scales =
        (tmac_float_type *) ((uint8_t *) x + 128 * 128 / nelems_per_byte * sizeof(uint8_t));  // Scales for A

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

void ggml_vec_dot_qlutattn_kv4_128x128(int n, ggml_fp16_t * GGML_RESTRICT C, size_t bs, const uint8_t * GGML_RESTRICT x,
                                       size_t bx, const ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
    GGML_ASSERT(bx % 128 == 0 && by % 128 == 0 && "Must be multiple of 128 for QLUTATTN KV4 128x128");
    // Initialize config system if needed
    if (!ggml_qlutattn_config_is_initialized()) {
        ggml_qlutattn_config_init();
    }
    const struct qlutattn_kernel_config * kernel_config = ggml_qlutattn_get_config(n, nrc, 4);
    if (kernel_config == nullptr) {
        GGML_LOG_ERROR("Failed to find kernel config for KV4 n=%d nrc=%d\n", n, nrc);
        return;
    }

    int bits           = kernel_config->bits;
    int bm             = kernel_config->bm;
    int act_group_size = kernel_config->act_group_size;
    int K              = QLUTATTN_PACK_SIZE;
    int M              = QLUTATTN_PACK_CHUNK_SIZE;
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
