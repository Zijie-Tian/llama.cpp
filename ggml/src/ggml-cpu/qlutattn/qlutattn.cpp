#include "qlutattn.h"

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "lut_ctor.h"

#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"

/****** T-MAC helper functions ******/
static inline bool is_qlutattn_2bit_type(enum ggml_type type) {
    return (type == GGML_TYPE_QLUTATTN_KV2_128x128);
}

static inline bool is_qlutattn_4bit_type(enum ggml_type type) {
    return (type == GGML_TYPE_QLUTATTN_KV4_128x128);
}

bool is_qlutattn_type(enum ggml_type type) {
    return (type == GGML_TYPE_QLUTATTN_KV1_128x128 || type == GGML_TYPE_QLUTATTN_KV2_128x128 ||
            type == GGML_TYPE_QLUTATTN_KV4_128x128);
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

void ggml_vec_dot_qlutattn_kv4_128x128(int n, float * GGML_RESTRICT s, size_t bs, const uint8_t * GGML_RESTRICT x,
                                       size_t bx, const ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
    for (int i = 0; i < bs; i++) {
        s[i] = 1.0f;  // Replace with actual computation
    }
}
