#include <algorithm>
#include <string>

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-traits.h"
#include "qlutattn.h"

// #define GGML_USE_QLUTATTN
#if defined(GGML_USE_QLUTATTN)

namespace ggml::cpu::qlutattn {

class tensor_traits : public ggml::cpu::tensor_traits {
public:
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        size = 0;
        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op == GGML_OP_FLASH_ATTN) {
            return forward_flash_attn(params, op);
        }
        return false;
    }

private:
    bool forward_flash_attn(struct ggml_compute_params * params, struct ggml_tensor * op) {
        // 这里是占位符 - 实际实现不需要完成
        // 只需要捕获FLASH_ATTN算子，且输入KV类型为Q4_0的即可
        printf("QLUTATTN: Captured FLASH_ATTN operation\n");
        return true;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    static tensor_traits traits;
    return &traits;
}

// 仅支持Q4_0类型
static bool is_type_supported(ggml_type type) {
    return type == GGML_TYPE_Q4_0;
}

// 检查操作是否可以被qlutattn处理
static bool ggml_qlutattn_can_flash_attn(const struct ggml_tensor * op) {
    if (op->op != GGML_OP_FLASH_ATTN) {
        return false;
    }

    // 检查输入是否为Q4_0类型
    // 假设第一个输入是query，第二个是key，第三个是value
    if (op->n_tasks >= 2 && op->src[1] != nullptr && op->src[1]->type != GGML_TYPE_Q4_0) {
        return false;
    }
    
    if (op->n_tasks >= 3 && op->src[2] != nullptr && op->src[2]->type != GGML_TYPE_Q4_0) {
        return false;
    }

    return true;
}

class extra_buffer_type : public ggml::cpu::extra_buffer_type {
public:
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        return ggml_qlutattn_can_flash_attn(op);
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_FLASH_ATTN && 
            op->src[0]->buffer && 
            op->src[0]->buffer->buft == ggml_backend_qlutattn_buffer_type()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }
        return nullptr;
    }
};

} // namespace ggml::cpu::qlutattn

// 初始化函数
void ggml_qlutattn_init() {
    // 简化的初始化函数
}

// 缓冲区管理函数
static void ggml_backend_qlutattn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_aligned_free(buffer->context, buffer->size);
}

static void * ggml_backend_qlutattn_buffer_get_base(ggml_backend_buffer_t buffer) {
    uintptr_t data = (uintptr_t)buffer->context;
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }
    return (void *)data;
}

static enum ggml_status ggml_backend_qlutattn_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::qlutattn::get_tensor_traits(buffer, tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_qlutattn_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                   const void * data, size_t offset, size_t size) {
    // 对于所有类型，直接复制数据
    memcpy((char *) tensor->data + offset, data, size);
}

static void ggml_backend_qlutattn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static void ggml_backend_qlutattn_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                     uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);
}

// Buffer interface定义
static ggml_backend_buffer_i ggml_backend_qlutattn_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_qlutattn_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_qlutattn_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_qlutattn_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_qlutattn_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_qlutattn_buffer_clear,
    /* .memset_tensor   = */ ggml_backend_qlutattn_buffer_memset_tensor,
    /* .reset           = */ nullptr,
};

// Buffer type 函数
static const char * ggml_backend_qlutattn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "QLUTATTN";
}

static ggml_backend_buffer_t ggml_backend_qlutattn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_qlutattn_buffer_interface, data, size);
}

static size_t ggml_backend_qlutattn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;
}

static size_t ggml_backend_qlutattn_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    // 使用默认大小计算
    return ggml_nbytes(tensor);
}

static bool ggml_backend_qlutattn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;
}

// 返回buffer类型的主函数
ggml_backend_buffer_type_t ggml_backend_qlutattn_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_qlutattn = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_qlutattn_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_qlutattn_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_qlutattn_buffer_type_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ ggml_backend_qlutattn_buffer_type_get_alloc_size,
            /* .is_host          = */ ggml_backend_qlutattn_buffer_type_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::qlutattn::extra_buffer_type(),
    };

    return &ggml_backend_buffer_type_qlutattn;
}

#endif // GGML_USE_QLUTATTN 