# 添加新模型架构

> 本文档说明如何为 llama.cpp 添加新的模型架构支持。
>
> 引用: `@file:docs/arch/add-model.md`

---

## 概述

添加模型需要三个主要步骤：

1. 转换模型到 GGUF 格式
2. 在 llama.cpp 中定义模型架构
3. 构建 GGML 计算图实现

---

## 1. 转换模型到 GGUF

使用 Python 转换脚本和 [gguf](https://pypi.org/project/gguf/) 库。

### 1.1 注册模型

在 `convert_hf_to_gguf.py` 中定义模型：

```python
@ModelBase.register("MyModelForCausalLM")
class MyModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MYMODEL
```

### 1.2 定义张量布局

在 `gguf-py/gguf/constants.py` 中添加：

```python
MODEL_ARCH.MYMODEL: [
    MODEL_TENSOR.TOKEN_EMBD,
    MODEL_TENSOR.OUTPUT_NORM,
    MODEL_TENSOR.OUTPUT,
    MODEL_TENSOR.ATTN_NORM,
    MODEL_TENSOR.ATTN_QKV,
    MODEL_TENSOR.ATTN_OUT,
    MODEL_TENSOR.FFN_DOWN,
    MODEL_TENSOR.FFN_UP,
]
```

### 1.3 映射张量名称

在 `gguf-py/gguf/tensor_mapping.py` 中映射原始名称到 GGUF 标准名称：

```python
block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
    MODEL_TENSOR.ATTN_NORM: (
        "transformer.h.{bid}.ln_1",  # 原始名称
        # ...
    )
}
```

---

## 2. 在 llama.cpp 中定义架构

### 2.1 添加架构枚举

在 `src/llama-arch.h` 中添加：

```cpp
enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_MYMODEL,  // 新模型
    // ...
};
```

### 2.2 定义架构元数据

在 `src/llama-arch.cpp` 中：

```cpp
// 添加到 LLM_ARCH_NAMES
{ LLM_ARCH_MYMODEL, "mymodel" }

// 添加到 llm_get_tensor_names
LLM_ARCH_MYMODEL: {
    LLM_TENSOR_NAMES[LLM_TENSOR_TOKEN_EMBD],
    // ...
}
```

### 2.3 RoPE 类型（如果有）

在 `src/llama-model.cpp` 的 `llama_model_rope_type` 中添加 case。

---

## 3. 构建 GGML 计算图

### 3.1 实现图构建器

在 `src/llama-model.cpp` 中创建新的构建器：

```cpp
struct llm_build_mymodel : public llm_graph_context {
    llm_build_mymodel(const llama_model & model, const llm_build_params & params)
        : llm_graph_context(model, params) {}

    ggml_cgraph * build() {
        // 实现模型前向传播图
        // 参考 llm_build_llama, llm_build_dbrx 等
    }
};
```

### 3.2 注册构建器

在 `llama_model::build_graph` 方法中添加：

```cpp
case LLM_ARCH_MYMODEL:
    return llm_build_mymodel(*this, params).build();
```

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `convert_hf_to_gguf.py` | 模型转换脚本 |
| `gguf-py/gguf/constants.py` | GGUF 常量定义 |
| `gguf-py/gguf/tensor_mapping.py` | 张量名称映射 |
| `src/llama-arch.h/cpp` | 架构枚举和元数据 |
| `src/llama-model.cpp` | 计算图构建 |
| `src/llama-model-loader.cpp` | 模型加载 |

---

## GGUF 规范

https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

---

## 验证

添加模型后，验证以下工具正常工作：

- `llama-cli` - 基本推理
- `llama-server` - HTTP API
- `llama-quantize` - 量化
- `imatrix` - 重要性矩阵
