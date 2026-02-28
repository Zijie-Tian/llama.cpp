# llama.cpp 算子 (Operators)

> 本文档详细说明 ggml/llama.cpp 中的核心算子实现。
>
> 引用: `@file:docs/ops/overview.md`

---

## 算子分类

### 1. 基础张量操作

| 算子 | 函数 | 描述 |
|------|------|------|
| **Add** | `ggml_add` | 逐元素加法 |
| **Mul** | `ggml_mul` | 逐元素乘法 |
| **Sub** | `ggml_sub` | 逐元素减法 |
| **Div** | `ggml_div` | 逐元素除法 |
| **Scale** | `ggml_scale` | 标量乘法 |
| **Clamp** | `ggml_clamp` | 值裁剪 |

### 2. 矩阵运算

| 算子 | 函数 | 描述 | 注意 |
|------|------|------|------|
| **MatMul** | `ggml_mul_mat` | 矩阵乘法 | C = B * A^T |
| **MatMulId** | `ggml_mul_mat_id` | 条件矩阵乘 | MoE 路由用 |
| **OutProd** | `ggml_out_prod` | 外积 | |

**矩阵乘法约定**:
```cpp
// 注意：ggml 的矩阵乘法是反常规的
// C = ggml_mul_mat(ctx, A, B) 表示 C^T = A * B^T
// 或等价地 C = B * A^T

// 标准数学: Y = X @ W^T + b
// ggml 实现: Y = ggml_mul_mat(ctx, W, X) + b
```

### 3. 激活函数

| 算子 | 函数 | 描述 |
|------|------|------|
| **ReLU** | `ggml_relu` | ReLU 激活 |
| **GELU** | `ggml_gelu` | GELU 激活 |
| **SiLU** | `ggml_silu` | SiLU/Swish 激活 |
| **Tanh** | `ggml_tanh` | 双曲正切 |
| **Sigmoid** | `ggml_sigmoid` | Sigmoid 激活 |
| **GELU Quick** | `ggml_gelu_quick` | 快速 GELU 近似 |

### 4. 归一化

| 算子 | 函数 | 描述 | 用途 |
|------|------|------|------|
| **Norm** | `ggml_norm` | L2 归一化 | RMSNorm |
| **RMS Norm** | `ggml_rms_norm` | RMS 归一化 | LLaMA 使用 |
| **LayerNorm** | `ggml_layer_norm` | 层归一化 | BERT 使用 |
| **GroupNorm** | `ggml_group_norm` | 组归一化 | Conv 模型 |

### 5. 注意力机制

| 算子 | 函数 | 描述 |
|------|------|------|
| **FlashAttention** | `ggml_flash_attn` | 高效注意力 (fused) |
| **Softmax** | `ggml_soft_max` | Softmax |
| **SoftmaxExt** | `ggml_soft_max_ext` | 扩展 Softmax (alibi, mask) |
| **Rope** | `ggml_rope` | RoPE 位置编码 |
| **RopeExt** | `ggml_rope_ext` | 扩展 RoPE |

### 6. 量化解压

| 算子 | 函数 | 描述 |
|------|------|------|
| **Dequantize** | `ggml_dequantize` | 反量化到 FP32/FP16 |
| **Quantize** | `ggml_quantize` | 量化到目标格式 |
| **QuantizeFree** | `ggml_quantize_free` | 释放量化资源 |

支持格式: Q4_0, Q4_1, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, Q8_K, IQ4_XS, ...

### 7. 形状操作

| 算子 | 函数 | 描述 |
|------|------|------|
| **Reshape** | `ggml_reshape` | 改变张量形状 |
| **View** | `ggml_view` | 创建视图 (无拷贝) |
| **Permute** | `ggml_permute` | 维度置换 |
| **Transpose** | `ggml_transpose` | 转置 |
| **Cont** | `ggml_cont` | 转为连续内存 |
| **Copy** | `ggml_cpy` | 拷贝张量 |

### 8. 归约操作

| 算子 | 函数 | 描述 |
|------|------|------|
| **Sum** | `ggml_sum` | 求和 |
| **Mean** | `ggml_mean` | 平均值 |
| **ArgMax** | `ggml_argmax` | 最大值的索引 |
| **CountEqual** | `ggml_count_equal` | 计数相等元素 |

### 9. 条件/控制

| 算子 | 函数 | 描述 |
|------|------|------|
| **GetRows** | `ggml_get_rows` | 按索引取行 |
| **GetRowsBack** | `ggml_get_rows_back` | 反向传播 |
| **DiagMaskInf** | `ggml_diag_mask_inf` | 对角掩码 (因果注意力) |

---

## 算子实现位置

### ggml.c 中的通用实现

```c
// ggml/src/ggml.c

// 基础算子
static void ggml_compute_forward_add(...);
static void ggml_compute_forward_mul(...);
static void ggml_compute_forward_silu(...);
static void ggml_compute_forward_norm(...);
static void ggml_compute_forward_rms_norm(...);

// 矩阵乘法
static void ggml_compute_forward_mul_mat(...);
static void ggml_compute_forward_mul_mat_q(...);  // 量化版本

// 注意力
static void ggml_compute_forward_flash_attn(...);
static void ggml_compute_forward_soft_max(...);
static void ggml_compute_forward_rope(...);
```

### 后端特定优化

#### CUDA (ggml-cuda/)

```cuda
// ggml/src/ggml-cuda/ggml-cuda.cu

// 模板化核函数
template<typename T>
__global__ void add_kernel(const T* x, const T* y, T* dst, int n);

// 量化矩阵乘法
static void ggml_cuda_mul_mat(...);
static void ggml_cuda_mul_mat_q(...);

// FlashAttention
static void ggml_cuda_flash_attn(...);
```

#### CPU 优化 (ggml-cpu/)

```cpp
// ggml/src/ggml-cpu/ggml-cpu-x86.cpp

// AVX2/AVX512 优化
static void ggml_compute_forward_add_q4_0_avx2(...);
static void ggml_compute_forward_mul_mat_q4_0_avx512(...);

// ARM NEON 优化
// ggml/src/ggml-cpu/ggml-cpu-aarch64.c
static void ggml_compute_forward_add_q4_0_neon(...);
```

---

## 算子融合 (Operator Fusion)

### 自动融合

```cpp
// llama-graph.cpp 中的融合优化

// 融合: silu(x) * y
// 原为: mul(silu(x), y)
// 融合为: ggml_mul_silu(ctx, x, y)

// 融合: norm + mul + add
// RMSNorm 的缩放和偏移
```

### 手动融合算子

| 融合算子 | 等效操作 | 用途 |
|----------|----------|------|
| `ggml_mul_mat_id` | mul_mat + get_rows | MoE |
| `ggml_flash_attn` | q @ k^T / sqrt(d) + softmax + @ v | 注意力 |
| `ggml_rms_norm` | sqrt(mean(x^2)) + scale | LLaMA Norm |

---

## 添加新算子

### 步骤

1. **ggml.h** - 添加函数声明:

```c
GGML_API struct ggml_tensor * ggml_my_op(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);
```

2. **ggml.c** - 实现通用版本:

```c
static void ggml_compute_forward_my_op(...) {
    // 通用 CPU 实现
}
```

3. **ggml-backend.cpp** - 添加调度:

```cpp
// 检查后端是否支持
bool ggml_backend_supports_op(...);
```

4. **后端特定优化** - 在 CUDA/Metal 等中添加:

```cuda
// ggml-cuda.cu
static void ggml_cuda_my_op(...);
```

5. **测试** - 添加到 test-backend-ops:

```cpp
// tests/test-backend-ops.cpp
TEST_CASE("my_op") {
    // 测试所有后端的一致性
}
```

---

## 量化算子详情

### 量化类型

| 类型 | 位宽 | 特点 | 精度 |
|------|------|------|------|
| Q4_0 | 4.5bpw | 快速，简单 | 较低 |
| Q4_K | 4.5bpw | K-quant，平衡 | 中等 |
| Q5_K | 5.5bpw | K-quant，较高 | 中高 |
| Q6_K | 6.6bpw | K-quant，高 | 高 |
| Q8_0 | 8.5bpw | 接近无损 | 很高 |
| IQ4_XS | 4.25bpw | 重要性加权 | 中等+ |

### 量化矩阵乘法

```cpp
// 反量化后乘法 (慢但通用)
FP16 = dequantize(Q4_0)
FP16 @ FP16

// 直接量化乘法 (快)
// CUDA: 使用 dp4a/dp2a 指令
// CPU: 使用 AVX2/NEON 向量化
```

---

## 性能优化指南

### 1. 内存布局

```cpp
// ✅ 连续内存访问更快
ggml_cont(ctx, tensor);  // 确保连续

// ✅ 批量操作优于循环
// 不要: for i in range(n): add(a[i], b[i])
// 要:   add(a, b)  // 批量
```

### 2. 后端选择

```cpp
// 让后端自动选择
// 或显式指定:
ggml_tensor_set_backend(tensor, backend);
```

### 3. 融合机会

```cpp
// ❌ 分离操作
x1 = ggml_norm(ctx, x);
x2 = ggml_scale(ctx, x1, scale);
y = ggml_add(ctx, x2, bias);

// ✅ 融合 (如果可用)
y = ggml_norm_inplace(ctx, x, scale, bias);
```
