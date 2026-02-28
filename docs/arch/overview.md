# llama.cpp 架构总览

> 本文档描述 llama.cpp 的高层次架构和代码组织。
>
> 引用: `@file:docs/arch/overview.md`

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                        Applications                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │llama-cli │ │llama-srv │ │examples/ │ │  Python tools   │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────────────────┘ │
├───────┼────────────┼────────────┼───────────────────────────┤
│       └────────────┴────────────┘                           │
│                     common/                                  │
│         (参数解析、采样、对话模板、Jinja 引擎)                 │
├─────────────────────────────────────────────────────────────┤
│                       libllama                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │  Model   │ │ Context  │ │ KV Cache │ │    Sampler      │ │
│  │ Loading  │ │  Mgmt    │ │  Mgmt    │ │   Strategies    │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │  Graph   │ │  Vocab   │ │ Grammar  │ │   Quantize      │ │
│  │   Ops    │ │  Tokenize│ │  Constrain│ │                │ │
│  └────┬─────┘ └──────────┘ └──────────┘ └─────────────────┘ │
├───────┼─────────────────────────────────────────────────────┤
│       │                     ggml                             │
│       │              (Tensor Computation Library)            │
│       └──────────────┬──────────────────────────────────────┘
│                      │
│  ┌───────────────────┼───────────────────────────────────────┐
│  │                   ▼                                       │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│  │  │ggml-cpu │ │ggml-cuda│ │ggml-mltl│ │ggml-vulk│ ...     │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘         │
│  └───────────────────────────────────────────────────────────┘
```

---

## 核心组件

### 1. ggml - 张量计算库

**位置**: `ggml/`

| 文件/目录 | 用途 |
|-----------|------|
| `ggml.c` | 核心张量操作、计算图 |
| `ggml-backend.cpp` | 后端抽象层，自动调度 |
| `ggml-alloc.c` | 内存分配管理 |
| `ggml-quants.c` | 量化格式实现 |
| `ggml-cpu/` | CPU 后端 (ARM NEON, x86 AVX/AVX2/AVX512) |
| `ggml-cuda/` | NVIDIA CUDA 后端 |
| `ggml-metal/` | Apple Metal 后端 |
| `ggml-vulkan/` | Vulkan 后端 |
| `ggml-sycl/` | Intel SYCL 后端 |
| `ggml-hip/` | AMD HIP 后端 |

**关键概念**:
- **Backend**: 硬件后端抽象 (CPU, CUDA, Metal, 等)
- **Buffer Type**: 内存类型 (主机内存、设备内存)
- **Tensor**: 张量，数据的基本单位
- **Graph**: 计算图，描述运算流程

### 2. llama - 推理引擎

**位置**: `src/`

| 文件 | 用途 |
|------|------|
| `llama-model.cpp` | 模型加载、架构识别、权重加载 |
| `llama-context.cpp` | 推理上下文、解码循环 |
| `llama-graph.cpp` | 构建计算图、算子调度 |
| `llama-kv-cache.cpp` | KV 缓存管理、注意力优化 |
| `llama-sampler.cpp` | 采样策略 (temperature, top-k, top-p, 等) |
| `llama-vocab.cpp` | 分词器 (SentencePiece, BPE, WPM) |
| `llama-grammar.cpp` | GBNF 语法约束生成 |
| `llama-chat.cpp` | 对话模板处理 |
| `llama-quant.cpp` | 模型量化实现 |
| `llama-arch.cpp` | 支持的模型架构定义 |

**位置**: `src/models/`

每个模型架构一个文件，约 100+ 种模型：
- `llama.cpp`, `qwen.cpp`, `gemma.cpp`, `phi.cpp`, ...
- 每个文件实现该架构的图层构建逻辑

### 3. common - 共享工具

**位置**: `common/`

| 文件 | 用途 |
|------|------|
| `common.cpp` | 通用工具函数 |
| `arg.cpp` | 命令行参数解析 |
| `sampling.cpp` | 高级采样逻辑 |
| `chat.cpp` | 对话格式化 |
| `jinja/` | Jinja2 模板引擎 (用于 chat templates) |
| `peg-parser.cpp` | PEG 解析器 |

### 4. Tools - 可执行工具

**位置**: `tools/`

| 工具 | 用途 |
|------|------|
| `cli/` | llama-cli: 交互式命令行 |
| `server/` | llama-server: OpenAI 兼容 HTTP API |
| `quantize/` | llama-quantize: 模型量化 |
| `perplexity/` | llama-perplexity: 困惑度测量 |
| `llama-bench/` | 性能基准测试 |
| `imatrix/` | 重要性矩阵计算 (用于量化) |

---

## 数据流

### 推理流程

```
1. 加载模型 (llama_load_model)
   ├── 解析 GGUF 文件
   ├── 识别架构
   ├── 加载权重到内存
   └── 初始化后端

2. 创建上下文 (llama_new_context)
   ├── 分配 KV 缓存
   ├── 设置参数 (ctx_size, batch_size, 等)
   └── 初始化采样器

3. Tokenize (llama_tokenize)
   └── 使用模型特定的分词器

4. 推理循环 (llama_decode)
   ├── 构建计算图 (build_graph)
   ├── 后端调度 (compute)
   │   └── 自动选择最优后端
   ├── KV 缓存更新
   └── 采样 (sample)

5. 输出 (llama_token_to_piece)
   └── 将 token 转换为文本
```

### 矩阵乘法约定

**重要**: ggml 的矩阵乘法是反常规的：

```cpp
C = ggml_mul_mat(ctx, A, B)  // 表示 C^T = A * B^T 或 C = B * A^T
```

张量按行主序存储，维度 0 = 列，维度 1 = 行。

---

## 后端系统

### 自动调度

```cpp
// 代码示例：后端选择优先级
ggml_backend_t backend = ggml_backend_cpu_init();  // 默认 CPU

// 如果可用，使用 GPU 后端
if (ggml_backend_cuda_available()) {
    backend = ggml_backend_cuda_init(device);
}
```

**运行时选择**:
```bash
# 列出可用设备
./llama-cli --list-devices

# 指定设备
./llama-cli -m model.gguf --device CUDA0
```

### 多后端构建

可以同时构建多个后端：

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_METAL=ON
```

---

## 模型支持

### 架构定义

在 `src/llama-arch.h/cpp` 中定义：

```cpp
enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_QWEN2,
    LLM_ARCH_GEMMA,
    // ... 100+ 架构
};
```

### 添加新模型

参见 [docs/development/HOWTO-add-model.md](../../development/HOWTO-add-model.md)

关键文件：
- `src/llama-arch.cpp`: 添加架构枚举和元数据
- `src/models/xxx.cpp`: 实现图层构建
- `convert_hf_to_gguf.py`: 添加转换支持

---

## 关键接口

### C API

**头文件**: `include/llama.h` (80K+ 行，非常详细)

主要类型：
- `llama_model*`: 模型句柄
- `llama_context*`: 上下文句柄
- `llama_token`: Token ID (int32)

主要函数：
- `llama_load_model_from_file()`: 加载模型
- `llama_tokenize()`: 分词
- `llama_decode()`: 解码
- `llama_sampler_sample()`: 采样

### C++ API

**头文件**: `include/llama-cpp.h` (简洁包装)
