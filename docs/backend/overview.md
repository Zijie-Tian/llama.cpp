# llama.cpp Backend 后端系统

> 本文档详细说明 llama.cpp 的后端架构和各硬件后端实现。
>
> 引用: `@file:docs/backend/overview.md`

---

## 后端架构

### 后端抽象层

```
┌─────────────────────────────────────────────────────────────┐
│                    ggml-backend.h                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Backend   │  │ Buffer Type │  │      Tensor         │  │
│  │  Interface  │  │  Interface  │  │     Interface       │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          ▼                ▼                    ▼
  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐
  │  ggml-cuda   │  │ ggml-metal   │  │   ggml-cpu     │
  │  (NVIDIA)    │  │  (Apple)     │  │  (Generic)     │
  └──────────────┘  └──────────────┘  └────────────────┘
  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐
  │  ggml-vulkan │  │  ggml-hip    │  │  ggml-sycl     │
  │   (KHR)      │  │   (AMD)      │  │  (Intel)       │
  └──────────────┘  └──────────────┘  └────────────────┘
```

---

## 后端注册与调度

### 后端注册

```cpp
// ggml-backend.cpp
struct ggml_backend_registry {
    std::vector<ggml_backend_dev_t> backends;

    // 自动注册所有编译的后端
    void register_backends() {
        // CPU 后端始终注册
        register_backend(ggml_backend_cpu_reg());

        #ifdef GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
        #endif

        #ifdef GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
        #endif

        // ... 其他后端
    }
};
```

### 缓冲区类型

| Buffer Type | 用途 | 设备 |
|-------------|------|------|
| `GGML_BACKEND_CPU` | 主机内存 | CPU |
| `GGML_BACKEND_GPU` | 设备内存 | GPU |
| `GGML_BACKEND_GPU_SPLIT` | 分割模型 | 多 GPU |

### 自动调度策略

```cpp
// 后端优先级 (从高到低)
// CUDA > Metal > Vulkan > SYCL > HIP > CPU

ggml_backend_t select_backend(struct ggml_tensor * tensor) {
    // 1. 检查张量是否在特定后端缓冲区
    // 2. 检查后端是否支持该算子
    // 3. 选择优先级最高的可用后端
}
```

---

## 各后端详解

### 1. CPU 后端 (ggml-cpu)

**位置**: `ggml/src/ggml-cpu/`

**优化**:
- ARM NEON (Apple Silicon, ARMv8)
- x86 AVX/AVX2/AVX512
- AMX (Apple M2+/Intel Sapphire Rapids)
- RISC-V RVV

**关键文件**:
- `ggml-cpu.c` - 主实现
- `ggml-cpu-aarch64.c` - ARM64 优化
- `ggml-cpu-x86.cpp` - x86 优化

### 2. CUDA 后端 (ggml-cuda)

**位置**: `ggml/src/ggml-cuda/`

**特性**:
- cuBLAS 矩阵乘法
- 自定义量化核 (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0)
- FlashAttention
- CUDA Graphs
- 多 GPU 支持 (NCCL)

**关键文件**:
- `ggml-cuda.cu` - 主实现
- `fattn.cu` - FlashAttention
- `mmq.cu` - 矩阵量化乘法
- `mma.cu` - 矩阵乘法

### 3. Metal 后端 (ggml-metal)

**位置**: `ggml/src/ggml-metal/`

**特性**:
- Apple Silicon 优化
- 统一内存架构利用
- Metal Performance Shaders

**关键文件**:
- `ggml-metal.m` - Objective-C 实现
- `ggml-metal.metal` - Metal 着色器

### 4. Vulkan 后端 (ggml-vulkan)

**位置**: `ggml/src/ggml-vulkan/`

**特性**:
- 跨平台 GPU 支持
- SPIR-V 着色器
- 计算着色器实现

**关键文件**:
- `ggml-vulkan.cpp` - 主实现
- `shaders/` - GLSL 着色器

### 5. SYCL 后端 (ggml-sycl)

**位置**: `ggml/src/ggml-sycl/`

**特性**:
- Intel GPU 支持 (Arc, Data Center)
- oneAPI 基础

**关键文件**:
- `ggml-sycl.cpp` - 主实现

### 6. HIP 后端 (ggml-hip)

**位置**: `ggml/src/ggml-hip/`

**特性**:
- AMD GPU 支持
- ROCm 基础
- 与 CUDA 代码共享大部分逻辑

**关键文件**:
- `ggml-hip.cpp` - 主实现

---

## 后端开发指南

### 添加新后端

1. **创建目录**: `ggml/src/ggml-<name>/`

2. **实现必需接口**:

```cpp
// ggml-backend-impl.h 定义的结构
struct ggml_backend_interface {
    // 分配/释放缓冲区
    ggml_backend_buffer_t (*alloc_buffer)(...);
    void (*free_buffer)(...);

    // 数据传输
    void (*cpy_tensor_from)(...);
    void (*cpy_tensor_to)(...);

    // 执行计算图
    void (*graph_compute)(...);

    // 同步
    void (*synchronize)(...);
};
```

3. **注册后端**:

```cpp
GGML_API ggml_backend_reg_t ggml_backend_<name>_reg(void) {
    static ggml_backend_reg reg = {
        .api_version = GGML_BACKEND_API_VERSION,
        .interface = &ggml_backend_<name>_interface,
        .context = NULL,
    };
    return &reg;
}
```

4. **CMake 集成**:

```cmake
# ggml/CMakeLists.txt
option(GGML_<NAME> "ggml: enable <name> backend" OFF)

if (GGML_<NAME>)
    add_subdirectory(src/ggml-<name>)
    list(APPEND GGML_BACKENDS ggml-<name>)
endif()
```

---

## 后端测试

### test-backend-ops

```bash
# 测试所有后端的一致性
./build/bin/test-backend-ops

# 测试特定后端
GGML_BACKEND=cuda ./build/bin/test-backend-ops
```

### 后端性能对比

```bash
# 比较 CPU vs GPU
./build/bin/llama-bench -m model.gguf -ngl 0    # CPU
./build/bin/llama-bench -m model.gguf -ngl 99   # GPU
```

---

## 多后端混合推理

### CPU+GPU 混合

```bash
# 部分层在 GPU，部分在 CPU
./llama-cli -m model.gguf -ngl 35  # 前35层 GPU，其余 CPU
```

### 多 GPU

```bash
# 多 GPU 张量并行
./llama-server -m model.gguf -ngl 99 -sm row
```

### RPC 后端

分布式推理：
```bash
# 主机
./rpc-server -H 0.0.0.0 -p 50052

# 客户端
./llama-cli -m model.gguf --rpc 192.168.1.100:50052
```
