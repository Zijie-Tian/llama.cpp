# llama.cpp 构建系统

> 本文档详细说明 llama.cpp 的构建方式和配置选项。
>
> 引用: `@file:docs/basic/build-system.md`

---

## 快速构建

### CPU 构建

```bash
cmake -B build
cmake --build build --config Release -j$(nproc)
```

### GPU 后端构建

```bash
# CUDA (NVIDIA)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Metal (Apple Silicon) - macOS 默认启用
cmake -B build
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)

# HIP (AMD GPU)
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030
cmake --build build --config Release -j$(nproc)

# Vulkan
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)

# SYCL (Intel GPU)
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON
cmake --build build --config Release -j$(nproc)
```

---

## 构建选项

### 核心选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `-DBUILD_SHARED_LIBS` | ON | 构建共享库 |
| `-DLLAMA_BUILD_TESTS` | ON (standalone) | 构建测试 |
| `-DLLAMA_BUILD_TOOLS` | ON (standalone) | 构建工具 |
| `-DLLAMA_BUILD_SERVER` | ON (standalone) | 构建服务器 |
| `-DLLAMA_BUILD_EXAMPLES` | ON (standalone) | 构建示例 |
| `-DLLAMA_FATAL_WARNINGS` | OFF | 警告视为错误 |
| `-DCMAKE_BUILD_TYPE` | Release | 构建类型 |

### 调试构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

### 静态构建

```bash
cmake -B build -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release
```

---

## 后端特定选项

### CUDA 选项

| 选项 | 说明 |
|------|------|
| `-DGGML_CUDA_FORCE_MMQ` | 强制使用量化矩阵乘法核 |
| `-DGGML_CUDA_FORCE_CUBLAS` | 强制使用 cuBLAS FP16 |
| `-DCMAKE_CUDA_ARCHITECTURES` | 指定 GPU 架构 (如 "86;89") |
| `-DGGML_CUDA_GRAPHS` | 启用 CUDA Graphs (默认 ON) |
| `-DGGML_CUDA_FA_ALL_QUANTS` | 所有 KV 缓存量化类型的 FlashAttention |

### Metal 选项

| 选项 | 说明 |
|------|------|
| `-DGGML_METAL=OFF` | 禁用 Metal |
| `-DGGML_METAL_EMBED_LIBRARY` | 嵌入 Metal 库 |
| `-DGGML_METAL_USE_BF16` | 使用 BFloat16 |
| `-DGGML_METAL_SHADER_DEBUG` | 启用着色器调试 |

### 其他后端

- `-DGGML_BLAS=ON` - BLAS 支持
- `-DGGML_BLAS_VENDOR=OpenBLAS` - OpenBLAS
- `-DGGML_BLAS_VENDOR=Intel10_64lp` - Intel MKL
- `-DGGML_VULKAN=ON` - Vulkan
- `-DGGML_SYCL=ON` - SYCL/Intel GPU
- `-DGGML_CANN=ON` - Ascend CANN
- `-DGGML_MUSA=ON` - Moore Threads MUSA
- `-DGGML_HIP=ON` - AMD HIP/ROCm
- `-DGGML_ZENDNN=ON` - AMD ZenDNN
- `-DGGML_RPC=ON` - RPC 后端

---

## 构建优化

### 使用 ccache

```bash
# 安装 ccache
# Ubuntu: sudo apt install ccache
# macOS: brew install ccache

cmake -B build -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

### 使用 Ninja

```bash
cmake -B build -G Ninja
cmake --build build
```

### Windows 构建

```bash
# Visual Studio 2022
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release

# Windows on ARM
cmake --preset arm64-windows-llvm-release -DGGML_OPENMP=OFF
cmake --build build-arm64-windows-llvm-release
```

---

## 构建产物

构建完成后，二进制文件位于：

```
build/bin/
├── llama-cli              # 交互式 CLI
├── llama-server           # HTTP 服务器
├── llama-quantize         # 量化工具
├── llama-perplexity       # 困惑度测试
├── llama-bench            # 性能基准
├── test-*                 # 测试可执行文件
└── example-*              # 示例程序
```

库文件：
- `build/libllama.so` (Linux) / `build/libllama.dylib` (macOS) / `build/llama.dll` (Windows)
- `build/libggml.so` 等
