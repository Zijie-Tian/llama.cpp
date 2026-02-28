# llama.cpp 测试指南

> 本文档详细说明如何运行和编写 llama.cpp 的测试。
>
> 引用: `@file:docs/basic/testing.md`

---

## 快速测试

### 运行所有测试

```bash
cd build
ctest -L main --verbose --timeout 900
```

### 运行特定标签测试

```bash
ctest -L model --verbose    # 需要模型的测试
ctest -L python --verbose   # Python 相关测试
```

---

## 测试分类

### 1. 单元测试 (tests/)

构建后位于 `build/bin/test-*`：

| 测试 | 用途 | 何时运行 |
|------|------|----------|
| `test-sampling` | 采样策略测试 | 修改 sampler |
| `test-grammar-parser` | GBNF 语法解析测试 | 修改 grammar |
| `test-tokenizer-0` | 分词器测试 | 修改 vocab/tokenizer |
| `test-chat-template` | 对话模板测试 | 修改 chat template |
| `test-quantize-fns` | 量化函数测试 | 修改 quantization |
| `test-rope` | RoPE 测试 | 修改 RoPE 实现 |
| `test-gguf` | GGUF 格式测试 | 修改 GGUF |
| `test-backend-ops` | **后端操作一致性测试** | **修改 ggml 核心** |
| `test-thread-safety` | 线程安全测试 | 修改并行解码 |

### 2. 关键测试详解

#### test-backend-ops (最重要)

测试不同后端实现的一致性。修改 ggml 后**必须**运行：

```bash
./build/bin/test-backend-ops
```

要求：需要至少两个不同的 ggml 后端（如 CPU + CUDA）

#### test-tokenizer-0

需要词汇表文件，位于 `models/ggml-vocab-*.gguf`：

```bash
./build/bin/test-tokenizer-0 models/ggml-vocab-llama-bpe.gguf
```

#### test-quantize-fns

测试量化/反量化函数：

```bash
./build/bin/test-quantize-fns
```

#### test-thread-safety

需要下载模型：

```bash
./build/bin/test-thread-safety \
    -m models/tinyllamas/stories15M-q4_0.gguf \
    -ngl 99 -p "test" -n 128 -c 256 -ub 32 -np 4 -t 2
```

---

## 服务器测试

位于 `tools/server/tests/`，使用 pytest：

```bash
cd tools/server/tests

# 安装依赖
pip install pytest requests

# 运行所有测试
pytest -v -x

# 运行特定测试
pytest -v -x test_chat_completion.py
```

---

## 完整 CI 测试

### 本地运行完整 CI

```bash
mkdir -p tmp/results tmp/mnt

# CPU-only 构建测试
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# CUDA 构建测试
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# SYCL 构建测试
source /opt/intel/oneapi/setvars.sh
GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# MUSA 构建测试
GG_BUILD_MUSA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

---

## 性能测试

### llama-bench

```bash
# 基本基准测试
./build/bin/llama-bench -m model.gguf

# 特定后端
./build/bin/llama-bench -m model.gguf -ngl 99

# 批量大小对比
./build/bin/llama-bench -m model.gguf -p 512,1024,2048
```

### llama-perplexity

```bash
# 测量困惑度
./build/bin/llama-perplexity -m model.gguf -f text.txt

# 通常用于验证量化质量
```

---

## 调试测试

### Sanitizer 构建

```bash
# Address Sanitizer
cmake -B build -DLLAMA_SANITIZE_ADDRESS=ON

# Thread Sanitizer
cmake -B build -DLLAMA_SANITIZE_THREAD=ON

# Undefined Behavior Sanitizer
cmake -B build -DLLAMA_SANITIZE_UNDEFINED=ON
```

### 内存检查 (macOS)

```bash
leaks -atExit -- ./build/bin/test-thread-safety ...
```

---

## 添加新测试

### C++ 测试

在 `tests/` 目录添加：

```cpp
// tests/test-my-feature.cpp
#include "testing_common.h"

int main(int argc, char ** argv) {
    // 测试代码
    return 0;
}
```

在 `tests/CMakeLists.txt` 中添加：

```cmake
llama_build_and_test(test-my-feature.cpp)
```

### 服务器测试

在 `tools/server/tests/` 添加 pytest 测试文件。

---

## 测试模型

测试需要的小型模型会自动下载到 `build/models/`：

- `tinyllamas/stories15M-q4_0.gguf` - 15M 参数故事模型
- 词汇表文件 `models/ggml-vocab-*.gguf`

如需手动下载：

```bash
# 使用测试工具自动下载
./build/bin/test-thread-safety -m models/tinyllamas/stories15M-q4_0.gguf ...
```
