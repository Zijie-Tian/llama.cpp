# 调试测试

> 本文档说明如何调试 llama.cpp 的测试。
>
> 引用: `@file:docs/basic/debugging.md`

---

## 快速调试脚本

使用 `scripts/debug-test.sh` 进行交互式调试。

### 基本用法

```bash
# 运行测试并获取 PASS/FAIL
./scripts/debug-test.sh test-tokenizer

# 使用 GDB 调试
./scripts/debug-test.sh -g test-tokenizer
```

### 指定测试编号

```bash
# 运行第 23 个测试
./scripts/debug-test.sh test 23
```

---

## 手动调试流程

### 1. 创建构建目录

```bash
rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug
```

### 2. 编译调试版本

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CUDA=1 -DLLAMA_FATAL_WARNINGS=ON ..
make -j
```

### 3. 查找测试命令

```bash
ctest -R "test-tokenizer" -V -N
```

### 4. 运行 GDB

```bash
gdb --args ./build-ci-debug/bin/test-tokenizer-0 "./models/ggml-vocab-llama-spm.gguf"
```

---

## 内存检查

### Address Sanitizer

```bash
cmake -B build -DLLAMA_SANITIZE_ADDRESS=ON
cmake --build build
```

### Thread Sanitizer

```bash
cmake -B build -DLLAMA_SANITIZE_THREAD=ON
cmake --build build
```

### macOS Leaks

```bash
leaks -atExit -- ./build/bin/test-thread-safety ...
```
