# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is llama.cpp - a C/C++ implementation of Meta's LLaMA model (and others) with minimal dependencies. The project uses the ggml tensor library for model evaluation and focuses on performance, simplicity, and cross-platform compatibility.

## Build Commands

The project uses CMake (Makefile is deprecated):

```bash
# Basic CPU build
cmake -B build
cmake --build build --config Release -j 8

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# GPU-accelerated builds
cmake -B build -DGGML_CUDA=ON      # CUDA
cmake -B build -DGGML_VULKAN=ON    # Vulkan
cmake -B build -DGGML_HIP=ON       # AMD HIP
cmake -B build -DGGML_SYCL=ON      # Intel SYCL
```

Binaries are generated in `build/bin/`

## Testing

```bash
# Build tests (included by default)
cmake -B build -DLLAMA_BUILD_TESTS=ON

# Run specific tests
./build/bin/test-backend-ops
./build/bin/test-tokenizer-0
./build/bin/test-sampling

# Run full CI locally
mkdir tmp
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# With CUDA
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

## Code Architecture

### Core Components
- **ggml**: Low-level tensor library (in `ggml/`)
- **llama**: Main library implementing model loading, inference, and cache management
- **common**: Shared utilities and helpers
- **examples**: Various tools and applications

### Key Executables
- `llama-cli`: Main CLI tool for model inference
- `llama-server`: HTTP server with OpenAI-compatible API
- `llama-bench`: Performance benchmarking tool
- `llama-perplexity`: Model quality evaluation
- `llama-quantize`: Model quantization tool

### Current Development Focus

The current branch (`tzj/qlutattn`) is implementing a **Mixed Precision KV Cache** feature:
- Dual unified cache architecture: hot cache (FP16) + cold cache (Q4_0 quantized)
- Automatic quantization triggers when hot cache usage exceeds 80%
- Based on SWA (Sliding Window Attention) design patterns
- Implementation in `ggml/src/ggml-cpu/ops.cpp`

## Code Style and Conventions

1. **Formatting**:
   - 4 spaces for indentation (no tabs except in Makefiles)
   - Brackets on the same line
   - `void * ptr`, `int & a` spacing
   - Use `snake_case` for functions, variables, and types
   - Clean up trailing whitespaces

2. **Naming**:
   - Pattern: `<class>_<method>` where `<method>` is `<action>_<noun>`
   - Enum values in UPPER_CASE prefixed with enum name
   - Use `_t` suffix for opaque types
   - C/C++ filenames: lowercase with dashes (`.h`, `.c`, `.cpp`)

3. **Development Guidelines**:
   - Avoid third-party dependencies
   - Keep code simple - avoid fancy STL constructs and templates
   - Use sized integer types (`int32_t`) in public APIs
   - Test changes with full CI locally before publishing
   - Create separate PRs for each feature/fix

4. **Before Committing**:
   - Run linting if available
   - Test with `test-backend-ops` if modifying ggml operators
   - Verify perplexity and performance are not negatively affected
   - Ensure cross-platform compatibility

## Important Notes

- Main branch is `master` (not `main`)
- Always test builds with different backends before submitting PRs
- The project prioritizes minimal dependencies and cross-platform support
- When working on the mixed KV cache feature, refer to `MIXED_KV_CACHE_STATUS.md` for implementation details

## Claude Memory

- 请你将当前的知识全部存储下来: A request to store all current knowledge comprehensively