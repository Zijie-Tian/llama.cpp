# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

llama.cpp uses CMake (3.14+). The old Makefile is deprecated.

```bash
# CPU-only release build
cmake -B build
cmake --build build --config Release -j $(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# With CUDA (NVIDIA GPUs) - Recommended build with Ninja and clang
CFLAGS="-march=native" CXXFLAGS="-march=native" cmake \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -G Ninja \
  -DGGML_CUDA=ON \
  -B build-x86
cmake --build build-x86 --config Release -j $(nproc)

# With Metal (Apple Silicon)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release

# With HIP (AMD GPUs)
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# With Vulkan
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release

# Build with tests enabled
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release -j $(nproc)
```

Binaries output to `build/bin/`. Install **ccache** for faster rebuilds.

## Testing

```bash
# Run all tests
ctest --test-dir build --output-on-failure -j $(nproc)

# Run single test by name
ctest --test-dir build -R test_name --output-on-failure

# Test basic inference
./build/bin/llama-cli -m model.gguf -p "Hello" -n 10

# Test backend operations
./build/bin/test-backend-ops

# Benchmark performance
./build/bin/llama-bench -m model.gguf

# Evaluate perplexity
./build/bin/llama-perplexity -m model.gguf -f dataset.txt
```

Server tests are in `tools/server/tests/` and require Python venv activation.

## Code Architecture

**Core Libraries:**
- `src/` - Core llama library implementation
  - `llama.cpp` - Main API entry point
  - `llama-arch.cpp/h` - Model architecture implementations (100+ architectures)
  - `llama-context.cpp/h` - Context and computation management
  - `llama-graph.cpp` - Computation graph building
  - `llama-vocab.cpp/h` - Tokenization (SPM, BPE, WPM, UGM, RWKV)
  - `llama-sampling.cpp/h` - Sampling/decoding strategies
  - `llama-model-loader.cpp/h` - GGUF model loading
- `include/llama.h` - Public C API (~1400 lines) - **read this first**
- `ggml/` - Tensor library with backend abstraction (cuda, metal, cpu, hip, vulkan, sycl, etc.)
- `common/` - Shared utilities for examples/tools

**Tools & Examples:**
- `tools/main/` - Main CLI (`llama-cli`)
- `tools/server/` - HTTP server with OpenAI-compatible API (`llama-server`)
- `tools/quantize/` - Model quantization (`llama-quantize`)
- `tools/llama-bench/` - Performance benchmarking
- `examples/` - 30+ example programs (simple, batched, speculative, embedding, etc.)

**Key Patterns:**
- C-style object system with opaque pointers (`llama_model`, `llama_context`, `llama_sampler`)
- Function naming: `llama_<class>_<action>` (e.g., `llama_model_load_from_file`, `llama_context_free`)
- Tensors stored in row-major order; dimension 0 = columns, 1 = rows
- Matrix multiplication: `C = ggml_mul_mat(A, B)` means `C^T = A B^T`

## Code Style

Format with `git clang-format` before committing. Key conventions:
- 4-space indentation, 120 char line limit
- snake_case for functions, variables, types
- UPPERCASE for enum values with prefix (e.g., `LLAMA_VOCAB_TYPE_SPM`)
- Pointer/reference alignment: `void * ptr`, `int & ref`
- Prefer longest common prefix for related names (`number_small`, `number_big` not `small_number`, `big_number`)
- Use basic for loops, avoid fancy STL
- Use sized integer types in public API (`int32_t`, etc.)
- Avoid third-party dependencies

Run `pre-commit run --all-files` before commits.

## CI

Add `ggml-ci` to commit message to trigger heavy CI workloads. Run full CI locally:
```bash
mkdir tmp && bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

For PRs: verify perplexity unchanged (`llama-perplexity`), verify performance (`llama-bench`), run `test-backend-ops` for GGML changes.
