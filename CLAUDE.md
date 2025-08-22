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

## LUT-based Quantization Knowledge

### Current Branch: tzj/opt_tmac
Working on TMAC (Table-based Matrix Acceleration) - LUT-based quantization for efficient matrix multiplication.

### TMAC Architecture Overview

TMAC is a sophisticated LUT-based quantization system integrated into llama.cpp through the extra_buffer mechanism. It transforms traditional matrix multiplication into efficient table lookup operations.

### Core Design Components

1. **Extra Buffer Type Mechanism**:
   - Extends `ggml::cpu::extra_buffer_type` to create specialized TMAC buffer type
   - Implements `supports_op()` to identify TMAC-compatible operations
   - Stores TMAC-specific metadata (scales, quantized weights) via `tensor->extra` field
   - Minimal intrusion into existing ggml architecture

2. **LUT (Look-Up Table) Core Concepts**:
   - **g=4**: Each LUT group processes 4 elements atomically
   - **ngroups_per_elem**: Typically 2 (8 bits / 4 bits per group)
   - Pre-computes dot products for all 2^g bit patterns
   - Transforms matrix multiplication into table lookups

3. **Weight Transformation Pipeline**:
   - `ggml_backend_tmac_buffer_set_tensor()`: Intercepts weight setting operations
   - `ggml_tmac_transform_tensor()`: Performs complex weight permutation
   - Optimizes layout for SIMD instructions and memory access patterns
   - Supports both in-place and out-of-place transformations

4. **Kernel Configuration System**:
   ```cpp
   struct tmac_kernel_config {
       int32_t g;                 // Group size for LUT (typically 4)
       int32_t ngroups_per_elem;  // Groups per element (8/g)
       int32_t q_group_size;      // Quantization group size (64/128)
       int32_t act_group_size;    // Activation group size
       int32_t bm;                // Block size (256/512/1024/2048)
       int32_t kfactor;           // K-dimension unrolling factor
       int32_t simd_n_in;         // SIMD input width
       int32_t simd_n_out;        // SIMD output width
       bool has_scale;            // Whether scales are used
       bool has_zero_point;       // Whether zero points are used
       bool one_scale;            // Single scale for all weights
   }
   ```

5. **Quantization Type Support**:
   - **2-bit types**: 
     - TMAC_W2G64_0/1 (64-element groups, with/without zero point)
     - TMAC_W2G128_0/1 (128-element groups, with/without zero point)
     - TMAC_BN_0 (BitNet-style, single scale)
   - **4-bit types**:
     - TMAC_W4G64_0/1 (64-element groups)
     - TMAC_W4G128_0/1 (128-element groups)
   - **Compatible types**: Q4_0, TQ1_0, TQ2_0

6. **Auto-tuning System**:
   - Dynamically searches optimal kernel configurations
   - Tests different bm values: {256, 512, 1024, 2048, 320, 640, 1280}
   - Evaluates kfactor options: {8, 16}
   - Caches configurations per (M, K, bits) combination
   - Uses performance microbenchmarks for selection

7. **Parallel Execution Strategy**:
   - Chunk-based work distribution
   - Supports multi-threading with atomic operations
   - Separates LUT construction (init phase) from computation
   - Efficient barrier synchronization

### Implementation Details

1. **Memory Layout Optimization**:
   - Weight permutation follows pattern: (M, K) → (M/bm, bm/bits, K/g)
   - SIMD-friendly layout with interleaved bit planes
   - Aligned memory allocation (64-byte alignment)

2. **Scale Storage Strategy**:
   - **Multi-channel**: Scales shape (M, K/group_size)
   - **Per-channel**: Scales shape (M, K)
   - **BitNet mode**: Single scale for entire tensor
   - Zero points stored separately when enabled

3. **Critical Constraints**:
   - Weight quantization group_size must be ≥ g (typically 4)
   - LUT processes g elements as atomic unit
   - Activation group size must be multiple of g
   - M dimension must be divisible by bm/bits

4. **Performance Optimizations**:
   - Pre-computed LUT tables reduce runtime computation
   - SIMD instructions for parallel processing
   - Cache-friendly memory access patterns
   - Minimal data movement through in-place operations

### File Structure

- `tmac.h/cpp`: Main TMAC buffer type implementation and integration
- `lut_mul_mat.h/cpp`: Core matrix multiplication logic and weight transformation
- `lut_ctor.h/cpp`: LUT construction utilities
- `tbl.h/cpp`: Table-based computation kernels

### Key Functions

- `ggml_backend_tmac_buffer_type()`: Returns TMAC buffer type singleton
- `ggml_tmac_transform_tensor()`: Transforms weights to TMAC format
- `ggml_tmac_tune_kernel_config()`: Auto-tunes kernel parameters
- `ggml_backend_tmac_mul_mat()`: Main matrix multiplication entry point
- `qgemm_lut_int8_g4()`: Core LUT-based GEMM implementation

### Current Understanding
- TMAC achieves efficiency by converting matrix multiplication to table lookups
- The extra_buffer mechanism allows seamless integration without modifying core ggml
- Auto-tuning ensures optimal performance across different tensor shapes
- Trade-off exists between quantization granularity (accuracy) and LUT compatibility (efficiency)
- The system is designed for minimal code intrusion while maximizing performance gains

## Comment Format Standards

Use the following comment format standards throughout the codebase:

```cpp
// TODO: Description of task to be done
// NOTE: Important note or observation
// NOTICE: Warning or important notice
// WHY: Explanation of why something is done this way
// EXPLAIN: Detailed explanation of complex logic
// HACK: Temporary workaround that needs improvement
//> ===================================================================================================
//> Section or module separator description
//> ===================================================================================================
```

Pragma directives:
```cpp
#pragma unroll count
for (int i = 0; i < N; i++) {}
```

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
