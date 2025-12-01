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

## KV Cache Architecture

**Key Files:**
- `src/llama-kv-cache.cpp/h` - Core KV cache implementation
- `src/llama-kv-cache-iswa.cpp/h` - Interleaved SWA cache (dual-cache architecture)
- `src/llama-kv-cells.h` - Cell metadata management
- `src/llama-hparams.cpp` - SWA mask logic (`is_masked_swa`)

**Core Data Structures:**
```
llama_kv_cache
├── layers[]              # Per-layer K/V tensors
│   ├── k: [n_embd_k_gqa, kv_size, n_stream]
│   └── v: [n_embd_v_gqa, kv_size, n_stream]
├── v_cells[]             # Cell metadata per stream
│   ├── pos[]             # Token position at each cell
│   ├── seq[]             # Which sequences occupy each cell (bitset)
│   └── shift[]           # Pending position shifts (for RoPE)
├── v_heads[]             # Next free cell index per stream
└── seq_to_stream[]       # Sequence ID -> Stream ID mapping
```

**Unified vs Non-Unified Mode:**
```
Non-Unified (n_stream = n_seq_max):        Unified (n_stream = 1):
┌─────────────────────────────────┐        ┌─────────────────────────────────┐
│  Stream 0 (seq_id=0)            │        │  Single Stream (all sequences)  │
│  ┌─────────────────────────┐    │        │  ┌─────────────────────────┐    │
│  │ cells for sequence 0    │    │        │  │ seq0 │ seq1 │ seq0 │seq2│    │
│  └─────────────────────────┘    │        │  │ p=0  │ p=0  │ p=1  │p=0 │    │
├─────────────────────────────────┤        │  └─────────────────────────┘    │
│  Stream 1 (seq_id=1)            │        │  Sequences share same buffer    │
│  ┌─────────────────────────┐    │        │  (cells can have multiple seqs) │
│  │ cells for sequence 1    │    │        └─────────────────────────────────┘
│  └─────────────────────────┘    │
└─────────────────────────────────┘        Use: -kvu or --kv-unified
Each sequence has isolated buffer
```

### SWA (Sliding Window Attention) Types

**1. LLAMA_SWA_TYPE_NONE - Full Causal Attention:**
```
          Key positions (j)
          0   1   2   3   4   5   6   7
        ┌───┬───┬───┬───┬───┬───┬───┬───┐
     0  │ ■ │   │   │   │   │   │   │   │
     1  │ ■ │ ■ │   │   │   │   │   │   │
Q    2  │ ■ │ ■ │ ■ │   │   │   │   │   │  Every token sees
(i)  3  │ ■ │ ■ │ ■ │ ■ │   │   │   │   │  all previous tokens
     4  │ ■ │ ■ │ ■ │ ■ │ ■ │   │   │   │
     5  │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │   │   │  ■ = attends
        └───┴───┴───┴───┴───┴───┴───┴───┘  (empty) = causal mask
```

**2. LLAMA_SWA_TYPE_STANDARD - Sliding Window (Mistral, Qwen2):**
```
n_swa = 3: mask when (i - j) >= n_swa

          Key positions (j)
          0   1   2   3   4   5   6   7
        ┌───┬───┬───┬───┬───┬───┬───┬───┐
     0  │ ■ │   │   │   │   │   │   │   │
     1  │ ■ │ ■ │   │   │   │   │   │   │
Q    2  │ ■ │ ■ │ ■ │   │   │   │   │   │
(i)  3  │ × │ ■ │ ■ │ ■ │   │   │   │   │  Only sees last n_swa
     4  │ × │ × │ ■ │ ■ │ ■ │   │   │   │  tokens (sliding band)
     5  │ × │ × │ × │ ■ │ ■ │ ■ │   │   │
        └───┴───┴───┴───┴───┴───┴───┴───┘  × = SWA masked
```

**3. LLAMA_SWA_TYPE_CHUNKED - Chunk-based (Gemma 2):**
```
n_swa = 4 (chunk size): mask when j < chunk_start(i)

          Key positions (j)
          0   1   2   3   4   5   6   7
        ┌───┬───┬───┬───┬───┬───┬───┬───┐
     0  │ ■ │   │   │   │   │   │   │   │
     1  │ ■ │ ■ │   │   │   │   │   │   │  Chunk 0
     2  │ ■ │ ■ │ ■ │   │   │   │   │   │  [0-3]
     3  │ ■ │ ■ │ ■ │ ■ │   │   │   │   │
        ╞═══╧═══╧═══╧═══╪═══╧═══╧═══╧═══╡  ← chunk boundary
     4  │ × │ × │ × │ × │ ■ │   │   │   │
     5  │ × │ × │ × │ × │ ■ │ ■ │   │   │  Chunk 1
     6  │ × │ × │ × │ × │ ■ │ ■ │ ■ │   │  [4-7]
     7  │ × │ × │ × │ × │ ■ │ ■ │ ■ │ ■ │  Cannot cross chunks
        └───┴───┴───┴───┴───┴───┴───┴───┘
```

**4. LLAMA_SWA_TYPE_SYMMETRIC - Bidirectional Window:**
```
n_swa = 4 (half = 2): mask when |i - j| > n_swa/2

          Key positions (j)
          0   1   2   3   4   5   6   7
        ┌───┬───┬───┬───┬───┬───┬───┬───┐
     0  │ ■ │ ■ │ ■ │   │   │   │   │   │
     1  │ ■ │ ■ │ ■ │ ■ │   │   │   │   │  Bidirectional!
Q    2  │ ■ │ ■ │ ■ │ ■ │ ■ │   │   │   │  Can see both
(i)  3  │ × │ ■ │ ■ │ ■ │ ■ │ ■ │   │   │  past and future
     4  │ × │ × │ ■ │ ■ │ ■ │ ■ │ ■ │   │  within window
     5  │ × │ × │ × │ ■ │ ■ │ ■ │ ■ │ ■ │
        └───┴───┴───┴───┴───┴───┴───┴───┘  (non-causal)
```

### ISWA (Interleaved SWA) Architecture

Models like Gemma2, Cohere2 use **mixed attention**: some layers use global attention, others use SWA.

```
llama_kv_cache_iswa
├── kv_base  (for non-SWA layers)     ← Full size, stores all history
└── kv_swa   (for SWA layers)         ← Smaller, only stores window

Layer Assignment Example (8 layers):
┌─────────┬──────────────────┬─────────────┐
│  Layer  │  Attention Type  │  KV Cache   │
├─────────┼──────────────────┼─────────────┤
│    0    │  Global          │  kv_base    │
│    1    │  SWA             │  kv_swa     │
│    2    │  Global          │  kv_base    │
│    3    │  SWA             │  kv_swa     │
│   ...   │  ...             │  ...        │
└─────────┴──────────────────┴─────────────┘

Memory Layout:
┌────────────────────────────────────────────────────────┐
│  kv_base (size = kv_size)                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │ pos: 0, 1, 2, 3, 4, 5, ... N                     │  │
│  │ All history preserved for global attention       │  │
│  └──────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────┤
│  kv_swa (size = n_swa + n_ubatch, much smaller!)       │
│  ┌─────────────────────┐                               │
│  │ pos: N-w, ..., N    │  Only recent window           │
│  │ Ring buffer style   │  Old positions overwritten    │
│  └─────────────────────┘                               │
└────────────────────────────────────────────────────────┘
```

### KV Cache Data Flow

```
Token Generation Flow:

1. find_slot()     - Find empty cells for new tokens
2. apply_ubatch()  - Update cell metadata (pos, seq_id)
3. cpy_k/cpy_v()   - Copy K/V tensors to cache (with type conversion)
4. get_k/get_v()   - Get cache views for attention computation

Type Conversion (automatic):
┌─────────────┐    ggml_set_rows()    ┌─────────────┐
│  k_cur      │ ──────────────────→   │  k_cache    │
│  (f32)      │    (quantizes if      │  (f16/q8_0) │
│  from model │     types differ)     │  in cache   │
└─────────────┘                       └─────────────┘
```

### Debugging KV Cache

```bash
# Enable debug output (levels 1-3)
export LLAMA_KV_CACHE_DEBUG=1

# Run with verbose logging
./build-x86/bin/llama-cli -m model.gguf -p "Hello" -n 10 -v

# Test with quantized KV cache
./build-x86/bin/llama-cli -m model.gguf -ctk q8_0 -ctv q8_0 ...

# Check SWA model info
# Look for "n_swa" and "is_swa_any" in model loading output
```

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
