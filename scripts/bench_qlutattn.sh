#!/bin/bash

# Configurable parameters
THREADS=12
MODEL_PATH="/data/gguf/Llama-2-7B-GGUF/llama-2-7b.Q2_K.gguf"

# Binary path
BENCH_BIN="./build-arm64/bin/llama-bench"

# Fixed parameters
N_GEN=32
CTK="q4_0"
CTV="q4_0"
NGL=0
FA=1

# Array of n-prompt values to test
N_PROMPT_VALUES=(16384 32768 65536)  # 16*1024, 32*1024, 64*1024

echo "========================================="
echo "QLUT Attention Benchmark Test"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Threads: $THREADS"
echo "========================================="
echo

# Iterate through different n-prompt values
for N_PROMPT in "${N_PROMPT_VALUES[@]}"; do
    echo "Testing with n-prompt: $N_PROMPT ($(($N_PROMPT / 1024))K)"
    echo "-----------------------------------------"

    # Run the benchmark
    $BENCH_BIN \
        -m "$MODEL_PATH" \
        --n-prompt $N_PROMPT \
        --n-gen $N_GEN \
        -ctk $CTK \
        -ctv $CTV \
        -ngl $NGL \
        -t $THREADS \
        -fa $FA

    echo
    echo "========================================="
    echo
done

echo "Benchmark testing completed!"