#!/bin/zsh

# NAME="llama-2-7B.q4_0"
NAME="hf-bitnet-3B"

# Define the program and arguments
PROGRAM="./build/bin/llama-cli"  # 请替换为你要测试的实际程序路径
ARGS="-m /data/gguf/Bitnet/${NAME}.gguf -p \"Write a resignation letter.\" -n 128 -ngl 99 -t 4 -c 2048"        # 请替换为程序的参数

# nsys profile --wait primary --force-overwrite true --cuda-memory-usage true -o /home/edgellm/Code/llama.cpp/nsys_outputs/${NAME} \
CMD="nsys profile --wait primary --force-overwrite true --cuda-memory-usage true -o /home/edgellm/Code/llama.cpp/nsys_outputs/${NAME} ${PROGRAM} ${ARGS}"
# CMD="${PROGRAM} ${ARGS}"
# Run the nsys profiler
# nsys profile --stats=true --output="${OUTPUT_NAME}" ${PROGRAM} ${ARGS}

echo "Running: ${CMD}"
eval ${CMD}

# Display a message to confirm completion
echo "nsys profiling complete."
