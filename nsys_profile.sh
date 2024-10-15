#!/bin/zsh

NAME="llama-2-7B.q4_0"

# Define the program and arguments
PROGRAM="./build/bin/llama-cli"  # 请替换为你要测试的实际程序路径
ARGS="-m /data/gguf/Hermes/baseline/${NAME}.gguf -p \"Who are you?\" -n 128 -ngl 99 -t 12 --seed 2024"        # 请替换为程序的参数

# nsys profile --wait primary --force-overwrite true --cuda-memory-usage true -o /home/edgellm/Code/llama.cpp/nsys_outputs/${NAME} \
CMD="nsys profile --wait primary --force-overwrite true --cuda-memory-usage true -o /home/edgellm/Code/llama.cpp/nsys_outputs/${NAME} ${PROGRAM} ${ARGS}"

# Run the nsys profiler
# nsys profile --stats=true --output="${OUTPUT_NAME}" ${PROGRAM} ${ARGS}

echo "Running: ${CMD}"
eval ${CMD}

# Display a message to confirm completion
echo "nsys profiling complete."
