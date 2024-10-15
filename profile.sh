#! /bin/zsh

# MODEL_NAME=llama-2-7B.q4_0
MODEL_NAME=ops-bench

MODEL=/data/gguf/Hermes/baseline/${MODEL_NAME}.gguf

# PROGRAM=./build/bin/llama-cli
# ARGS="-m ${MODEL} -p 'Who are you' -n 128 -ngl 99 -t 1 --seed 2024"

# PROGRAM=./build/bin/llama-bench
# ARGS="-m ${MODEL} -p 512 -n 1 -ngl 99 -t 1"

PROGRAM=./build/bin/test-backend-ops
ARGS="perf -o MUL_MAT -b CUDA0"

NCU=$(which ncu)

# KERNEL_FILTER="--kernel-name mul_mat_vec_q"

rm -rf ./profile_output/ncu_outputs/${MODEL_NAME}.*

CMD="$PROGRAM $ARGS"
# CMD="echo 1 | sudo -S $NCU --set roofline --replay-mode kernel \
#     --target-processes all \
#     -f -o ./profile_output/ncu_outputs/${MODEL_NAME} \
#     $KERNEL_FILTER \
#     $PROGRAM $ARGS"

echo "Running: $CMD"
eval $CMD

