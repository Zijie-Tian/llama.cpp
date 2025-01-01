#!/bin/zsh

echo "Starting batch evaluation..."

export CUDA_VISIBLE_DEVICES=1

# 定义量化模型路径
QUANT_MODEL_PATH=/data/tzj/models/Qwen2.5-0.5B-imatrix/

# 定义 llama.cpp 路径
LLAMA_CPP_PATH=$(realpath $(dirname "$0")/..)

# 定义测试数据集路径
TEST_DATA_PATH=/data/tzj/datasets/wikitext-2-raw/wiki.test.raw

# 定义输出结果文件
OUTPUT_FILE=$LLAMA_CPP_PATH/ppl_results.csv
touch $OUTPUT_FILE

# 进入 llama.cpp 的构建目录
cd $LLAMA_CPP_PATH/build/

# 清空或创建输出文件，并写入表头
echo "Quantization Type,Model Size (GiB),PPL Avg,PPL Std Dev,Prefill Speed (t/s),Prefill Std Dev (t/s),Decode Speed (t/s),Decode Std Dev (t/s)" > $OUTPUT_FILE

# 定义需要排除的量化类型
exclude_types=(    
    "TQ2_0"
    "TQ1_0"
)

# 获取所有量化模型文件
quant_models=($QUANT_MODEL_PATH/ggml-model-*.gguf)

# 遍历每个量化模型并计算 PPL 和速度
for model in "${quant_models[@]}"; do
    # 提取量化类型
    quant_type=$(basename $model | sed 's/ggml-model-\(.*\)\.gguf/\1/')
    
    # 检查当前量化类型是否需要排除
    if [[ " ${exclude_types[@]} " =~ " ${quant_type} " ]]; then
        echo "Skipping excluded quantization type: $quant_type"
        continue
    fi

    echo "Processing quantization type: $quant_type"
    
    # 使用 llama-perplexity 工具计算 PPL，并捕获输出
    echo "Calculating PPL for $quant_type..."
    ppl_output=$(./bin/llama-perplexity --model $model -ngl 99 -t 4 -f $TEST_DATA_PATH 2>&1)
    
    # 从输出中提取 PPL 的平均值和标准差
    ppl_avg=$(echo "$ppl_output" | grep -oP 'Final estimate: PPL = \K[0-9.]+')
    ppl_std=$(echo "$ppl_output" | grep -oP 'Final estimate: PPL = [0-9.]+ ± \K[0-9.]+')
    
    # 如果 ppl_std 为空，则将其设为 0
    if [ -z "$ppl_std" ]; then
        ppl_std=0
    fi
    
    # 使用 llama-bench 工具测试 prefill 和 decode 速度，并捕获输出
    echo "Benchmarking prefill and decode for $quant_type..."
    bench_output=$(./bin/llama-bench -m $model -n 128 -t 12 -b 64 -ngl 100 -fa 1 2>&1)
    
    # 从输出中提取模型大小（匹配 GiB 前面的数字，只取第一个匹配项）
    model_size=$(echo "$bench_output" | grep -oP '\|\s*\K[0-9.]+(?=\s*GiB)' | head -n 1)
    
    # 从输出中提取 prefill (pp) 和 decode (tg) 速度
    prefill_speed=$(echo "$bench_output" | grep -oP 'pp[^|]+\s*\|\s*\K[0-9.]+ ± [0-9.]+')
    decode_speed=$(echo "$bench_output" | grep -oP 'tg[^|]+\s*\|\s*\K[0-9.]+ ± [0-9.]+')
    
    # 拆分 prefill 速度的平均值和标准差
    prefill_avg=$(echo "$prefill_speed" | awk -F' ± ' '{print $1}')
    prefill_std=$(echo "$prefill_speed" | awk -F' ± ' '{print $2}')
    
    # 拆分 decode 速度的平均值和标准差
    decode_avg=$(echo "$decode_speed" | awk -F' ± ' '{print $1}')
    decode_std=$(echo "$decode_speed" | awk -F' ± ' '{print $2}')
    
    # 将结果写入 CSV 文件
    echo "$quant_type,$model_size,$ppl_avg,$ppl_std,$prefill_avg,$prefill_std,$decode_avg,$decode_std" >> $OUTPUT_FILE
    
    echo "Results for $quant_type:"
    echo "Model Size: $model_size GiB"
    echo "PPL: $ppl_avg ± $ppl_std"
    echo "Prefill speed: $prefill_avg ± $prefill_std"
    echo "Decode speed: $decode_avg ± $decode_std"
    echo "--------------------------------------------------"
done

echo "Batch evaluation completed. Results saved to $OUTPUT_FILE"