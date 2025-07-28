#!/usr/bin/env bash
set -euo pipefail

# 使用方式: ./convert_hf_to_gguf.sh /path/to/model_dir [target_dir]
MODEL_DIR="$1"
TARGET_DIR="${2:-$MODEL_DIR}" # 如果没有提供第二个参数，使用模型目录作为目标目录

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: 模型目录 '$MODEL_DIR' 不存在"
    exit 1
fi

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 定义支持的输出类型
TYPES=("f32" "f16" "q8_0" "tq1_0" "tq2_0" "tmac_w2g64_0" "tmac_w2g64_1" "tmac_w2g128_0" "tmac_w2g128_1" "tmac_w4g64_0" "tmac_w4g64_1" "tmac_w4g128_0" "tmac_w4g128_1")

# 基础模型名从目录名称提取
MODEL_NAME="$(basename "$MODEL_DIR")"

for t in "${TYPES[@]}"; do
    OUTFILE="${TARGET_DIR}/${MODEL_NAME}-${t}.gguf"

    # 如果文件已存在，跳过转换
    if [[ -f "$OUTFILE" ]]; then
        echo "跳过 $t: 文件 '$OUTFILE' 已存在"
        continue
    fi

    echo "Converting to $t: output => $OUTFILE"

    # 检查是否为 tmac 类型，如果是则添加 --enable-t-mac 参数
    if [[ "$t" == tmac_* ]]; then
        python convert_hf_to_gguf.py \
            --outtype "$t" \
            --outfile "$OUTFILE" \
            --enable-t-mac \
            "$MODEL_DIR"
    else
        python convert_hf_to_gguf.py \
            --outtype "$t" \
            --outfile "$OUTFILE" \
            "$MODEL_DIR"
    fi
done

echo "All done!"
