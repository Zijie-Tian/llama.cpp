#!/bin/zsh

#region Quantization types
# Define the quantization types and their bpw separately
quant_types=(
    "Q4_0"
    "Q4_1"
    "Q5_0"
    "Q5_1"
    # "IQ2_XXS"
    # "IQ2_XS"
    # "IQ2_S"
    # "IQ2_M"
    # "IQ1_S"
    # "IQ1_M"
    "TQ1_0"
    "TQ2_0"
    "Q2_K"
    "Q2_K_S"
    # "IQ3_XXS"
    # "IQ3_S"
    # "IQ3_M"
    "Q3_K"
    # "IQ3_XS"
    "Q3_K_S"
    "Q3_K_M"
    "Q3_K_L"
    # "IQ4_NL"
    # "IQ4_XS"
    "Q4_K"
    "Q4_K_S"
    "Q4_K_M"
    "Q5_K"
    "Q5_K_S"
    "Q5_K_M"
    "Q6_K"
    "Q8_0"
    "F16"
    # "BF16"
    # "F32"
)

bpw_values=(
    ""          # Q4_0
    ""          # Q4_1
    ""          # Q5_0
    ""          # Q5_1
    "2.06"      # IQ2_XXS
    "2.31"      # IQ2_XS
    "2.5"       # IQ2_S
    "2.7"       # IQ2_M
    "1.56"      # IQ1_S
    "1.75"      # IQ1_M
    "1.69"      # TQ1_0
    "2.06"      # TQ2_0
    "2.5625"    # Q2_K
    "2.5625"    # Q2_K_S
    "3.06"      # IQ3_XXS
    "3.44"      # IQ3_S
    "3.66"      # IQ3_M
    "3.4375"    # Q3_K
    "3.3"       # IQ3_XS
    "3.4375"    # Q3_K_S
    "3.4375"    # Q3_K_M
    "3.4375"    # Q3_K_L
    "4.50"      # IQ4_NL
    "4.25"      # IQ4_XS
    "4.5"       # Q4_K
    "4.5"       # Q4_K_S
    "4.5"       # Q4_K_M
    "5.5"       # Q5_K
    "5.5"       # Q5_K_S
    "5.5"       # Q5_K_M
    "6.5625"    # Q6_K
    ""          # Q8_0
    "16"        # F16
    # "16"      # BF16
    # "32"      # F32
)

#endregion

LLAMA_CPP_PATH=$(realpath $(dirname "$0")/..)

HF_MODEL_PATH=/data/tzj/huggingface/Qwen2.5-0.5B
QUANT_MODEL_PATH=/data/tzj/models/Qwen2.5-0.5B-quant/

mkdir -p $QUANT_MODEL_PATH

# Check if the f16 file already exists
if [ -f "$QUANT_MODEL_PATH/ggml-model-F16.gguf" ]; then
    echo "f16 file already exists. Skipping conversion."
else
    cd $LLAMA_CPP_PATH
    python convert_hf_to_gguf.py $HF_MODEL_PATH --outtype f16 --outfile $QUANT_MODEL_PATH/ggml-model-F16.gguf
fi

cd $LLAMA_CPP_PATH/build/

# Loop through each quantization type and execute the command
for i in {1..${#quant_types[@]}}; do
    quant_type=${quant_types[i]}
    bpw=${bpw_values[i]}
    output_file="$QUANT_MODEL_PATH/ggml-model-${quant_type}.gguf"
    
    if [ -f "$output_file" ]; then
        echo "File for quantization type $quant_type already exists. Skipping."
    else
        echo "Processing quantization type: $quant_type with bpw: $bpw BPW"
        ./bin/llama-quantize $QUANT_MODEL_PATH/ggml-model-F16.gguf $output_file ${quant_type}
    fi
done


