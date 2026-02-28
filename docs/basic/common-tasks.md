# llama.cpp 常见任务

> 快速参考常见开发任务的操作步骤。
>
> 引用: `@file:docs/basic/common-tasks.md`

---

## 运行示例

### 基本对话

```bash
# 本地模型
./build/bin/llama-cli -m model.gguf -cnv

# 从 Hugging Face 下载并运行
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# 指定量化版本
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF:Q4_K_M
```

### 启动服务器

```bash
# 基本服务器
./build/bin/llama-server -m model.gguf --port 8080

# 多用户并行
./build/bin/llama-server -m model.gguf -c 16384 -np 4

# 推测解码 (speculative decoding)
./build/bin/llama-server -m model.gguf -md draft.gguf
```

---

## 模型转换

### Hugging Face → GGUF

```bash
# 基本转换
python convert_hf_to_gguf.py /path/to/hf-model --outfile model.gguf

# 指定上下文长度
python convert_hf_to_gguf.py /path/to/hf-model --outfile model.gguf --ctx-size 32768
```

### 量化模型

```bash
# Q4_0 量化 (速度快，质量较低)
./build/bin/llama-quantize model.gguf model-q4_0.gguf Q4_0

# Q4_K_M 量化 (平衡)
./build/bin/llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M

# Q5_K_M 量化 (质量更高)
./build/bin/llama-quantize model.gguf model-q5_k_m.gguf Q5_K_M

# Q8_0 量化 (接近无损)
./build/bin/llama-quantize model.gguf model-q8_0.gguf Q8_0

# 使用重要性矩阵 (更高质量)
./build/bin/llama-quantize model.gguf model-imat.gguf Q4_K_M imatrix.dat
```

### 生成重要性矩阵

```bash
./build/bin/llama-imatrix -m model.gguf -f train.txt -o imatrix.dat
```

---

## 性能调优

### GPU 层卸载

```bash
# 自动选择最佳层数
./build/bin/llama-cli -m model.gguf -ngl 99

# 手动指定层数
./build/bin/llama-cli -m model.gguf -ngl 35

# CPU-only
./build/bin/llama-cli -m model.gguf -ngl 0
# 或
./build/bin/llama-cli -m model.gguf --device none
```

### 线程设置

```bash
# 指定线程数
./build/bin/llama-cli -m model.gguf -t 8

# 提示处理批量
./build/bin/llama-cli -m model.gguf -ub 512
```

### 内存优化

```bash
# 启用内存映射 (默认)
./build/bin/llama-cli -m model.gguf --mlock

# 禁用内存映射 (低内存设备)
./build/bin/llama-cli -m model.gguf --no-mmap

# 启用统一内存 (Linux CUDA)
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./llama-cli -m model.gguf
```

---

## 调试

### Sanitizer 构建

```bash
# Address Sanitizer (内存错误)
cmake -B build -DLLAMA_SANITIZE_ADDRESS=ON
cmake --build build

# Thread Sanitizer (数据竞争)
cmake -B build -DLLAMA_SANITIZE_THREAD=ON
cmake --build build
```

### 详细日志

```bash
# 启用详细日志
./build/bin/llama-cli -m model.gguf -lv 1

# 查看后端选择
./build/bin/llama-cli -m model.gguf --verbose
```

---

## GGUF 操作

### 查看信息

```bash
# 使用 gguf-py
python -m gguf.gguf_dump model.gguf

# 提取张量
python -m gguf.gguf_new_metadata --model model.gguf --output info.txt
```

### 合并/分割

```bash
# 分割大模型
./build/bin/llama-gguf-split --split-max-size 5G model.gguf model-split.gguf

# 合并
./build/bin/llama-gguf-split --merge model-split-00001-of-00004.gguf model-merged.gguf
```

---

## 环境变量

### CUDA

```bash
# 选择 GPU
CUDA_VISIBLE_DEVICES=0,1 ./llama-server -m model.gguf

# 隐藏第一个 GPU
CUDA_VISIBLE_DEVICES="-0" ./llama-server -m model.gguf

# 扩展命令缓冲区
CUDA_SCALE_LAUNCH_QUEUES=4x ./llama-server -m model.gguf
```

### 其他后端

```bash
# HIP (AMD)
HIP_VISIBLE_DEVICES=0 ./llama-server -m model.gguf
HSA_OVERRIDE_GFX_VERSION=10.3.0 ./llama-server -m model.gguf  # 非官方支持 GPU

# Vulkan
VK_ICD_FILENAMES=/path/to/icd.json ./llama-server -m model.gguf
```

---

## 批量处理

```bash
# 批量生成
./build/bin/llama-cli -m model.gguf -f prompts.txt -o outputs.txt

# 批处理 bench
./build/bin/llama-batched-bench -m model.gguf -p 512,1024 -n 128,256
```
