# 性能优化

> 本文档说明 llama.cpp 的 token 生成性能优化。
>
> 引用: `@file:docs/basic/performance.md`

---

## GPU 验证

确保模型在 GPU 上运行：

```bash
./llama-cli -m model.gguf -ngl 200000 -p "test"
```

查看输出中的 cuBLAS 信息：

```
llama_model_load_internal: [cublas] offloading 60 layers to GPU
llama_model_load_internal: [cublas] total VRAM used: 17223 MB
```

---

## CPU 线程优化

### 避免超线程饱和

`-t` 参数非常重要。如果生成速度极慢，尝试设置为 1：

```bash
# 测试单线程
./llama-cli -m model.gguf -t 1

# 逐步增加，找到最佳值
./llama-cli -m model.gguf -t 4
./llama-cli -m model.gguf -t 7  # 物理核心数
```

### 推荐设置

设置为物理 CPU 核心数（非逻辑核心）。

---

## 性能对比示例

配置：A6000 (48GB), 7 物理核心, 32GB RAM

模型：30B 参数, Q4_0 量化

| 配置 | tokens/秒 |
|------|----------|
| `-ngl 2000000` | < 0.1 |
| `-t 7` | 1.7 |
| `-t 1 -ngl 2000000` | 5.5 |
| `-t 7 -ngl 2000000` | 8.7 |
| `-t 4 -ngl 2000000` | 9.1 |

---

## 性能工具

### llama-bench

```bash
./llama-bench -m model.gguf
```

### llama-perplexity

```bash
./llama-perplexity -m model.gguf -f test.txt
```
