# 推测解码架构

> 本文档说明 llama.cpp 中推测解码的实现。
>
> 引用: `@file:docs/arch/speculative.md`

---

## 概述

[推测解码](https://en.wikipedia.org/wiki/Transformer_(deep_learning)#Speculative_decoding) 通过预测多个 token 并批量验证来加速生成。

核心思想：批量计算 n 个 token 比顺序计算 n 个 token 更高效。

---

## 实现类型

### 1. Draft Model (`draft`)

使用较小的草稿模型生成候选 token。

最常用方法，需要额外的草稿模型。

### 2. N-gram Cache (`ngram-cache`)

基于 n-gram 统计生成草稿，从已生成的文本中学习模式。

### 3. N-gram Map

搜索 token 历史中的模式，使用匹配序列作为草稿。

| 类型 | 描述 |
|------|------|
| `ngram-simple` | 查找历史匹配，使用后续 token |
| `ngram-map-k` | 使用哈希表查找 n-gram |
| `ngram-map-k4v` | 每个 key 跟踪最多 4 个 value |
| `ngram-mod` | 使用 LCG 哈希，共享哈希池 |

---

## 架构流程

```
1. 生成阶段
   - 草稿模型/n-gram 生成候选 token 序列
   - 形成 speculative batch

2. 验证阶段
   - 主模型批量验证候选 token
   - 接受匹配的 token前缀

3. 回退阶段
   - 从第一个不匹配处重新生成
```

---

## 关键参数

| 参数 | 说明 |
|------|------|
| `--draft-max N` | 最大草稿 token 数 |
| `--draft-min N` | 最小草稿 token 数 |
| `--spec-type TYPE` | 推测解码类型 |
| `--spec-ngram-size-n N` | n-gram 查找长度 |
| `--spec-ngram-size-m M` | m-gram 生成长度 |
| `--spec-ngram-min-hits H` | 最小命中次数 |

---

## 统计数据

```
draft acceptance rate = 0.57576 (171 accepted / 297 generated)
```

指标：
- `#calls`: 调用次数
- `#gen drafts`: 生成的草稿数
- `#acc drafts`: 接受的草稿数
- `#gen tokens`: 生成的 token 数
- `#acc tokens`: 接受的 token 数

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `common/speculative.h/cpp` | 推测解码核心 |
| `common/ngram-cache.h/cpp` | N-gram 缓存 |
| `src/llama-context.cpp` | 批量解码逻辑 |
