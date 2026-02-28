# 功能调用架构

> 本文档说明 llama.cpp 中 OpenAI 风格功能调用的实现架构。
>
> 引用: `@file:docs/arch/function-calling.md`

---

## 概述

`common/chat.h` 实现了 OpenAI 风格功能调用，用于：

- `llama-server`（带 `--jinja` 标志）

---

## 支持格式

### 原生格式

- Llama 3.1 / 3.3（内置工具支持）
- Llama 3.2
- Functionary v3.1 / v3.2
- Hermes 2/3, Qwen 2.5
- Mistral Nemo
- Firefunction v2
- Command R7B
- DeepSeek R1

### 通用格式

当模板未被原生识别时使用（日志显示 `Chat format: Generic`）。

可通过 `--chat-template-file` 覆盖模板。

---

## 架构组件

### Chat Handler

位于 `common/chat.h/cpp`，处理：

- 工具定义解析
- 消息格式化
- 工具调用提取

### Jinja 模板引擎

位于 `common/jinja/`，用于：

- 渲染聊天模板
- 处理工具定义

### PEG 解析器

用于从模型输出中提取工具调用：

- Native: 解析 JSON 格式工具调用
- Constructed: 解析 XML/标签格式

---

## 并行工具调用

部分模型支持多/并行工具调用，默认禁用。

启用方式：在请求中设置 `"parallel_tool_calls": true`。

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `common/chat.h/cpp` | 功能调用核心 |
| `common/jinja/` | Jinja 模板引擎 |
| `common/chat-peg-parser.h/cpp` | 工具调用解析 |
| `models/templates/*.jinja` | 模型模板 |
