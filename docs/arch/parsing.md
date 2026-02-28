# PEG 解析器架构

> 本文档说明 common 库中的 PEG (Parsing Expression Grammar) 解析器实现。
>
> 引用: `@file:docs/arch/parsing.md`

---

## 概述

PEG 解析器位于 `common/` 库，用于解析模型输出。支持：

- 流式输入的部分解析
- 内置 JSON 解析器
- 带语义标签的 AST 生成

类型前缀：
- `common_peg_*` - 通用 PEG 类型
- `common_chat_peg_*` - 模型输出专用辅助类型

---

## 基本组合子

### 基础匹配器

| 组合子 | 描述 |
|--------|------|
| `eps()` | 空匹配（始终成功） |
| `start()` | 输入开始（锚点 `^`） |
| `end()` | 输入结束（锚点 `$`） |
| `literal(string)` | 精确匹配字符串 |
| `any()` | 任意单个字符（`.`） |

### 组合操作

| 组合子 | 描述 |
|--------|------|
| `sequence(...)` | 顺序匹配（全部必须成功） |
| `choice(...)` | 选择匹配（首个成功的） |
| `one_or_more(p)` | 一次或多次（`+`） |
| `zero_or_more(p)` | 零次或多次（`*`） |
| `optional(p)` | 零次或一次（`?`） |
| `repeat(p, min, max)` | 指定次数范围 |

### 前瞻

| 组合子 | 描述 |
|--------|------|
| `peek(p)` | 正向前瞻（不消费输入） |
| `negate(p)` | 负向前瞻 |

---

## JSON 解析器

| 组合子 | 描述 |
|--------|------|
| `json()` | 完整 JSON |
| `json_object()` | JSON 对象 |
| `json_array()` | JSON 数组 |
| `json_string()` | JSON 字符串 |
| `json_number()` | JSON 数字 |
| `json_member(key, p)` | JSON 对象成员 |

---

## GBNF 语法生成

PEG 解析器也可用于生成 GBNF 语法：

```cpp
data.grammar = build_grammar([&](const common_grammar_builder & builder) {
    parser.build_grammar(builder, data.grammar_lazy);
});
```

**限制**：
- `negate(p)` 不能转换为 CFG 语法
- PEG 要求无歧义语法

### 惰性语法

只有从 `trigger_rule` 可达的规则会被生成。

---

## AST 形状

### Simple 形状

适用于纯内容模型（可选推理）：

```cpp
build_chat_peg_parser([&](common_chat_peg_parser & p) {
    return p.sequence({
        p.optional("<think>" + p.reasoning(p.until("</think>")) + "</think>"),
        p.content(p.until("<tool_call>")),
        p.end()
    });
});
```

标签：
- `reasoning(p)` - 推理内容
- `content(p)` - 输出内容

### Native 形状

适用于工具参数为 JSON 的模型：

标签：
- `tool(p)` - 完整工具调用
- `tool_name(p)` - 工具名称
- `tool_args(p)` - 工具参数
- `tool_id(p)` - 工具 ID（可选）

### Constructed 形状

适用于参数为独立实体（XML 等）的模型：

标签：
- `tool_arg(p)` - 完整参数
- `tool_arg_name(p)` - 参数名
- `tool_arg_string_value(p)` - 字符串值
- `tool_arg_json_value(p)` - JSON 值

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `common/peg-parser.h/cpp` | PEG 解析器核心 |
| `common/chat-peg-parser.h/cpp` | 模型输出专用解析 |
| `tests/test-chat-peg-parser.cpp` | 测试示例 |
