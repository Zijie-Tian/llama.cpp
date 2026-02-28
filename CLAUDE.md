# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 必须首先阅读

- **[AGENTS.md](AGENTS.md)** - ⚠️ **AI 使用政策**: 本项目不接受完全 AI 生成的 PR

---

## 知识组织

本项目采用分层知识架构，详细规范见：
**@.claude/rules/knowledge-organization.md**

### 引用方式说明

| 类型 | 标记方式 | 加载行为 | 用途 |
|------|----------|----------|------|
| **热引用** | `@path` | 自动加载 | Claude Code 配置 |
| **冷引用** | `[名称](path)` | 按需访问 | 代码知识 |

---

## 热引用文档（自动加载）

### Claude Code 配置

**@docs/claude/claude-code-config.md** - Claude Code 配置机制、子代理、Skills、Hooks

---

## 冷引用文档（按需访问）

### 基础 (docs/basic/)

| 文档 | 说明 |
|------|------|
| [build-system.md](docs/basic/build-system.md) | 构建系统 |
| [testing.md](docs/basic/testing.md) | 测试指南 |
| [contribution.md](docs/basic/contribution.md) | 贡献规范 |
| [common-tasks.md](docs/basic/common-tasks.md) | 常见任务 |
| [debugging.md](docs/basic/debugging.md) | 调试测试 |
| [performance.md](docs/basic/performance.md) | 性能优化 |

### 架构 (docs/arch/)

| 文档 | 说明 |
|------|------|
| [overview.md](docs/arch/overview.md) | 架构总览 |
| [add-model.md](docs/arch/add-model.md) | 添加新模型 |
| [parsing.md](docs/arch/parsing.md) | PEG 解析器 |
| [function-calling.md](docs/arch/function-calling.md) | 功能调用 |
| [speculative.md](docs/arch/speculative.md) | 推测解码 |

### 后端 (docs/backend/)

| 文档 | 说明 |
|------|------|
| [overview.md](docs/backend/overview.md) | 后端系统 |

### 算子 (docs/ops/)

| 文档 | 说明 |
|------|------|
| [overview.md](docs/ops/overview.md) | 算子说明 |

---

## 核心规则

### AI 使用政策

> 本项目 **不接受** 完全或主要由 AI 生成的 PR。
> AI 只能作为辅助工具，大多数代码必须由人工编写。
> 使用 AI 必须**明确披露**。

### 修改前检查清单

| 修改范围 | 必须执行 |
|----------|----------|
| `ggml/` 目录 | 运行 `./build/bin/test-backend-ops` |
| 性能敏感代码 | 运行 `./build/bin/llama-bench` 对比前后 |
| tokenizer 相关 | 运行 `./build/bin/test-tokenizer-0` |
| 任何代码变更 | 本地运行 `bash ./ci/run.sh ./tmp/results ./tmp/mnt` |

### 文档引用规范

- **热引用**（Claude 配置）：使用 `@docs/claude/xxx.md`
- **冷引用**（代码知识）：使用 `docs/xxx/xxx.md`，对话中需要时再用 `@path` 加载

---

## 快速参考

### 构建

```bash
cmake -B build && cmake --build build --config Release -j$(nproc)
```

### 测试

```bash
cd build && ctest -L main --verbose --timeout 900
```

### 代码架构

```
ggml/          - 张量计算库 (CPU/CUDA/Metal/Vulkan/...)
src/           - llama 推理引擎 (model, context, kv-cache, sampler)
common/        - 共享工具 (arg, chat, sampling)
tools/         - 可执行工具 (cli, server, quantize)
include/       - C/C++ API 头文件
```

---

## 外部文档

- [docs/build.md](docs/build.md) - 官方构建文档（所有后端）
- [tools/server/README-dev.md](tools/server/README-dev.md) - llama-server 架构
- [grammars/README.md](grammars/README.md) - GBNF 语法约束
- [CONTRIBUTING.md](CONTRIBUTING.md) - 完整贡献指南

---

*CLAUDE.md 版本: 2025-02-28*
