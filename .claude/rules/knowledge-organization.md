---
name: knowledge-organization
description: |
  规范 llama.cpp 项目的知识组织方式。
  强制要求分层文档架构，禁止在 CLAUDE.md 中写入详细代码知识。
  确保代码知识按领域分类存储，通过 @path 引用访问。

  使用场景:
  - 用户要求创建或修改文档时
  - 需要确定知识存放位置时
  - 审查文档结构是否符合规范时

  触发条件:
  - 任何涉及 docs/ 目录的修改
  - 创建新文档的请求
  - 知识整理或重构任务

  必须遵守:
  - 代码知识必须下沉到 docs/{basic,arch,backend,ops}/
  - CLAUDE.md 只保留索引和规则，禁止详细实现
  - 通过 @path 引用下层文档
alwaysApply: true
---

# 知识组织规范 (Knowledge Organization Rule)

## 分层架构（强制）

本项目采用三层知识架构，所有文档必须按此规范存放：

### 第一层：项目入口

**位置**: `CLAUDE.md`（仓库根目录）

**允许内容**:
- 项目级规则和约束
- 文档索引表格
- 快速命令参考（不超过 20 行）
- 外部文档链接

**禁止内容**:
- ❌ 超过 50 行的详细实现说明
- ❌ 完整的代码示例
- ❌ 详细的 API 文档
- ❌ 重复的构建命令列表

**引用方式**: 直接访问（自动加载）

### 第二层：Claude Code 配置

**位置**: `docs/claude/`

**内容**:
- Claude Code 配置机制说明
- 子代理 (Subagents) 配置指南
- Skills 和 Hooks 使用说明
- 项目特定的 Claude 工作流

**当前文件**:
- `claude-code-config.md` - 配置机制总览

**引用方式**: `@docs/claude/<file>.md`

### 第三层：代码知识（按领域分类）

**位置**: `docs/{basic,arch,backend,ops}/`

| 分类 | 目录 | 内容范围 | 示例 |
|------|------|----------|------|
| **basic** | `docs/basic/` | 构建、测试、贡献、常见任务 | build-system.md, testing.md |
| **arch** | `docs/arch/` | 代码架构、组件说明、数据流 | overview.md, model-loading.md |
| **backend** | `docs/backend/` | 后端系统、硬件支持、调度 | overview.md, cuda-backend.md |
| **ops** | `docs/ops/` | 算子实现、优化、融合 | overview.md, quantization-ops.md |

**引用方式**: `@docs/<category>/<file>.md`

## 强制规则

### Rule 1: 禁止在 CLAUDE.md 写入详细代码知识

**违规示例**:
```markdown
# ❌ 错误：在 CLAUDE.md 中写入详细构建选项

## CUDA 构建选项

| 选项 | 说明 |
|------|------|
| GGML_CUDA_FORCE_MMQ | 强制使用量化矩阵乘法核 |
| GGML_CUDA_FORCE_CUBLAS | 强制使用 cuBLAS FP16 |
| ... (超过 50 行) |
```

**正确做法**:
```markdown
# ✅ 正确：在 CLAUDE.md 中只保留引用

详见 @docs/basic/build-system.md
```

### Rule 2: 代码知识必须下沉到对应分类

| 知识类型 | 必须存放位置 |
|----------|-------------|
| CMake 构建选项 | `docs/basic/build-system.md` |
| 测试命令和说明 | `docs/basic/testing.md` |
| 代码风格规范 | `docs/basic/contribution.md` |
| 架构总览 | `docs/arch/overview.md` |
| 后端实现细节 | `docs/backend/overview.md` |
| 算子实现 | `docs/ops/overview.md` |

### Rule 3: 必须通过 @path 引用下层文档

**对话中引用**:
```
用户: 如何构建 CUDA 版本？
助手: 请参考 @docs/basic/build-system.md
```

**CLAUDE.md 中引用**:
```markdown
详见 @docs/basic/build-system.md
```

### Rule 4: 创建新文档时必须选择正确分类

**决策流程**:

```
创建新文档？
├── 是关于 Claude Code 配置？ → docs/claude/
├── 是关于构建/测试/贡献？ → docs/basic/
├── 是关于代码架构？ → docs/arch/
├── 是关于后端/硬件？ → docs/backend/
└── 是关于算子/优化？ → docs/ops/
```

## 文档创建模板

### 基础文档 (docs/basic/)

```markdown
# 标题

> 一句话描述本文档内容。
> 引用: `@docs/basic/<name>.md`

---

## 快速开始

简要说明和命令。

## 详细说明

详细内容...

## 参考

- 相关文档链接
```

### 架构文档 (docs/arch/)

```markdown
# 标题

> 本文档描述 xxx 的架构设计。
> 引用: `@docs/arch/<name>.md`

---

## 架构图

## 组件说明

## 数据流

## 关键接口
```

### 后端文档 (docs/backend/)

```markdown
# 标题

> 本文档说明 xxx 后端的实现。
> 引用: `@docs/backend/<name>.md`

---

## 后端架构

## 关键实现

## 性能优化

## 调试方法
```

### 算子文档 (docs/ops/)

```markdown
# 标题

> 本文档描述 xxx 算子的实现。
> 引用: `@docs/ops/<name>.md`

---

## 算子定义

## 实现位置

## 优化策略

## 后端特定实现
```

## 违规检查清单

在提交文档变更前，确认：

- [ ] CLAUDE.md 中没有超过 50 行的详细实现
- [ ] 代码知识已下沉到 docs/{basic,arch,backend,ops}/
- [ ] 使用了正确的 `@path` 引用格式
- [ ] 新文档选择了正确的分类目录
- [ ] 文档头部包含引用路径说明

## 示例场景

### 场景 1: 用户询问构建选项

```
用户: CUDA 有哪些构建选项？

助手操作:
1. 不直接在回答中列出所有选项
2. 引用: "详见 @docs/basic/build-system.md"
3. 如果需要，读取该文件后提供摘要
```

### 场景 2: 需要创建新文档

```
用户: 请记录一下 Metal 后端的知识

助手操作:
1. 判断分类: 属于后端 → docs/backend/
2. 创建文件: docs/backend/metal.md
3. 在 CLAUDE.md 中添加引用
4. 不在 CLAUDE.md 中写入详细内容
```

### 场景 3: 发现 CLAUDE.md 过于臃肿

```
发现: CLAUDE.md 中有 100 行的构建说明

助手操作:
1. 将内容迁移到 docs/basic/build-system.md
2. 在 CLAUDE.md 中替换为引用链接
3. 确保不丢失任何信息
```
