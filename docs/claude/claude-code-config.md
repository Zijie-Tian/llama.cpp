# Claude Code 配置机制

> 本文档详细说明 Claude Code 的 Rule、Command、Agent、Skill 和 Hook 的配置方法。
>
> 引用: `@docs/claude/claude-code-config.md`

---

## 目录结构

```
project-root/
├── .claude/
│   ├── rules/           # 项目级规则
│   │   └── *.md
│   ├── commands/        # 自定义命令
│   │   └── *.md
│   ├── agents/          # 子代理
│   │   └── *.md
│   ├── skills/          # 技能
│   │   └── */SKILL.md
│   └── memory/          # 持久化记忆
│       └── MEMORY.md
└── CLAUDE.md            # 项目入口
```

---

## Rules（规则）

Rules 用于定义项目级的行为规范和约束。

### 文件位置

- `.claude/rules/*.md` - 项目级规则
- `~/.claude/rules/*.md` - 用户级规则（所有项目）

### 文件格式

```yaml
---
name: rule-name
description: |
  规则的详细描述。
  说明规则的作用和使用场景。

  使用场景:
  - 何时触发此规则
  - 适用的文件类型

  触发条件:
  - 特定关键词
  - 文件模式匹配

  必须遵守:
  - 具体的约束条件
alwaysApply: true  # 是否始终生效
---

# 规则正文

## 详细说明

规则的具体内容和示例。
```

### Frontmatter 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 规则名称（唯一标识） |
| `description` | string | 规则描述，支持多行 |
| `alwaysApply` | boolean | 是否始终生效（默认 false） |

### 示例：知识组织规则

```yaml
---
name: knowledge-organization
description: |
  规范项目的知识组织方式。
  强制代码知识下沉到 docs/ 目录。

  使用场景:
  - 用户要求创建文档时
  - 审查文档结构时

  触发条件:
  - 任何 docs/ 目录的修改

  必须遵守:
  - 代码知识在 docs/{basic,arch,backend,ops}/
  - CLAUDE.md 只保留引用
alwaysApply: true
---

# 规则正文...
```

---

## Commands（自定义命令）

Commands 允许定义自定义的斜杠命令（如 `/command-name`）。

### 文件位置

- `.claude/commands/*.md`
- `~/.claude/commands/*.md`

### 文件格式

```yaml
---
name: command-name
description: 命令的简短描述
---

# 命令执行的提示词模板

{{ARG1}} 和 {{ARG2}} 是用户传入的参数。

## 参数说明

- ARG1: 第一个参数
- ARG2: 第二个参数
```

### 使用方式

在 Claude Code 中执行：
```
/command-name arg1 arg2
```

### 示例

```yaml
---
name: review-code
description: 对指定文件进行代码审查
---

请对以下文件进行代码审查：

文件: {{FILE}}

审查要点：
1. 代码风格是否符合项目规范
2. 是否存在潜在 bug
3. 性能问题
4. 安全漏洞
```

使用：
```
/review-code src/main.cpp
```

---

## Agents（子代理）

Agents 是专门的 AI 助手，可以执行特定任务。

### 文件位置

| 优先级 | 位置 | 作用域 |
|--------|------|--------|
| 1 | `--agents` CLI 参数 | 当前会话 |
| 2 | `.claude/agents/` | 当前项目 |
| 3 | `~/.claude/agents/` | 所有项目 |
| 4 | Plugin 的 `agents/` | Plugin 启用处 |

### 文件格式

```yaml
---
name: agent-name
description: |
  Agent 的详细描述。
  说明其专长和使用场景。
prompt: |
  你是 [角色描述]，你的任务是...
tools: [Read, Grep, Glob, Bash, Edit]  # 允许的工具
disallowedTools: []  # 禁止的工具
model: sonnet  # 使用的模型
permissionMode: acceptEdits  # 权限模式
maxTurns: 50  # 最大对话轮数
skills: [security-audit]  # 启动时加载的技能
mcpServers: []  # MCP 服务器
hooks: []  # 生命周期钩子
background: false  # 是否后台运行
isolation: worktree  # 隔离模式
---
```

### 权限模式

| 模式 | 说明 |
|------|------|
| `default` | 默认权限，每次修改询问 |
| `acceptEdits` | 自动接受编辑请求 |
| `dontAsk` | 不询问，直接执行 |
| `bypassPermissions` | 完全绕过权限检查 |
| `plan` | 仅生成计划，不执行 |

### 隔离模式

| 模式 | 说明 |
|------|------|
| `worktree` | 在临时 git worktree 中运行 |
| `none` | 在当前目录运行 |

### 示例：代码审查代理

```yaml
---
name: code-reviewer
description: |
  专家级代码审查员。
  在代码变更后主动使用，检查代码质量。

  专长:
  - C/C++ 代码审查
  - 性能优化建议
  - 安全漏洞检测

prompt: |
  你是资深代码审查员，有 20 年 C/C++ 经验。
  你的任务是审查代码变更，发现：
  1. 潜在的 bug 和逻辑错误
  2. 性能瓶颈
  3. 代码风格问题
  4. 安全隐患
  5. 可维护性问题

tools: [Read, Grep, Glob, Bash]
model: sonnet
permissionMode: acceptEdits
maxTurns: 50
skills: [security-audit]
background: false
isolation: worktree
---
```

### 调用代理

```bash
# 命令行
claude --agent code-reviewer

# 对话中
@code-reviewer 请审查这个 PR
```

---

## Skills（技能）

Skills 扩展 Claude 的能力，可以包含工具、资源和提示词模板。

### 文件位置

- `.claude/skills/skill-name/SKILL.md`
- `~/.claude/skills/skill-name/SKILL.md`

### 目录结构

```
.claude/skills/my-skill/
├── SKILL.md           # 技能定义
├── prompt.md          # 提示词模板（可选）
├── tools/             # 自定义工具（可选）
└── resources/         # 资源文件（可选）
```

### 文件格式

```yaml
---
name: skill-name
description: 技能的简短描述
eager: false  # 是否立即加载
userInvocable: true  # 用户是否可调用
disableModelInvocation: false  # 禁止模型自动调用
---

# 技能正文

## 工具

定义技能提供的工具。

## 提示词

技能的提示词模板。
```

### 调用控制

| Frontmatter | 用户可调用 | 模型可调用 |
|-------------|------------|------------|
| (默认) | Yes | Yes |
| `disable-model-invocation: true` | Yes | No |
| `user-invocable: false` | No | Yes |

### 加载模式

| 模式 | 配置 | 行为 |
|------|------|------|
| Eager（热加载） | `eager: true` | 会话启动时立即加载 |
| Lazy（懒加载） | `eager: false` | 需要时才加载 |

---

## Hooks（钩子）

Hooks 提供生命周期自动化，在特定事件触发时执行。

### 钩子类型

| Hook | 触发时机 | 可阻止？ |
|------|----------|----------|
| `PreToolUse` | 工具调用前 | ✅ |
| `PostToolUse` | 工具调用完成后 | ❌ |
| `PermissionRequest` | 权限对话框显示时 | ✅ |
| `UserPromptSubmit` | 用户提交 prompt 前 | ✅ |
| `Notification` | Claude 发送通知时 | ❌ |
| `Stop` | Claude 完成响应时 | ❌ |
| `SubagentStop` | 子代理任务完成时 | ❌ |

### 定义方式

Hooks 可以在以下位置定义：
- `.claude/hooks/` 目录
- Rule/Agent 的 frontmatter 中

### 示例：PreToolUse Hook

```yaml
# .claude/hooks/pre-tool-use.yaml
hook: PreToolUse
tool: Edit  # 可选：限制特定工具
condition: |
  {{#if (eq tool.name "Edit")}}
    {{#if (contains file.path "CLAUDE.md")}}
      true
    {{/if}}
  {{/if}}
action: |
  你正在修改 CLAUDE.md，请确认：
  1. 没有添加超过 50 行的详细实现
  2. 代码知识已下沉到 docs/
```

---

## 引用语法总结

### 热引用（自动加载）

```markdown
@docs/claude/claude-code-config.md
@.claude/rules/knowledge-organization.md
```

### 冷引用（按需加载）

```markdown
[文档名称](docs/basic/build-system.md)
```

对话中显式加载：
```
@docs/basic/build-system.md
```

---

## 最佳实践

### 1. Rules
- 保持 `alwaysApply: true` 的规则简洁高效
- 详细规则使用 `alwaysApply: false`，需要时显式触发
- 规则描述要清晰说明触发条件

### 2. Agents
- 为特定任务创建专门的代理
- 使用 `isolation: worktree` 保护主仓库
- 限制工具权限，遵循最小权限原则

### 3. Skills
- 高频使用的技能设置 `eager: true`
- 低频/专项技能使用 `eager: false`
- 提供清晰的 `userInvocable` 设置

### 4. Hooks
- 谨慎使用阻止型 Hooks，避免干扰正常操作
- 条件判断要精确，避免误触发
- 提供清晰的提示信息

---

## 参考

- [Claude Code 官方文档](https://code.claude.com/docs)
