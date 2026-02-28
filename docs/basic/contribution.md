# llama.cpp 贡献指南

> 本文档为 Claude Code 提供项目特定的贡献规范。
>
> 引用: `@file:docs/basic/contribution.md`

---

## 重要声明

### AI 使用政策 (必须遵守)

> ⚠️ **本项目不接受完全或主要由 AI 生成的 PR**

- AI 只能作为辅助工具，**大多数代码必须由人工编写**
- 使用 AI 必须**明确披露**
- AI 生成的代码即使经过大量编辑仍被视为 AI 生成

**允许**:
- 询问代码库结构
- 学习特定技术
- 人类编写代码后，AI 帮助格式化或小修改
- 生成重复模式的短代码片段

**禁止**:
- AI 编写完整的 PR
- AI 生成大型代码块
- AI 绕过人类贡献者的理解或责任

详见 [AGENTS.md](../../AGENTS.md) 和 [CONTRIBUTING.md](../../CONTRIBUTING.md)

---

## 代码风格

### 基本风格

- **缩进**: 4 个空格
- **括号**: 同一行
- **命名**: `snake_case`
- **指针/引用**: `void * ptr`, `int & a` (空格在前)
- **格式化**: 使用 `clang-format` (v15+)

### 命名规范

```cpp
// ✅ 最长公共前缀优先
int number_small;
int number_big;

// ❌ 避免
int small_number;
int big_number;

// ✅ 枚举大写 + 前缀
enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NONE = 0,
    LLAMA_VOCAB_TYPE_SPM  = 1,
    LLAMA_VOCAB_TYPE_BPE  = 2,
};

// ✅ 类_动作_名词 模式
llama_model_init();
llama_sampler_chain_remove();
llama_set_embeddings();
```

### 类型规范

```cpp
// ✅ 公共 API 使用定长类型
int32_t count;
size_t  buffer_size;

// ✅ struct 声明
struct foo {};  // 不是 typedef struct foo {} foo;

// ✅ C++ 中省略可选关键字
llama_context * ctx;           // ✅
const llama_rope_type rope_type;  // ✅

// ❌ 避免
struct llama_context * ctx;              // 不需要 struct
const enum llama_rope_type rope_type;    // 不需要 enum
```

---

## PR 准备

### 测试要求

1. **本地运行完整 CI**:
   ```bash
   mkdir -p tmp/results tmp/mnt
   bash ./ci/run.sh ./tmp/results ./tmp/mnt
   ```

2. **GGML 修改** → 运行 `test-backend-ops`
   ```bash
   ./build/bin/test-backend-ops
   ```

3. **性能敏感变更** → 运行基准测试
   ```bash
   ./build/bin/llama-bench -m model.gguf
   ./build/bin/llama-perplexity -m model.gguf -f test.txt
   ```

4. **格式化检查**:
   ```bash
   # 使用 clang-format
   clang-format -i src/your-file.cpp
   ```

### PR 规范

- **单一功能**: 每个 PR 一个功能或修复
- **CPU 优先**: 新功能先实现 CPU 支持，GPU 后续 PR 添加
- **说明清晰**: 说明修改的原因和影响
- **允许维护者编辑**: 方便快速修改

### Commit 格式

维护者使用 squash-merge，格式：

```
<module> : <commit title> (#<issue_number>)

示例:
llama : fix kv cache memory leak (#1234)
ggml  : optimize q4_0 dequantization (#1235)
```

---

## 代码维护

### CODEOWNERS

修改代码后考虑添加到 [CODEOWNERS](../../CODEOWNERS)：
- 表示愿意修复相关 bug
- 审核相关 PR
- 提供开发指导

### 模块选择

参考 [Wiki - Modules](https://github.com/ggml-org/llama.cpp/wiki/Modules) 选择 commit 的模块前缀。

---

## 依赖政策

### 核心原则

**避免添加第三方依赖**

- 优先使用 `vendor/` 中的单头文件库
- 如需新功能，先考虑自己实现
- 跨平台兼容性必须考虑

### 现有依赖

`vendor/` 中的单头库：
- `cpp-httplib` - HTTP 服务器
- `json.hpp` - JSON 处理
- `stb_image.h` - 图像解码
- `miniaudio.h` - 音频解码

---

## 常见问题

### 构建失败

```bash
# 清理重建
rm -rf build
cmake -B build
cmake --build build --config Release
```

### 测试失败

确保模型文件存在：
```bash
# 测试会自动下载，但可能需要手动触发
./build/bin/test-thread-safety -m models/tinyllamas/stories15M-q4_0.gguf ...
```

### 性能回归

- 使用 `llama-bench` 对比修改前后
- 检查是否意外禁用了优化
- 验证量化格式是否正确
