#!/usr/bin/env python3
import re

def fix_memset_bug():
    """修复ggml-cpu/ops.cpp中第7287-7290行的memset bug"""
    
    filepath = 'ggml/src/ggml-cpu/ops.cpp'
    
    # 读取文件
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # 定位问题行（约第7287-7290行）
    for i, line in enumerate(lines):
        if 'if (v->type == GGML_TYPE_F16) {' in line and i > 7280 and i < 7300:
            print(f"Found bug at line {i+1}: {line.strip()}")
            
            # 找到完整的if-else块
            start_line = i
            brace_count = 0
            end_line = i
            
            for j in range(i, len(lines)):
                if '{' in lines[j]:
                    brace_count += lines[j].count('{')
                if '}' in lines[j]:
                    brace_count -= lines[j].count('}')
                    if brace_count == 0:
                        end_line = j
                        break
            
            print(f"Bug block spans lines {start_line+1} to {end_line+1}")
            
            # 提取缩进
            indent = ''
            for char in lines[start_line]:
                if char in ' \t':
                    indent += char
                else:
                    break
            
            # 创建修复后的代码
            fixed_lines = [
                f"{indent}// Only reset accumulator on first call (when M == -INFINITY)\n",
                f"{indent}// Otherwise, preserve previous accumulated values for continuation\n", 
                f"{indent}if (M == -INFINITY) {{\n",
                f"{indent}    if (v->type == GGML_TYPE_F16) {{\n",
                f"{indent}        memset(VKQ16, 0, DV*sizeof(ggml_fp16_t));\n",
                f"{indent}    }} else {{\n",
                f"{indent}        memset(VKQ32, 0, DV*sizeof(float));\n",
                f"{indent}    }}\n",
                f"{indent}}}\n"
            ]
            
            # 替换原来的行
            lines[start_line:end_line+1] = fixed_lines
            
            # 写回文件
            with open(filepath, 'w') as f:
                f.writelines(lines)
            
            print("✅ Successfully fixed the memset bug!")
            print("Fixed code:")
            for line in fixed_lines:
                print(f"  {line.rstrip()}")
            return True
    
    print("❌ Bug location not found")
    return False

if __name__ == "__main__":
    fix_memset_bug()