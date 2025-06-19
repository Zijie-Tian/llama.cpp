#!/usr/bin/env python3

def fix_state_timing():
    """修复状态读取时机：在segment处理前保存状态，处理后合并"""
    
    filepath = 'ggml/src/ggml-cpu/ops.cpp'
    
    # 读取文件
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 找到需要修复的完整代码块 - 从状态读取到写入
    old_pattern = '''        // Read initial S and M values from state tensor 
        // State format: [M, S] for each head/position
        float S = state_data[state_idx * 2 + 1];     // sum (index 1)
        float M = state_data[state_idx * 2 + 0];     // maximum KQ value (index 0)

        // If this is the first call (indicated by M == -INFINITY), initialize properly
        if (M == -INFINITY) {
            S = 0.0f;
        }'''
    
    new_pattern = '''        // Read initial S and M values from state tensor 
        // State format: [M, S] for each head/position
        float initial_S = state_data[state_idx * 2 + 1];     // sum (index 1)
        float initial_M = state_data[state_idx * 2 + 0];     // maximum KQ value (index 0)
        
        float S = initial_S;
        float M = initial_M;

        // If this is the first call (indicated by M == -INFINITY), initialize properly
        if (M == -INFINITY) {
            S = 0.0f;
        }'''
    
    # 替换第一部分
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        
        # 现在修复dst处理逻辑 - 移除重复的状态读取
        old_dst_pattern = '''        // Read state BEFORE current segment processing
        const float initial_M = state_data[state_idx * 2 + 0];
        const float initial_S = state_data[state_idx * 2 + 1];
        
        // Check if this is a continuation call (not the first segment)
        const bool is_continuation = (initial_M != -INFINITY && initial_S > 0.0f);'''
        
        new_dst_pattern = '''        // Check if this is a continuation call (not the first segment)
        // Use the initial state values read at the beginning
        const bool is_continuation = (initial_M != -INFINITY && initial_S > 0.0f);'''
        
        content = content.replace(old_dst_pattern, new_dst_pattern)
        
        # 写回文件
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ Successfully fixed state timing!")
        print("Fixed areas:")
        print("  - Read initial state at function start")  
        print("  - Use saved initial state for continuation detection")
        print("  - Proper timing for state-based merging")
        return True
    else:
        print("❌ Pattern not found for state timing fix")
        return False

if __name__ == "__main__":
    fix_state_timing()