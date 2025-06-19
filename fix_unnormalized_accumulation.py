#!/usr/bin/env python3

def fix_unnormalized_accumulation():
    """修复flash attention with state：dst存储未归一化结果，避免每次都归一化"""
    
    filepath = 'ggml/src/ggml-cpu/ops.cpp'
    
    # 读取文件
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 找到需要修复的dst处理逻辑
    old_pattern = '''        if (is_continuation) {
            // This is a continuation: merge with previous accumulated result
            
            // Current VKQ32 contains result of current segment processing (unnormalized)
            // Previous result in dst was normalized, need to merge properly
            
            const float prev_M = initial_M; // M before current segment
            const float prev_S = initial_S; // S before current segment  
            const float curr_M = M;         // M after current segment
            const float curr_S = S;         // S after current segment
            
            if (curr_M > prev_M) {
                // Current segment had higher max, need to scale previous result
                const float scale_factor = expf(prev_M - curr_M);
                for (int64_t d = 0; d < DV; ++d) {
                    // Unnormalize previous result and apply scaling
                    const float prev_unnormalized = dst_ptr[d] * prev_S * scale_factor;
                    // Add current segment's unnormalized result
                    VKQ32[d] = VKQ32[d] + prev_unnormalized;
                }
            } else if (prev_M > curr_M) {
                // Previous M was higher, current VKQ32 needs scaling
                const float scale_factor = expf(curr_M - prev_M);
                for (int64_t d = 0; d < DV; ++d) {
                    // Scale current result and add unnormalized previous result  
                    VKQ32[d] = VKQ32[d] * scale_factor + dst_ptr[d] * prev_S;
                }
                // Update M to use previous M since it was higher
                M = prev_M;
            } else {
                // M values equal, just add unnormalized results
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ32[d] = VKQ32[d] + dst_ptr[d] * prev_S;
                }
            }
        }

        // Write updated S and M values back to state tensor
        state_data[state_idx * 2 + 0] = M; // maximum KQ value (index 0)
        state_data[state_idx * 2 + 1] = S; // sum (index 1)

        // Final normalization with total S
        const float S_inv = 1.0f / S;
        ggml_vec_scale_f32(DV, VKQ32, S_inv);

        // Write final normalized result to dst
        memcpy(dst_ptr, VKQ32, nb1);'''
    
    new_pattern = '''        if (is_continuation) {
            // This is a continuation: merge with previous accumulated result
            
            // Current VKQ32 contains result of current segment processing (unnormalized)
            // Previous result in dst was also unnormalized, need to merge using flash attention math
            
            const float prev_M = initial_M; // M before current segment
            const float prev_S = initial_S; // S before current segment  
            const float curr_M = M;         // M after current segment
            const float curr_S = S;         // S after current segment
            
            if (curr_M > prev_M) {
                // Current segment had higher max, need to scale previous result
                const float scale_factor = expf(prev_M - curr_M);
                for (int64_t d = 0; d < DV; ++d) {
                    // Scale previous unnormalized result and add current unnormalized result
                    VKQ32[d] = VKQ32[d] + dst_ptr[d] * scale_factor;
                }
            } else if (prev_M > curr_M) {
                // Previous M was higher, current VKQ32 needs scaling
                const float scale_factor = expf(curr_M - prev_M);
                for (int64_t d = 0; d < DV; ++d) {
                    // Scale current result and add previous unnormalized result  
                    VKQ32[d] = VKQ32[d] * scale_factor + dst_ptr[d];
                }
                // Update M to use previous M since it was higher
                M = prev_M;
            } else {
                // M values equal, just add unnormalized results
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ32[d] = VKQ32[d] + dst_ptr[d];
                }
            }
        }

        // Write updated S and M values back to state tensor
        state_data[state_idx * 2 + 0] = M; // maximum KQ value (index 0)
        state_data[state_idx * 2 + 1] = S; // sum (index 1)

        // Store UNNORMALIZED result in dst for potential future accumulation
        // This allows proper flash attention accumulation across segments
        memcpy(dst_ptr, VKQ32, nb1);'''
    
    # 替换
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        
        # 写回文件
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ Successfully fixed unnormalized accumulation!")
        print("Fixed areas:")
        print("  - Store unnormalized results in dst for proper accumulation")
        print("  - Removed incorrect normalization during intermediate steps")
        print("  - Proper flash attention math for segment merging")
        print("  - Only normalize at final output (handled by caller)")
        return True
    else:
        print("❌ Pattern not found for unnormalized accumulation fix")
        return False

if __name__ == "__main__":
    fix_unnormalized_accumulation()