#!/usr/bin/env python3

def fix_dst_accumulation():
    """修复flash attention with state中的dst累积逻辑"""
    
    filepath = 'ggml/src/ggml-cpu/ops.cpp'
    
    # 读取文件
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 找到需要修复的dst写入逻辑
    old_pattern = '''        // V /= S
        const float S_inv = 1.0f / S;
        ggml_vec_scale_f32(DV, VKQ32, S_inv);

        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // original
        // memcpy((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3), V, nev0*sizeof(float));

        // permute(0, 2, 1, 3)
        memcpy((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);'''
    
    new_pattern = '''        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;
        
        // Calculate dst offset
        const size_t dst_offset = (i3*ne2*ne1 + i2 + i1*ne1)*nb1;
        float * dst_ptr = (float *)((char *)dst->data + dst_offset);

        // For segmented processing: accumulate results properly
        if (M == state_data[state_idx * 2 + 0] && S > state_data[state_idx * 2 + 1]) {
            // This is a continuation call - we need to merge with existing dst data
            
            // Read previous state values  
            const float prev_M = state_data[state_idx * 2 + 0];
            const float prev_S = state_data[state_idx * 2 + 1];
            
            if (prev_S > 0.0f) {
                // Reconstruct the previous unnormalized accumulated result
                // prev_dst was normalized by prev_S, so multiply back to get unnormalized
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ32[d] = VKQ32[d] + dst_ptr[d] * prev_S;
                }
            }
        }
        
        // Final normalization with current total S
        const float S_inv = 1.0f / S;
        ggml_vec_scale_f32(DV, VKQ32, S_inv);

        // Write final result to dst
        memcpy(dst_ptr, VKQ32, nb1);'''
    
    # 替换
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        
        # 写回文件
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ Successfully fixed dst accumulation logic!")
        print("Fixed areas:")
        print("  - Added dst read/accumulate logic for continuation calls")
        print("  - Proper merging of previous and current results")
        print("  - Maintained final normalization")
        return True
    else:
        print("❌ Pattern not found for dst accumulation fix")
        return False

if __name__ == "__main__":
    fix_dst_accumulation()