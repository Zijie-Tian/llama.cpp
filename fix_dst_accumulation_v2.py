#!/usr/bin/env python3

def fix_dst_accumulation_v2():
    """基于flash attention数学原理，正确修复dst累积逻辑"""
    
    filepath = 'ggml/src/ggml-cpu/ops.cpp'
    
    # 读取文件
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 更复杂但正确的修复逻辑
    old_pattern = '''        // Write updated S and M values back to state tensor
        state_data[state_idx * 2 + 0] = M; // maximum KQ value (index 0)
        state_data[state_idx * 2 + 1] = S; // sum (index 1)

        if (v->type == GGML_TYPE_F16) {
            for (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // V /= S
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
        
        // Read previous state to check if this is a continuation
        const float initial_M = state_data[state_idx * 2 + 0];
        const float initial_S = state_data[state_idx * 2 + 1];
        
        // Check if this is a continuation call (not the first segment)
        const bool is_continuation = (initial_M != -INFINITY && initial_S > 0.0f);
        
        if (is_continuation) {
            // This is a continuation: merge with previous accumulated result
            
            // The current VKQ32 contains the result of current segment processing
            // We need to merge it with the previous result stored in dst
            
            // Previous result was normalized by previous S, so we need to:
            // 1. Unnormalize the previous result by multiplying by prev_S
            // 2. Apply scaling factor if M changed
            // 3. Add current segment result
            // 4. Normalize by new total S
            
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
            } else {
                // Previous M was >= current M, current VKQ32 needs scaling
                const float scale_factor = expf(curr_M - prev_M);
                for (int64_t d = 0; d < DV; ++d) {
                    // Scale current result and add unnormalized previous result
                    VKQ32[d] = VKQ32[d] * scale_factor + dst_ptr[d] * prev_S;
                }
                // Update M to use previous M since it was higher
                M = prev_M;
            }
        }

        // Convert FP16 to FP32 if needed
        if (v->type == GGML_TYPE_F16) {
            for (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_FP16_TO_FP32(VKQ16[d]);
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
    
    # 替换
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        
        # 写回文件
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ Successfully fixed dst accumulation logic v2!")
        print("Fixed areas:")
        print("  - Proper continuation detection using state values")
        print("  - Correct scaling based on M value changes")  
        print("  - Proper merge of previous and current results")
        print("  - Final normalization with total S")
        return True
    else:
        print("❌ Pattern not found for dst accumulation fix v2")
        return False

if __name__ == "__main__":
    fix_dst_accumulation_v2()