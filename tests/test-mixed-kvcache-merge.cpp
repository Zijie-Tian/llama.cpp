#include "ggml.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

// 填充tensor，每个头使用不同的值
static void fill_tensor_with_head_values(ggml_tensor * tensor) {
    const int64_t head_dim = tensor->ne[0];
    const int64_t seq_len = tensor->ne[1]; 
    const int64_t n_heads = tensor->ne[2];
    const int64_t batch = tensor->ne[3];
    
    printf("Filling tensor: [%ld, %ld, %ld, %ld]\n", head_dim, seq_len, n_heads, batch);
    
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) tensor->data;
        
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t h = 0; h < n_heads; h++) {
                for (int64_t s = 0; s < seq_len; s++) {
                    for (int64_t d = 0; d < head_dim; d++) {
                        int64_t idx = d + s * head_dim + h * head_dim * seq_len + b * head_dim * seq_len * n_heads;
                        // 每个头使用不同的值：head_0=1.0, head_1=2.0, head_2=3.0, etc.
                        data[idx] = ggml_fp32_to_fp16((float)(h + 1));
                    }
                }
            }
        }
    } else if (tensor->type == GGML_TYPE_F32) {
        float * data = (float *) tensor->data;
        
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t h = 0; h < n_heads; h++) {
                for (int64_t s = 0; s < seq_len; s++) {
                    for (int64_t d = 0; d < head_dim; d++) {
                        int64_t idx = d + s * head_dim + h * head_dim * seq_len + b * head_dim * seq_len * n_heads;
                        // 每个头使用不同的值：head_0=1.0, head_1=2.0, head_2=3.0, etc.
                        data[idx] = (float)(h + 1);
                    }
                }
            }
        }
    } else {
        // 处理量化的tensor类型
        ggml_from_float_t from_float = ggml_get_type_traits_cpu(tensor->type)->from_float;
        if (from_float) {
            // 创建临时浮点缓冲区
            const int64_t total_elements = head_dim * seq_len * n_heads * batch;
            std::vector<float> temp_buffer(total_elements);
            
            // 填充临时缓冲区
            for (int64_t b = 0; b < batch; b++) {
                for (int64_t h = 0; h < n_heads; h++) {
                    for (int64_t s = 0; s < seq_len; s++) {
                        for (int64_t d = 0; d < head_dim; d++) {
                            int64_t idx = d + s * head_dim + h * head_dim * seq_len + b * head_dim * seq_len * n_heads;
                            temp_buffer[idx] = (float)(h + 1);
                        }
                    }
                }
            }
            
            // 将浮点值转换为量化值
            from_float(temp_buffer.data(), tensor->data, total_elements);
        } else {
            printf("ERROR: Unsupported tensor type for filling\n");
            return;
        }
    }
}

// 打印tensor的一些样本值用于验证
static void print_tensor_samples(const char * name, ggml_tensor * tensor, int max_samples = 20) {
    const int64_t head_dim = tensor->ne[0];
    const int64_t seq_len = tensor->ne[1]; 
    const int64_t n_heads = tensor->ne[2];
    const int64_t batch = tensor->ne[3];
    
    printf("%s [%ld,%ld,%ld,%ld] head values:\n", name, head_dim, seq_len, n_heads, batch);
    
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) tensor->data;
        
        // Print first few values from each head
        for (int64_t h = 0; h < n_heads && h < 4; h++) {
            printf("  Head %ld: ", h);
            for (int64_t s = 0; s < std::min((int64_t)4, seq_len); s++) {
                for (int64_t d = 0; d < std::min((int64_t)4, head_dim); d++) {
                    int64_t idx = d + s * head_dim + h * head_dim * seq_len;
                    printf("%.2f ", ggml_fp16_to_fp32(data[idx]));
                }
            }
            printf("...\n");
        }
    } else if (tensor->type == GGML_TYPE_F32) {
        float * data = (float *) tensor->data;
        
        // Print first few values from each head
        for (int64_t h = 0; h < n_heads && h < 4; h++) {
            printf("  Head %ld: ", h);
            for (int64_t s = 0; s < std::min((int64_t)4, seq_len); s++) {
                for (int64_t d = 0; d < std::min((int64_t)4, head_dim); d++) {
                    int64_t idx = d + s * head_dim + h * head_dim * seq_len;
                    printf("%.2f ", data[idx]);
                }
            }
            printf("...\n");
        }
    } else {
        // 处理量化的tensor类型
        ggml_to_float_t to_float = ggml_get_type_traits(tensor->type)->to_float;
        if (to_float) {
            // 创建临时缓冲区来存储转换后的浮点值
            float* temp_buffer = new float[head_dim * seq_len * n_heads * batch];
            to_float(tensor->data, temp_buffer, head_dim * seq_len * n_heads * batch);
            
            // 打印转换后的值
            for (int64_t h = 0; h < n_heads && h < 4; h++) {
                printf("  Head %ld: ", h);
                for (int64_t s = 0; s < std::min((int64_t)4, seq_len); s++) {
                    for (int64_t d = 0; d < std::min((int64_t)4, head_dim); d++) {
                        int64_t idx = d + s * head_dim + h * head_dim * seq_len;
                        printf("%.2f ", temp_buffer[idx]);
                    }
                }
                printf("...\n");
            }
            
            delete[] temp_buffer;
        } else {
            printf("ERROR: Unsupported tensor type for printing\n");
            return;
        }
    }
}

// 合并两个KV tensor分片的函数 - 使用1D视图简化操作
static struct ggml_cgraph * build_merge_kv_graph(
    ggml_context * ctx,
    ggml_tensor * dst,
    ggml_tensor * fp16_shard,
    ggml_tensor * quant_shard
) {
    printf("Building merge graph...\n");
    printf("  dst: [%ld, %ld, %ld, %ld]\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    printf("  fp16_shard: [%ld, %ld, %ld, %ld]\n", fp16_shard->ne[0], fp16_shard->ne[1], fp16_shard->ne[2], fp16_shard->ne[3]);
    printf("  quant_shard: [%ld, %ld, %ld, %ld]\n", quant_shard->ne[0], quant_shard->ne[1], quant_shard->ne[2], quant_shard->ne[3]);
    
    GGML_ASSERT(fp16_shard->ne[0] == quant_shard->ne[0]); // head_dim should match
    GGML_ASSERT(fp16_shard->ne[2] == quant_shard->ne[2]); // n_heads should match  
    GGML_ASSERT(fp16_shard->ne[3] == quant_shard->ne[3]); // batch should match
    GGML_ASSERT(dst->ne[1] == fp16_shard->ne[1] + quant_shard->ne[1]); // seq_len should be sum

    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // Calculate elements per head
    size_t fp16_elements_per_head = fp16_shard->ne[0] * fp16_shard->ne[1];  // head_dim * fp16_seq_len
    size_t quant_elements_per_head = quant_shard->ne[0] * quant_shard->ne[1]; // head_dim * quant_seq_len
    
    for (int i = 0; i < (int)dst->ne[3]; i++) {        // batch
        for (int j = 0; j < (int)dst->ne[2]; j++) {    // n_heads
            
            // Calculate offsets for this head
            size_t dst_head_offset = (i * dst->ne[2] + j) * dst->nb[2];
            size_t fp16_src_offset = (i * fp16_shard->ne[2] + j) * fp16_shard -> nb[2];
            size_t quant_src_offset = (i * quant_shard->ne[2] + j) * quant_shard -> nb[2];

            // Create 1D views for FP16 part
            ggml_tensor * dst_fp16_part = ggml_view_1d(
                ctx, dst,
                fp16_elements_per_head,
                dst_head_offset
            );
            
            ggml_tensor * fp16_src_head = ggml_view_1d(
                ctx, fp16_shard,
                fp16_elements_per_head,
                fp16_src_offset
            );

            // Create 1D views for quantized part 
            ggml_tensor * dst_quant_part = ggml_view_1d(
                ctx, dst,
                quant_elements_per_head,
                dst_head_offset + fp16_elements_per_head * dst->nb[0]
            );
            
            ggml_tensor * quant_src_head = ggml_view_1d(
                ctx, quant_shard,
                quant_elements_per_head,
                quant_src_offset
            );

            // 复制操作
            ggml_tensor * copy_fp16 = ggml_cpy(ctx, fp16_src_head, dst_fp16_part);
            ggml_tensor * copy_quant = ggml_cpy(ctx, quant_src_head, dst_quant_part);

            ggml_build_forward_expand(gf, copy_fp16);
            ggml_build_forward_expand(gf, copy_quant);
        }
    }

    return gf;
}

// 验证合并结果的函数
static bool verify_merge_result(ggml_tensor * merged, ggml_tensor * fp16_shard, ggml_tensor * quant_shard) {
    if (merged->type != GGML_TYPE_F16 && merged->type != GGML_TYPE_F32) {
        printf("ERROR: Only F16 or F32 tensors supported for merged tensor\n");
        return false;
    }
    
    if (fp16_shard->type != GGML_TYPE_F16 && fp16_shard->type != GGML_TYPE_F32) {
        printf("ERROR: Only F16 or F32 tensors supported for fp16_shard\n");
        return false;
    }
    
    const int64_t head_dim = merged->ne[0];
    const int64_t total_seq_len = merged->ne[1];
    const int64_t n_heads = merged->ne[2];
    const int64_t batch = merged->ne[3];
    
    const int64_t fp16_seq_len = fp16_shard->ne[1];
    const int64_t quant_seq_len = quant_shard->ne[1];
    
    printf("Verifying merge result...\n");
    printf("  total_seq_len=%ld, fp16_seq_len=%ld, quant_seq_len=%ld\n", 
           total_seq_len, fp16_seq_len, quant_seq_len);
    
    bool success = true;
    int errors = 0;
    
    // 创建临时浮点缓冲区用于量化tensor
    std::vector<float> quant_buffer;
    if (quant_shard->type != GGML_TYPE_F16 && quant_shard->type != GGML_TYPE_F32) {
        ggml_to_float_t to_float = ggml_get_type_traits(quant_shard->type)->to_float;
        if (!to_float) {
            printf("ERROR: Unsupported quantized tensor type for verification\n");
            return false;
        }
        
        const int64_t total_elements = head_dim * quant_seq_len * n_heads * batch;
        quant_buffer.resize(total_elements);
        to_float(quant_shard->data, quant_buffer.data(), total_elements);
    }
    
    for (int64_t b = 0; b < batch && success; b++) {
        for (int64_t h = 0; h < n_heads && success; h++) {
            for (int64_t s = 0; s < total_seq_len; s++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    int64_t merged_idx = d + s * head_dim + h * head_dim * total_seq_len + b * head_dim * total_seq_len * n_heads;
                    float merged_val;
                    
                    if (merged->type == GGML_TYPE_F16) {
                        merged_val = ggml_fp16_to_fp32(((ggml_fp16_t*)merged->data)[merged_idx]);
                    } else {
                        merged_val = ((float*)merged->data)[merged_idx];
                    }
                    
                    float expected_val;
                    if (s < fp16_seq_len) {
                        // 应该来自fp16_shard
                        int64_t src_idx = d + s * head_dim + h * head_dim * fp16_seq_len + b * head_dim * fp16_seq_len * n_heads;
                        if (fp16_shard->type == GGML_TYPE_F16) {
                            expected_val = ggml_fp16_to_fp32(((ggml_fp16_t*)fp16_shard->data)[src_idx]);
                        } else {
                            expected_val = ((float*)fp16_shard->data)[src_idx];
                        }
                    } else {
                        // 应该来自quant_shard
                        int64_t quant_s = s - fp16_seq_len;
                        int64_t src_idx = d + quant_s * head_dim + h * head_dim * quant_seq_len + b * head_dim * quant_seq_len * n_heads;
                        
                        if (quant_shard->type == GGML_TYPE_F16) {
                            expected_val = ggml_fp16_to_fp32(((ggml_fp16_t*)quant_shard->data)[src_idx]);
                        } else if (quant_shard->type == GGML_TYPE_F32) {
                            expected_val = ((float*)quant_shard->data)[src_idx];
                        } else {
                            // 使用预先转换的浮点缓冲区
                            expected_val = quant_buffer[src_idx];
                        }
                    }
                    
                    if (fabs(merged_val - expected_val) > 1e-5) {
                        if (errors < 10) {  // 只打印前10个错误
                            printf("ERROR at [%ld,%ld,%ld,%ld]: got %.6f, expected %.6f\n", 
                                   d, s, h, b, merged_val, expected_val);
                        }
                        errors++;
                        if (errors >= 100) {  // 如果错误太多就停止检查
                            success = false;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    printf("Verification result: %s (%d errors found)\n", success ? "PASSED" : "FAILED", errors);
    return success;
}

// 合并两个mask tensor分片的函数 - mask的shape是[kv_len, q_len, 1, batch]
static struct ggml_cgraph * build_merge_mask_graph(
    ggml_context * ctx,
    ggml_tensor * dst,
    ggml_tensor * fp16_shard,
    ggml_tensor * quant_shard
) {
    printf("Building mask merge graph...\n");
    printf("  dst: [%ld, %ld, %ld, %ld]\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    printf("  fp16_shard: [%ld, %ld, %ld, %ld]\n", fp16_shard->ne[0], fp16_shard->ne[1], fp16_shard->ne[2], fp16_shard->ne[3]);
    printf("  quant_shard: [%ld, %ld, %ld, %ld]\n", quant_shard->ne[0], quant_shard->ne[1], quant_shard->ne[2], quant_shard->ne[3]);
    
    GGML_ASSERT(fp16_shard->ne[1] == quant_shard->ne[1]); // q_len should match
    GGML_ASSERT(fp16_shard->ne[2] == quant_shard->ne[2]); // should both be 1
    GGML_ASSERT(fp16_shard->ne[3] == quant_shard->ne[3]); // batch should match
    GGML_ASSERT(dst->ne[0] == fp16_shard->ne[0] + quant_shard->ne[0]); // kv_len should be sum
    GGML_ASSERT(dst->ne[1] == fp16_shard->ne[1]); // q_len should match
    
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    
    const int64_t fp16_kv_len = fp16_shard->ne[0];
    const int64_t quant_kv_len = quant_shard->ne[0];
    const int64_t q_len = dst->ne[1];
    const int64_t batch = dst->ne[3];
    
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t q = 0; q < q_len; q++) {
            // 拷贝 FP16 部分: dst[0:fp16_kv_len, q, 0, b] = fp16_shard[:, q, 0, b]
            ggml_tensor * dst_fp16_slice = ggml_view_2d(
                ctx, dst,
                fp16_kv_len, 1,  // [fp16_kv_len, 1]
                dst->nb[0], 
                b * dst->nb[3] + q * dst->nb[1]  // offset to [0, q, 0, b]
            );
            
            ggml_tensor * fp16_src_slice = ggml_view_2d(
                ctx, fp16_shard, 
                fp16_kv_len, 1,  // [fp16_kv_len, 1]
                fp16_shard->nb[0],
                b * fp16_shard->nb[3] + q * fp16_shard->nb[1]  // offset to [0, q, 0, b]
            );
            
            // 拷贝 quantized 部分: dst[fp16_kv_len:total_kv_len, q, 0, b] = quant_shard[:, q, 0, b]
            ggml_tensor * dst_quant_slice = ggml_view_2d(
                ctx, dst,
                quant_kv_len, 1,  // [quant_kv_len, 1]
                dst->nb[0],
                b * dst->nb[3] + q * dst->nb[1] + fp16_kv_len * dst->nb[0]  // offset to [fp16_kv_len, q, 0, b]
            );
            
            ggml_tensor * quant_src_slice = ggml_view_2d(
                ctx, quant_shard,
                quant_kv_len, 1,  // [quant_kv_len, 1]
                quant_shard->nb[0],
                b * quant_shard->nb[3] + q * quant_shard->nb[1]  // offset to [0, q, 0, b]
            );
            
            // Copy operations
            ggml_tensor * copy_fp16 = ggml_cpy(ctx, fp16_src_slice, dst_fp16_slice);
            ggml_tensor * copy_quant = ggml_cpy(ctx, quant_src_slice, dst_quant_slice);
            
            ggml_build_forward_expand(gf, copy_fp16);
            ggml_build_forward_expand(gf, copy_quant);
        }
    }
    
    return gf;
}

// 填充mask tensor，每个batch使用不同的模式
static void fill_mask_tensor(ggml_tensor * mask) {
    const int64_t kv_len = mask->ne[0];
    const int64_t q_len = mask->ne[1]; 
    const int64_t batch = mask->ne[3];
    
    printf("Filling mask tensor: [%ld, %ld, %ld, %ld]\n", kv_len, q_len, 1L, batch);
    
    if (mask->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) mask->data;
        
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t q = 0; q < q_len; q++) {
                for (int64_t kv = 0; kv < kv_len; kv++) {
                    int64_t idx = kv + q * kv_len + b * kv_len * q_len;
                    // 创建因果掩码：只允许注意到之前和当前的token
                    // 同时为不同batch添加偏移以便区分
                    if (kv <= q + b) {
                        data[idx] = ggml_fp32_to_fp16(0.0f);  // 可以注意到
                    } else {
                        data[idx] = ggml_fp32_to_fp16(-INFINITY);  // 不能注意到
                    }
                }
            }
        }
    } else if (mask->type == GGML_TYPE_F32) {
        float * data = (float *) mask->data;
        
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t q = 0; q < q_len; q++) {
                for (int64_t kv = 0; kv < kv_len; kv++) {
                    int64_t idx = kv + q * kv_len + b * kv_len * q_len;
                    // 创建因果掩码
                    if (kv <= q + b) {
                        data[idx] = 0.0f;  // 可以注意到
                    } else {
                        data[idx] = -INFINITY;  // 不能注意到
                    }
                }
            }
        }
    } else {
        // 处理量化的mask类型
        ggml_from_float_t from_float = ggml_get_type_traits_cpu(mask->type)->from_float;
        if (from_float) {
            // 创建临时浮点缓冲区
            const int64_t total_elements = kv_len * q_len * batch;
            std::vector<float> temp_buffer(total_elements);
            
            // 填充临时缓冲区
            for (int64_t b = 0; b < batch; b++) {
                for (int64_t q = 0; q < q_len; q++) {
                    for (int64_t kv = 0; kv < kv_len; kv++) {
                        int64_t idx = kv + q * kv_len + b * kv_len * q_len;
                        // 创建因果掩码
                        if (kv <= q + b) {
                            temp_buffer[idx] = 0.0f;  // 可以注意到
                        } else {
                            temp_buffer[idx] = -INFINITY;  // 不能注意到
                        }
                    }
                }
            }
            
            // 将浮点值转换为量化值
            from_float(temp_buffer.data(), mask->data, total_elements);
        } else {
            printf("ERROR: Unsupported mask tensor type for filling\n");
            return;
        }
    }
}

// 打印mask tensor的全部值用于验证
static void print_mask_samples(const char * name, ggml_tensor * mask, int max_samples = 20) {
    const int64_t kv_len = mask->ne[0];
    const int64_t q_len = mask->ne[1]; 
    const int64_t batch = mask->ne[3];
    
    printf("%s [%ld,%ld,%ld,%ld] mask values:\n", name, kv_len, q_len, 1L, batch);
    
    if (mask->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) mask->data;
        
        // Print all values from all batches
        for (int64_t b = 0; b < batch; b++) {
            printf("  Batch %ld:\n", b);
            for (int64_t q = 0; q < q_len; q++) {
                printf("    Q%-2ld: ", q);
                for (int64_t kv = 0; kv < kv_len; kv++) {
                    int64_t idx = kv + q * kv_len + b * kv_len * q_len;
                    float val = ggml_fp16_to_fp32(data[idx]);
                    printf("%-2s ", (val == 0.0f) ? "0" : (val == -INFINITY) ? "-∞" : "?");
                }
                printf("\n");
            }
        }
    } else if (mask->type == GGML_TYPE_F32) {
        float * data = (float *) mask->data;
        
        // Print all values from all batches
        for (int64_t b = 0; b < batch; b++) {
            printf("  Batch %ld:\n", b);
            for (int64_t q = 0; q < q_len; q++) {
                printf("    Q%-2ld: ", q);
                for (int64_t kv = 0; kv < kv_len; kv++) {
                    int64_t idx = kv + q * kv_len + b * kv_len * q_len;
                    float val = data[idx];
                    printf("%-2s ", (val == 0.0f) ? "0" : (val == -INFINITY) ? "-∞" : "?");
                }
                printf("\n");
            }
        }
    } else {
        // 处理量化的mask类型
        ggml_to_float_t to_float = ggml_get_type_traits(mask->type)->to_float;
        if (to_float) {
            // 创建临时缓冲区来存储转换后的浮点值
            float* temp_buffer = new float[kv_len * q_len * batch];
            to_float(mask->data, temp_buffer, kv_len * q_len * batch);
            
            // 打印转换后的值
            for (int64_t b = 0; b < batch; b++) {
                printf("  Batch %ld:\n", b);
                for (int64_t q = 0; q < q_len; q++) {
                    printf("    Q%-2ld: ", q);
                    for (int64_t kv = 0; kv < kv_len; kv++) {
                        int64_t idx = kv + q * kv_len + b * kv_len * q_len;
                        float val = temp_buffer[idx];
                        printf("%-2s ", (val == 0.0f) ? "0" : (val == -INFINITY) ? "-∞" : "?");
                    }
                    printf("\n");
                }
            }
            
            delete[] temp_buffer;
        } else {
            printf("ERROR: Unsupported mask tensor type for printing\n");
            return;
        }
    }
}

// 验证mask合并结果的函数
static bool verify_mask_merge_result(ggml_tensor * merged, ggml_tensor * fp16_shard, ggml_tensor * quant_shard) {
    if (merged->type != GGML_TYPE_F16 && merged->type != GGML_TYPE_F32) {
        printf("ERROR: Only F16 or F32 tensors supported for merged mask\n");
        return false;
    }
    
    if (fp16_shard->type != GGML_TYPE_F16 && fp16_shard->type != GGML_TYPE_F32) {
        printf("ERROR: Only F16 or F32 tensors supported for fp16_shard mask\n");
        return false;
    }
    
    const int64_t fp16_kv_len = fp16_shard->ne[0];
    const int64_t quant_kv_len = quant_shard->ne[0];
    const int64_t total_kv_len = merged->ne[0];
    const int64_t q_len = merged->ne[1];
    const int64_t batch = merged->ne[3];
    
    if (total_kv_len != fp16_kv_len + quant_kv_len) {
        printf("ERROR: Merged mask kv_len (%ld) != fp16_kv_len (%ld) + quant_kv_len (%ld)\n", 
               total_kv_len, fp16_kv_len, quant_kv_len);
        return false;
    }
    
    bool success = true;
    int errors = 0;
    
    // 创建临时浮点缓冲区用于量化mask
    std::vector<float> quant_buffer;
    if (quant_shard->type != GGML_TYPE_F16 && quant_shard->type != GGML_TYPE_F32) {
        ggml_to_float_t to_float = ggml_get_type_traits(quant_shard->type)->to_float;
        if (!to_float) {
            printf("ERROR: Unsupported quantized mask tensor type for verification\n");
            return false;
        }
        
        const int64_t total_elements = quant_kv_len * q_len * batch;
        quant_buffer.resize(total_elements);
        to_float(quant_shard->data, quant_buffer.data(), total_elements);
    }
    
    for (int64_t b = 0; b < batch && success; b++) {
        for (int64_t q = 0; q < q_len && success; q++) {
            for (int64_t kv = 0; kv < total_kv_len; kv++) {
                int64_t merged_idx = kv + q * total_kv_len + b * total_kv_len * q_len;
                float merged_val;
                
                if (merged->type == GGML_TYPE_F16) {
                    merged_val = ggml_fp16_to_fp32(((ggml_fp16_t*)merged->data)[merged_idx]);
                } else {
                    merged_val = ((float*)merged->data)[merged_idx];
                }
                
                float expected_val;
                if (kv < fp16_kv_len) {
                    // 应该来自fp16_shard
                    int64_t src_idx = kv + q * fp16_kv_len + b * fp16_kv_len * q_len;
                    if (fp16_shard->type == GGML_TYPE_F16) {
                        expected_val = ggml_fp16_to_fp32(((ggml_fp16_t*)fp16_shard->data)[src_idx]);
                    } else {
                        expected_val = ((float*)fp16_shard->data)[src_idx];
                    }
                } else {
                    // 应该来自quant_shard
                    int64_t quant_kv = kv - fp16_kv_len;
                    int64_t src_idx = quant_kv + q * quant_kv_len + b * quant_kv_len * q_len;
                    
                    if (quant_shard->type == GGML_TYPE_F16) {
                        expected_val = ggml_fp16_to_fp32(((ggml_fp16_t*)quant_shard->data)[src_idx]);
                    } else if (quant_shard->type == GGML_TYPE_F32) {
                        expected_val = ((float*)quant_shard->data)[src_idx];
                    } else {
                        // 使用预先转换的浮点缓冲区
                        expected_val = quant_buffer[src_idx];
                    }
                }
                
                // 检查是否匹配（考虑浮点精度）
                float diff = fabsf(merged_val - expected_val);
                bool matches = (diff < 1e-5) || 
                              (merged_val == expected_val) ||
                              (std::isinf(merged_val) && std::isinf(expected_val) && 
                               ((merged_val < 0) == (expected_val < 0)));
                
                if (!matches) {
                    if (errors < 10) {  // 只打印前10个错误
                        printf("Mismatch at [b=%ld, q=%ld, kv=%ld]: merged=%.6f, expected=%.6f, diff=%.6f\n",
                               b, q, kv, merged_val, expected_val, diff);
                    }
                    errors++;
                    if (errors >= 100) {  // 如果错误太多就停止
                        success = false;
                        break;
                    }
                }
            }
        }
    }
    
    if (errors > 0) {
        printf("Mask verification completed with %d errors found\n", errors);
    } else {
        printf("Mask verification PASSED: All values match correctly\n");
    }
    
    return errors == 0;
}

int main() {
    printf("=== Mixed KV Cache Merge Test ===\n");

    // 测试参数
    const int head_dim = 32;
    const int fp16_seq_len = 16;   // fp16部分的序列长度
    const int quant_seq_len = 24;  // 量化部分的序列长度
    const int total_seq_len = fp16_seq_len + quant_seq_len;
    const int n_heads = 4;
    const int batch = 1;

    printf("Test parameters:\n");
    printf("  head_dim=%d, n_heads=%d, batch=%d\n", head_dim, n_heads, batch);
    printf("  fp16_seq_len=%d, quant_seq_len=%d, total_seq_len=%d\n", 
           fp16_seq_len, quant_seq_len, total_seq_len);

    // 初始化ggml context
    const size_t ctx_size = 64 * 1024 * 1024;  // 64MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize ggml context\n");
        return 1;
    }

    // 创建测试tensors
    printf("\n--- Creating test tensors ---\n");
    
    // 创建fp16分片 (较新的token)
    ggml_tensor * k_fp16_shard = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, fp16_seq_len, n_heads, batch);
    ggml_tensor * v_fp16_shard = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, fp16_seq_len, n_heads, batch);
    
    // 创建量化分片 (较旧的token)
    ggml_tensor * k_quant_shard = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, head_dim, quant_seq_len, n_heads, batch);
    ggml_tensor * v_quant_shard = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, head_dim, quant_seq_len, n_heads, batch);
    
    // 创建合并后的tensors
    ggml_tensor * k_merged = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, total_seq_len, n_heads, batch);
    ggml_tensor * v_merged = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, total_seq_len, n_heads, batch);

    printf("Created tensors:\n");
    printf("  k_fp16_shard: [%ld, %ld, %ld, %ld]\n", k_fp16_shard->ne[0], k_fp16_shard->ne[1], k_fp16_shard->ne[2], k_fp16_shard->ne[3]);
    printf("  k_quant_shard: [%ld, %ld, %ld, %ld]\n", k_quant_shard->ne[0], k_quant_shard->ne[1], k_quant_shard->ne[2], k_quant_shard->ne[3]);
    printf("  k_merged: [%ld, %ld, %ld, %ld]\n", k_merged->ne[0], k_merged->ne[1], k_merged->ne[2], k_merged->ne[3]);

    // 填充测试数据
    printf("\n--- Filling test data ---\n");
    fill_tensor_with_head_values(k_fp16_shard);
    fill_tensor_with_head_values(v_fp16_shard);  
    fill_tensor_with_head_values(k_quant_shard);
    fill_tensor_with_head_values(v_quant_shard);

    printf("Filled tensors with head-specific values\n");
    print_tensor_samples("k_fp16_shard", k_fp16_shard, 16);
    print_tensor_samples("k_quant_shard", k_quant_shard, 16);

    // 测试K tensor合并
    printf("\n--- Testing K tensor merge ---\n");
    struct ggml_cgraph * k_merge_graph = build_merge_kv_graph(ctx, k_merged, k_fp16_shard, k_quant_shard);
    
    printf("Computing K merge graph...\n");
    enum ggml_status k_status = ggml_graph_compute_with_ctx(ctx, k_merge_graph, 4);
    if (k_status != GGML_STATUS_SUCCESS) {
        printf("ERROR: K merge computation failed with status: %d\n", k_status);
        ggml_free(ctx);
        return 1;
    }
    
    printf("K merge computation successful\n");
    print_tensor_samples("k_merged", k_merged, 20);
    
    bool k_verify = verify_merge_result(k_merged, k_fp16_shard, k_quant_shard);

    // 测试V tensor合并
    printf("\n--- Testing V tensor merge ---\n");
    struct ggml_cgraph * v_merge_graph = build_merge_kv_graph(ctx, v_merged, v_fp16_shard, v_quant_shard);
    
    printf("Computing V merge graph...\n");
    enum ggml_status v_status = ggml_graph_compute_with_ctx(ctx, v_merge_graph, 4);
    if (v_status != GGML_STATUS_SUCCESS) {
        printf("ERROR: V merge computation failed with status: %d\n", v_status);
        ggml_free(ctx);
        return 1;
    }
    
    printf("V merge computation successful\n");
    print_tensor_samples("v_merged", v_merged, 20);
    
    bool v_verify = verify_merge_result(v_merged, v_fp16_shard, v_quant_shard);

    // 测试mask合并
    printf("\n--- Testing mask merge ---\n");
    
    // 创建mask分片 - mask的shape是[kv_len, q_len, 1, batch]
    // 使用F16类型因为量化类型不能很好地处理-INFINITY值
    ggml_tensor * mask_fp16_shard = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, fp16_seq_len, total_seq_len, 1, batch);
    ggml_tensor * mask_quant_shard = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, quant_seq_len, total_seq_len, 1, batch);
    ggml_tensor * mask_merged = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, total_seq_len, total_seq_len, 1, batch);
    
    printf("Created mask tensors:\n");
    printf("  mask_fp16_shard: [%ld, %ld, %ld, %ld]\n", mask_fp16_shard->ne[0], mask_fp16_shard->ne[1], mask_fp16_shard->ne[2], mask_fp16_shard->ne[3]);
    printf("  mask_quant_shard: [%ld, %ld, %ld, %ld]\n", mask_quant_shard->ne[0], mask_quant_shard->ne[1], mask_quant_shard->ne[2], mask_quant_shard->ne[3]);
    printf("  mask_merged: [%ld, %ld, %ld, %ld]\n", mask_merged->ne[0], mask_merged->ne[1], mask_merged->ne[2], mask_merged->ne[3]);
    
    // 填充mask数据
    printf("Filling mask data...\n");
    fill_mask_tensor(mask_fp16_shard);
    fill_mask_tensor(mask_quant_shard);
    
    printf("Filled mask tensors with causal patterns\n");
    print_mask_samples("mask_fp16_shard", mask_fp16_shard, 20);
    print_mask_samples("mask_quant_shard", mask_quant_shard, 20);
    
    struct ggml_cgraph * mask_merge_graph = build_merge_mask_graph(ctx, mask_merged, mask_fp16_shard, mask_quant_shard);
    
    printf("Computing mask merge graph...\n");
    enum ggml_status mask_status = ggml_graph_compute_with_ctx(ctx, mask_merge_graph, 4);
    if (mask_status != GGML_STATUS_SUCCESS) {
        printf("ERROR: Mask merge computation failed with status: %d\n", mask_status);
        ggml_free(ctx);
        return 1;
    }
    
    printf("Mask merge computation successful\n");
    print_mask_samples("mask_merged", mask_merged, 20);
    
    bool mask_verify = verify_mask_merge_result(mask_merged, mask_fp16_shard, mask_quant_shard);

    // 输出最终结果
    printf("\n=== Test Results ===\n");
    printf("K tensor merge: %s\n", k_verify ? "PASSED" : "FAILED");
    printf("V tensor merge: %s\n", v_verify ? "PASSED" : "FAILED");
    printf("Mask merge: %s\n", mask_verify ? "PASSED" : "FAILED");
    
    bool overall_success = k_verify && v_verify && mask_verify;
    printf("Overall test: %s\n", overall_success ? "PASSED" : "FAILED");

    // 清理
    ggml_free(ctx);
    
    return overall_success ? 0 : 1;
} 