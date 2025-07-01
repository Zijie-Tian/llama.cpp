#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

// PyTorch headers (when available)
#include <cstdlib>
#ifdef LLAMA_TORCH_AVAILABLE
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#endif

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <algorithm>

struct kqv_tensor_params {
    std::string input_file;
    bool verbose = false;
    bool show_data_stats = false;
    bool show_shape_details = false;
    int target_step = -1; // -1 means show all steps
    int target_layer = -1; // -1 means show all layers
};

static void print_usage(const char* program_name) {
    LOG_INF("Usage: %s [options]\n", program_name);
    LOG_INF("Options:\n");
    LOG_INF("  -i, --input <file>        Input GGUF file to read (required)\n");
    LOG_INF("  --shapes                  Show detailed shape and stride information\n");
    LOG_INF("  -h, --help                Show this help message\n");
    LOG_INF("\n");
    LOG_INF("Description:\n");
    LOG_INF("  Specialized tool to read and analyze kqv_out tensors and their direct\n");
    LOG_INF("  source tensors (QKV, mask) from GGUF files saved by kqv-trace-monitor.\n");
    LOG_INF("  Flash attention computation is automatically performed on all detected steps.\n");
    LOG_INF("\n");
    LOG_INF("Examples:\n");
    LOG_INF("  %s -i tensors.gguf                # Basic tensor listing with flash attention\n", program_name);
    LOG_INF("  %s -i tensors.gguf --shapes       # Show detailed shape information with flash attention\n", program_name);
}

static bool parse_args(int argc, char** argv, kqv_tensor_params& params) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (++i >= argc) {
                LOG_ERR("Error: --input requires a filename\n");
                return false;
            }
            params.input_file = argv[i];
        } else if (strcmp(argv[i], "--shapes") == 0) {
            params.show_shape_details = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return false;
        } else {
            LOG_ERR("Error: Unknown argument '%s'\n", argv[i]);
            return false;
        }
    }

    if (params.input_file.empty()) {
        LOG_ERR("Error: Input file is required (use -i or --input)\n");
        return false;
    }

    return true;
}

static int extract_step_from_name(const std::string& name) {
    size_t step_pos = name.find("_step_");
    if (step_pos != std::string::npos) {
        size_t start = step_pos + 6; // Position after "_step_"
        if (start < name.length()) {
            size_t end = start;
            while (end < name.length() && std::isdigit(name[end])) {
                end++;
            }
            if (end > start) {
                try {
                    return std::stoi(name.substr(start, end - start));
                } catch (...) {
                    return -1;
                }
            }
        }
    }
    return -1;
}

struct tensor_stats {
    double mean = 0.0;
    double std_dev = 0.0;
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    size_t elements = 0;
};

// Flash attention model structure
struct flash_attn_model {
    struct ggml_tensor * Q;
    struct ggml_tensor * K;
    struct ggml_tensor * V;
    struct ggml_tensor * K_quant;
    struct ggml_tensor * V_quant;
    struct ggml_tensor * mask;
    struct ggml_context * ctx;
};

// Initialize flash attention model with Q, K, V tensors
static bool init_flash_attn_model(
    flash_attn_model & model,
    ggml_tensor* q_src,
    ggml_tensor* k_src,
    ggml_tensor* v_src,
    ggml_tensor* mask_src = nullptr,
    ggml_tensor* k_quant_src = nullptr,
    ggml_tensor* v_quant_src = nullptr
) {
    // Calculate context size needed
    size_t ctx_size = 0;
    ctx_size += ggml_nbytes(q_src);
    ctx_size += ggml_nbytes(k_src);
    ctx_size += ggml_nbytes(v_src);
    if (mask_src) {
        ctx_size += ggml_nbytes(mask_src);
    }

    // Add space for result tensor (estimated)
    size_t result_size = q_src->ne[0] * q_src->ne[1] * q_src->ne[2] * q_src->ne[3] * ggml_type_size(GGML_TYPE_F32);
    ctx_size += result_size;

    ctx_size += 4 * ggml_tensor_overhead(); // tensors
    ctx_size += ggml_graph_overhead(); // compute graph
    ctx_size += 1024 * 1024; // extra overhead

    struct ggml_init_params params {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    // create context
    model.ctx = ggml_init(params);
    if (!model.ctx) {
        LOG_ERR("Failed to create ggml context for flash attention\n");
        return false;
    }

    // Create new tensors with same shapes and copy data
    model.Q = ggml_new_tensor_4d(model.ctx, q_src->type, q_src->ne[0], q_src->ne[1], q_src->ne[2], q_src->ne[3]);
    model.K = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, k_src->ne[0], k_src->ne[1], k_src->ne[2], k_src->ne[3]);
    model.V = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, v_src->ne[0], v_src->ne[1], v_src->ne[2], v_src->ne[3]);
    model.K_quant = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, k_quant_src->ne[0], k_quant_src->ne[1], k_quant_src->ne[2], k_quant_src->ne[3]);
    model.V_quant = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, v_quant_src->ne[0], v_quant_src->ne[1], v_quant_src->ne[2], v_quant_src->ne[3]);

    if (mask_src) {
        model.mask = ggml_new_tensor_4d(model.ctx, mask_src->type, mask_src->ne[0], mask_src->ne[1], mask_src->ne[2], mask_src->ne[3]);
        memcpy(model.mask->data, mask_src->data, ggml_nbytes(mask_src));
    } else {
        model.mask = nullptr;
    }

    // Copy data
    memcpy(model.Q->data, q_src->data, ggml_nbytes(q_src));

    // ggml_fp32_to_fp16_row((const float*)k_src->data, (ggml_fp16_t*)model.K->data, ggml_nelements(k_src));
    // ggml_fp32_to_fp16_row((const float*)v_src->data, (ggml_fp16_t*)model.V->data, ggml_nelements(v_src));

    return true;
}

static struct ggml_cgraph * build_merge_mask_graph(
    ggml_context * ctx,
    ggml_tensor * dst,
    ggml_tensor * fp16_shard,
    ggml_tensor * quant_shard
) {
    // printf("Building mask merge graph...\n");
    // printf("  dst: [%ld, %ld, %ld, %ld]\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    // printf("  fp16_shard: [%ld, %ld, %ld, %ld]\n", fp16_shard->ne[0], fp16_shard->ne[1], fp16_shard->ne[2], fp16_shard->ne[3]);
    // printf("  quant_shard: [%ld, %ld, %ld, %ld]\n", quant_shard->ne[0], quant_shard->ne[1], quant_shard->ne[2], quant_shard->ne[3]);
    
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

// 合并两个KV tensor分片的函数 - 使用1D视图简化操作
static struct ggml_cgraph * build_merge_kv_graph(
    ggml_context * ctx,
    ggml_tensor * dst,
    ggml_tensor * fp16_shard,
    ggml_tensor * quant_shard
) {
    // printf("Building merge graph...\n");
    // printf("  dst: [%ld, %ld, %ld, %ld]\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    // printf("  fp16_shard: [%ld, %ld, %ld, %ld]\n", fp16_shard->ne[0], fp16_shard->ne[1], fp16_shard->ne[2], fp16_shard->ne[3]);
    // printf("  quant_shard: [%ld, %ld, %ld, %ld]\n", quant_shard->ne[0], quant_shard->ne[1], quant_shard->ne[2], quant_shard->ne[3]);
    
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

// Build computation graph for flash attention
static struct ggml_cgraph * build_flash_attn_graph(
    ggml_context* ctx,
    ggml_tensor* Q,
    ggml_tensor* K,
    ggml_tensor* V,
    ggml_tensor* mask,
    float scale = 1.0f,
    float max_bias = 0.0f,
    float logit_softcap = 0.0f
) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // Perform flash attention: result = flash_attn_ext(Q, K, V, mask)
    struct ggml_tensor * result = ggml_flash_attn_ext(
        ctx,
        Q,
        K,
        V,
        mask,
        scale,
        max_bias,
        logit_softcap
    );
    ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);

    ggml_build_forward_expand(gf, result);
    return gf;
}

static struct ggml_cgraph * build_flash_attn_with_state_graph(
    ggml_context* ctx,
    ggml_tensor* Q,
    ggml_tensor* K,
    ggml_tensor* V,
    ggml_tensor* mask,
    ggml_tensor* K_quant,
    ggml_tensor* V_quant,
    ggml_tensor* qk_mask_quant,
    float scale         = 1.0f,
    float max_bias      = 0.0f,
    float logit_softcap = 0.0f
) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // Perform flash attention: result = flash_attn_ext(Q, K, V, mask)
    struct ggml_tensor * result = ggml_flash_attn_ext_with_state(
        ctx,
        Q,
        K,
        V,
        mask,
        K_quant,
        V_quant,
        qk_mask_quant,
        scale,
        max_bias,
        logit_softcap
    );
    ggml_flash_attn_ext_set_prec(result, GGML_PREC_WITH_STATE);

    ggml_build_forward_expand(gf, result);

    return gf;
}

static struct ggml_cgraph * build_subtract_graph(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b
) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * result = ggml_sub(ctx, a, b);
    ggml_build_forward_expand(gf, result);

    return gf;
}

static struct ggml_tensor * compute_graph(
    ggml_context* ctx,
    struct ggml_cgraph * gf,
    int n_threads = 12
) {
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    return ggml_graph_node(gf, -1);
}

// Professional tensor printing function similar to ggml_print_tensor
static void ggml_print_tensor_info(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, const std::string& name, int64_t n = 3) {
    if (!data || n <= 0) {
        LOG_INF("Tensor %s: NULL or invalid data\n", name.c_str());
        return;
    }

    LOG_INF("\n=== Tensor: %s ===\n", name.c_str());
    LOG_INF("Type: %s, Shape: [%ld, %ld, %ld, %ld]\n", ggml_type_name(type), ne[0], ne[1], ne[2], ne[3]);

    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG_INF("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                LOG_INF("                                      ..., \n");
                i2 = ne[2] - n;
            }
            LOG_INF("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    LOG_INF("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                LOG_INF("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        LOG_INF("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        v = 0.0f; // fallback for unsupported types
                    }
                    LOG_INF("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) LOG_INF(", ");
                }
                LOG_INF("],\n");
            }
            LOG_INF("                                      ],\n");
        }
        LOG_INF("                                     ]\n");
    }
    LOG_INF("Sum: %.6f\n", sum);
    LOG_INF("================\n\n");
}

// Simple tensor info without detailed data
static void print_tensor_summary(ggml_tensor* tensor, const std::string& name) {
    if (!tensor) {
        LOG_INF("| %-20s | NULL                                | NULL                                | NULL     | NULL       |\n", name.c_str());
        return;
    }
    LOG_INF("| %-21s | [%4ld,%4ld,%4ld,%4ld]                 | [%8ld,%8ld,%8ld,%8ld]            | %-8s | %10zu |\n",
            name.c_str(), 
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3],
            ggml_type_name(tensor->type), ggml_nelements(tensor));
}

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

/**
 * Print a visualization of the KQV attention mask.
 * Shows which tokens can attend to which other tokens.
 * x = can attend (0 or greater)
 * - = cannot attend (-INFINITY)
 */
static void print_kqv_mask(ggml_tensor* mask) {
    GGML_TENSOR_LOCALS(int64_t, ne_mask, mask, ne)
    GGML_TENSOR_LOCALS(int64_t, nb_mask, mask, nb)

    LOG_INF("\n=== KQV Attention Mask ===\n");
    LOG_INF("KV tokens →\n");

    // Print column numbers
    for (int i = 0; i < ne_mask0; i++) {
        LOG_INF("%d", i % 10);
    }
    LOG_INF("\n");

    // Print separator
    for (int i = 0; i < ne_mask0; i++) {
        LOG_INF("=");
    }
    LOG_INF("\n");

    // Get mask data pointer
    const char* mask_nonfp32 = (const char*)mask->data;
    const float* mask_fp32 = (const float*)mask->data;

    ggml_to_float_t to_float = ggml_get_type_traits(mask->type)->to_float;

    float* row_buffer = (float*)malloc(ne_mask0 * sizeof(float));
    for (int i = 0; i < ne_mask1; ++i) {
        if (to_float) {
            to_float(mask_nonfp32 + i * nb_mask1, row_buffer, ne_mask0);

            for (int j = 0; j < ne_mask0; ++j) {
                if (row_buffer[j] == 0.f) {
                    LOG_INF("x");
                } else if (row_buffer[j] == -INFINITY) {
                    LOG_INF("-");
                } else {
                    LOG_INF("?");
                }
            }
            LOG_INF("\n");
        } else {
            for (int j = 0; j < ne_mask0; ++j) {
                if (mask_fp32[j] == 0.f) {
                    LOG_INF("x");
                } else if (mask_fp32[j] == -INFINITY) {
                    LOG_INF("-");
                } else {
                    LOG_INF("?");
                }
            }
            LOG_INF("\n");
        }
    }

    free(row_buffer);
}

/**
 * Count valid KV cache entries in the first row of the attention mask.
 * Valid entries have value 0.0f, invalid entries have -INFINITY.
 * 
 * @param mask: attention mask tensor with shape [kv_len, q_len, 1, n_batch]
 * @param batch_idx: which batch to check (default 0 for first batch)
 * @return: number of valid KV cache entries in the first row
 */
static int count_valid_kv_cache_first_row(ggml_tensor* mask, int batch_idx = 0) {
    if (!mask || !mask->data) {
        LOG_ERR("Error: mask tensor is null or has no data\n");
        return 0;
    }

    GGML_TENSOR_LOCALS(int64_t, ne_mask, mask, ne)
    GGML_TENSOR_LOCALS(int64_t, nb_mask, mask, nb)

    // Validate batch index
    if (batch_idx >= ne_mask3) {
        LOG_ERR("Error: batch_idx %d exceeds n_batch %ld\n", batch_idx, ne_mask3);
        return 0;
    }

    // Validate tensor shape
    if (ne_mask2 != 1) {
        LOG_ERR("Error: expected ne[2] to be 1, got %ld\n", ne_mask2);
        return 0;
    }

    if (ne_mask1 == 0 || ne_mask0 == 0) {
        LOG_INF("Warning: empty mask tensor dimensions [%ld, %ld]\n", ne_mask0, ne_mask1);
        return 0;
    }

    int valid_count = 0;
    const int q_row = 0; // First row (q_len = 0)

    // Get mask data pointer for the specified batch
    const char* mask_nonfp32 = (const char*)mask->data;
    const float* mask_fp32 = (const float*)mask->data;

    ggml_to_float_t to_float = ggml_get_type_traits(mask->type)->to_float;

    if (to_float) {
        // For non-F32 types, convert to float first
        float* row_buffer = (float*)malloc(ne_mask0 * sizeof(float));
        if (!row_buffer) {
            LOG_ERR("Error: failed to allocate memory for row buffer\n");
            return 0;
        }

        // Calculate offset for the first row of the specified batch
        // Offset = batch_idx * nb[3] + 0 * nb[2] + q_row * nb[1] + 0 * nb[0]
        size_t batch_offset = batch_idx * nb_mask3 + q_row * nb_mask1;
        
        to_float(mask_nonfp32 + batch_offset, row_buffer, ne_mask0);

        // Count valid entries (value == 0.0f)
        for (int kv_idx = 0; kv_idx < ne_mask0; ++kv_idx) {
            if (row_buffer[kv_idx] == 0.0f) {
                valid_count++;
            }
        }

        free(row_buffer);
    } else {
        // For F32 type, access directly
        // Calculate offset for the first row of the specified batch
        size_t batch_offset = batch_idx * (nb_mask3 / sizeof(float)) + q_row * (nb_mask1 / sizeof(float));
        
        for (int kv_idx = 0; kv_idx < ne_mask0; ++kv_idx) {
            if (mask_fp32[batch_offset + kv_idx] == 0.0f) {
                valid_count++;
            }
        }
    }

    // LOG_INF("Valid KV cache count in first row (batch %d): %d / %ld\n", 
    //         batch_idx, valid_count, ne_mask0);

    return valid_count;
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG_INF("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                LOG_INF("                                      ..., \n");
                i2 = ne[2] - n;
            }
            LOG_INF("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    LOG_INF("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                LOG_INF("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        LOG_INF("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    LOG_INF("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) LOG_INF(", ");
                }
                LOG_INF("],\n");
            }
            LOG_INF("                                      ],\n");
        }
        LOG_INF("                                     ]\n");
        LOG_INF("                                     sum = %f\n", sum);
    }
}

static ggml_tensor* ggml_reshape_tensor(ggml_context* ctx, ggml_tensor* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {

    ggml_tensor* reshaped_tensor = ggml_reshape_4d(ctx, tensor, d0, d1, d2, d3);
    reshaped_tensor = ggml_cont(ctx, ggml_permute(ctx, reshaped_tensor, 0, 2, 1, 3));
    // LOG_INF("Reshape from [%lld, %lld, %lld, %lld] to [%lld, %lld, %lld, %lld]\n", d0, d1, d2, d3, reshaped_tensor->ne[0], reshaped_tensor->ne[1], reshaped_tensor->ne[2], reshaped_tensor->ne[3]);

    struct ggml_cgraph *g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, reshaped_tensor);          // 仅需把最终 dst 挂进图
    ggml_graph_compute_with_ctx(ctx, g, 8);              // 🚀 真正触发 memcpy

    return reshaped_tensor;
}

static tensor_stats compute_tensor_stats(ggml_tensor* tensor) {
    tensor_stats stats;
    
    if (!tensor || !tensor->data) {
        return stats;
    }
    
    size_t n_elements = ggml_nelements(tensor);
    stats.elements = n_elements;
    
    if (n_elements == 0) {
        return stats;
    }
    
    // Convert tensor data to float for computation
    std::vector<float> data(n_elements);
    
    if (tensor->type == GGML_TYPE_F32) {
        memcpy(data.data(), tensor->data, n_elements * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t* fp16_data = (ggml_fp16_t*)tensor->data;
        for (size_t i = 0; i < n_elements; i++) {
            data[i] = ggml_fp16_to_fp32(fp16_data[i]);
        }
    } else {
        // For other types, try to get conversion function
        auto tt = ggml_get_type_traits(tensor->type);
        if (tt->to_float) {
            tt->to_float(tensor->data, data.data(), n_elements);
        } else {
            LOG_ERR("Unsupported tensor type for stats computation: %s\n", ggml_type_name(tensor->type));
            return stats;
        }
    }
    
    // Compute statistics
    double sum = 0.0;
    stats.min_val = data[0];
    stats.max_val = data[0];
    
    for (size_t i = 0; i < n_elements; i++) {
        double val = data[i];
        sum += val;
        stats.min_val = std::min(stats.min_val, val);
        stats.max_val = std::max(stats.max_val, val);
    }
    
    stats.mean = sum / n_elements;
    
    // Compute standard deviation
    double var_sum = 0.0;
    for (size_t i = 0; i < n_elements; i++) {
        double diff = data[i] - stats.mean;
        var_sum += diff * diff;
    }
    stats.std_dev = std::sqrt(var_sum / n_elements);
    
    return stats;
}

// Calculate NMSE between two tensors
static double calculate_nmse(ggml_tensor* predicted, ggml_tensor* target) {
    if (!predicted || !target || !predicted->data || !target->data) {
        LOG_ERR("Invalid tensors for NMSE calculation\n");
        return -1.0;
    }
    
    size_t n_elements_pred = ggml_nelements(predicted);
    size_t n_elements_target = ggml_nelements(target);
    
    if (n_elements_pred != n_elements_target) {
        LOG_ERR("Tensor size mismatch: predicted=%zu, target=%zu\n", n_elements_pred, n_elements_target);
        return -1.0;
    }
    
    if (n_elements_pred == 0) {
        LOG_ERR("Empty tensors for NMSE calculation\n");
        return -1.0;
    }
    
    // Convert tensors to float arrays
    std::vector<float> pred_data(n_elements_pred);
    std::vector<float> target_data(n_elements_target);
    
    // Convert predicted tensor
    if (predicted->type == GGML_TYPE_F32) {
        memcpy(pred_data.data(), predicted->data, n_elements_pred * sizeof(float));
    } else if (predicted->type == GGML_TYPE_F16) {
        ggml_fp16_t* fp16_data = (ggml_fp16_t*)predicted->data;
        for (size_t i = 0; i < n_elements_pred; i++) {
            pred_data[i] = ggml_fp16_to_fp32(fp16_data[i]);
        }
    } else {
        auto tt = ggml_get_type_traits(predicted->type);
        if (tt->to_float) {
            tt->to_float(predicted->data, pred_data.data(), n_elements_pred);
        } else {
            LOG_ERR("Unsupported predicted tensor type: %s\n", ggml_type_name(predicted->type));
            return -1.0;
        }
    }
    
    // Convert target tensor
    if (target->type == GGML_TYPE_F32) {
        memcpy(target_data.data(), target->data, n_elements_target * sizeof(float));
    } else if (target->type == GGML_TYPE_F16) {
        ggml_fp16_t* fp16_data = (ggml_fp16_t*)target->data;
        for (size_t i = 0; i < n_elements_target; i++) {
            target_data[i] = ggml_fp16_to_fp32(fp16_data[i]);
        }
    } else {
        auto tt = ggml_get_type_traits(target->type);
        if (tt->to_float) {
            tt->to_float(target->data, target_data.data(), n_elements_target);
        } else {
            LOG_ERR("Unsupported target tensor type: %s\n", ggml_type_name(target->type));
            return -1.0;
        }
    }
    
    // Calculate MSE and target variance
    double mse_sum = 0.0;
    double target_sum = 0.0;
    double target_sq_sum = 0.0;
    
    for (size_t i = 0; i < n_elements_pred; i++) {
        double diff = pred_data[i] - target_data[i];
        mse_sum += diff * diff;
        target_sum += target_data[i];
        target_sq_sum += target_data[i] * target_data[i];
    }
    
    double mse = mse_sum / n_elements_pred;
    double target_mean = target_sum / n_elements_pred;
    double target_variance = target_sq_sum / n_elements_pred - target_mean * target_mean;
    
    // NMSE = MSE / Var(target)
    // If target variance is too small, use mean squared value instead
    double normalization = (target_variance > 1e-12) ? target_variance : (target_sq_sum / n_elements_pred);
    
    if (normalization < 1e-12) {
        LOG_INF("Warning: Target tensor has near-zero variance, NMSE may not be meaningful\n");
        return mse; // Return raw MSE if no meaningful normalization
    }
    
    double nmse = mse / normalization;
    
    return nmse;
}

// Print detailed comparison statistics in table format
static void print_comparison_stats(ggml_tensor* predicted, ggml_tensor* target, double nmse) {
    tensor_stats pred_stats = compute_tensor_stats(predicted);
    tensor_stats target_stats = compute_tensor_stats(target);
    
    // Determine assessment level
    const char* assessment = "";
    const char* quality_icon = "";
    if (nmse < 1e-6) {
        assessment = "Excellent (< 1e-6)";
        quality_icon = "🟢";
    } else if (nmse < 1e-4) {
        assessment = "Very Good (< 1e-4)";
        quality_icon = "🟢";
    } else if (nmse < 1e-2) {
        assessment = "Good (< 1e-2)";
        quality_icon = "🟡";
    } else if (nmse < 1e-1) {
        assessment = "Acceptable (< 1e-1)";
        quality_icon = "🟡";
    } else {
        assessment = "Poor (>= 1e-1)";
        quality_icon = "🔴";
    }
    
    LOG_INF("\n");
    LOG_INF("+-------------------------+------------+------------+------------+--------------+--------------------+\n");
    LOG_INF("| %-23s | %-10s | %-10s | %-10s | %-12s | %-18s |\n", 
            "Tensor Statistics", "Elements", "Mean", "Std Dev", "Min/Max", "NMSE & Assessment");
    LOG_INF("+-------------------------+------------+------------+------------+--------------+--------------------+\n");
    LOG_INF("| %-23s | %-10zu | %10.6f | %10.6f | %5.3f/%-4.3f | %-18s |\n",
            "Predicted (Flash Attn)", pred_stats.elements, pred_stats.mean, pred_stats.std_dev, 
            pred_stats.min_val, pred_stats.max_val, "");
    LOG_INF("| %-23s | %-10zu | %10.6f | %10.6f | %5.3f/%-4.3f | %-18s |\n",
            "Target (KQV Output)", target_stats.elements, target_stats.mean, target_stats.std_dev,
            target_stats.min_val, target_stats.max_val, "");
    LOG_INF("+-------------------------+------------+------------+------------+--------------+--------------------+\n");
    LOG_INF("| %-23s | %-10s | %-10s | %-10s | %-12s | %s %-15s |\n",
            "Difference Analysis", "-", "-", "-", "-", quality_icon, "");
    LOG_INF("| %-23s | %-10s | %-10s | %-10s | %-12s | %.6e       |\n",
            "NMSE", "-", "-", "-", "-", nmse);
    LOG_INF("| %-23s | %-10s | %-10s | %-10s | %-12s | %-18s |\n",
            "Quality Assessment", "-", "-", "-", "-", assessment);
    LOG_INF("+-------------------------+------------+------------+------------+--------------+--------------------+\n");
    LOG_INF("\n");
}

static bool read_kqv_tensors(const kqv_tensor_params& params) {
    LOG_INF("Reading KQV trace file: %s\n", params.input_file.c_str());
    LOG_INF("Flash attention computation enabled for all steps\n");
    LOG_INF("=====================================\n\n");

    // Load GGUF file
    struct ggml_context* ggml_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ggml_ctx,
    };

    const int n_threads = 1;

    // NOTICE : This is GGUF_CONTEXT
    struct gguf_context* ctx = gguf_init_from_file(params.input_file.c_str(), gguf_params);
    if (!ctx) {
        LOG_ERR("Error: Failed to load GGUF file: %s\n", params.input_file.c_str());
        return false;
    }

    // Get tensor context
    struct ggml_context* tensor_ctx = ggml_ctx;
    if (!tensor_ctx) {
        LOG_ERR("Error: Failed to get tensor context\n");
        gguf_free(ctx);
        return false;
    }

    //> Mapping step to tensors
    std::map<int, std::vector<std::pair<ggml_tensor*, std::string>>> step_tensor_map;
    for (ggml_tensor* tensor = ggml_get_first_tensor(tensor_ctx); tensor; tensor = ggml_get_next_tensor(tensor_ctx, tensor)) {
        std::string name = tensor->name && tensor->name[0] ? tensor->name : "unnamed";
        LOG_INF("Tensor name: %s\n", name.c_str());

        int step = extract_step_from_name(name);
        step_tensor_map[step].emplace_back(tensor, name);
    }

    // Add space for result tensor (estimated)
    struct ggml_init_params ctx_params {
        /*.mem_size   =*/ (size_t)(1 * 1024 * 1024 * 1024),
        /*.mem_buffer =*/ NULL,     //> Just alloc memory use
        /*.no_alloc   =*/ false,
    };

    //> Create compute context
    ggml_context* compute_ctx = ggml_init(ctx_params);

    // Output by step
    std::vector<double> nmse_results;
    std::vector<int> valid_steps;
    
    for (const auto& [step, tensors] : step_tensor_map) {
        if (tensors.size() < 4) {
            LOG_INF("Step %d has %zu tensors, skipping\n", tensors.size(), step);
            continue;
        }
        LOG_DBG("\n==== Step %d ====%s\n", step, (step == -1 ? " (unknown)" : ""));

        ggml_tensor * kqv_out = tensors[0].first;
        ggml_tensor * Q = tensors[1].first;
        ggml_tensor * K = tensors[2].first;
        ggml_tensor * V = tensors[3].first;
        ggml_tensor * kq_mask = tensors.size() > 4 ? tensors[4].first : nullptr;

        // ggml_print_tensor((uint8_t*)K->data, K->type, K->ne, K->nb, 8);
        // ggml_print_tensor((uint8_t*)V->data, V->type, V->ne, V->nb, 8);

        ggml_tensor * K_quant = nullptr;
        ggml_tensor * V_quant = nullptr;
        ggml_tensor * KQ_mask_quant = nullptr;
        if (tensors.size() > 5) {
            K_quant = tensors[5].first;
            V_quant = tensors[6].first;
            KQ_mask_quant = tensors[7].first;
            LOG_DBG("Quantized tensors - K_quant: %s, V_quant: %s, KQ_mask_quant: %s\n",
                    K_quant->name, V_quant->name, KQ_mask_quant->name);
        }

        // Run flash attention for all steps
        LOG_DBG("\nRunning Flash Attention at Step %d\n", step);
        // Print table header for tensor summary
        LOG_INF("\n+-----------------------+---------------------------------------+--------------------------------------------------+----------+------------+\n");
        LOG_INF("| %-21s | %-37s | %-48s | %-8s | %-10s |\n", "Tensor Name", "Dimensions [d0,d1,d2,d3]", "Strides [s0,s1,s2,s3]", "Type", "Elements");
        LOG_INF("+-----------------------+---------------------------------------+--------------------------------------------------+----------+------------+\n");

        // Print input tensor summary (without detailed data)
        print_tensor_summary(Q, "Q (Query)");
        print_tensor_summary(K, "K (Key)");
        print_tensor_summary(V, "V (Value)");

        int valid_count_orig    = 0;
        int valid_count_quant   = 0;
        int valid_count_merged  = 0;

        if (kq_mask) {
            print_tensor_summary(kq_mask, "Mask");
            // Count valid KV cache entries in the original mask
            // valid_count_orig = count_valid_kv_cache_first_row(kq_mask, 0);
        }
        if (K_quant && V_quant) {
            print_tensor_summary(K_quant, "K_quant");
            print_tensor_summary(V_quant, "V_quant");
            print_tensor_summary(KQ_mask_quant, "KQ_mask_quant");
            // Count valid KV cache entries in the quantized mask
            if (KQ_mask_quant) {
                valid_count_quant = count_valid_kv_cache_first_row(KQ_mask_quant, 0);
            }
        }

        // Print table footer
        LOG_INF("+-----------------------+---------------------------------------+--------------------------------------------------+----------+------------+\n");

        // NOTICE : Due to `to_float` can only convert to F32, so here is F32.
        ggml_tensor * K_merged = ggml_new_tensor_4d(
            compute_ctx, GGML_TYPE_F32, 
            K->ne[0], K->ne[1] + K_quant->ne[1], K->ne[2], K->ne[3]
        );
        ggml_tensor * V_merged = ggml_new_tensor_4d(
            compute_ctx, GGML_TYPE_F32, 
            V->ne[0], V->ne[1] + V_quant->ne[1], V->ne[2], V->ne[3]
        );
        ggml_tensor * mask_merged = ggml_new_tensor_4d(
            compute_ctx, GGML_TYPE_F32, 
            kq_mask->ne[0] + kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]
        );

        struct ggml_cgraph * k_merge_graph = build_merge_kv_graph(compute_ctx, K_merged, K, K_quant);
        enum ggml_status k_status = ggml_graph_compute_with_ctx(compute_ctx, k_merge_graph, 1);
        if (k_status != GGML_STATUS_SUCCESS) {
            printf("ERROR: K merge computation failed with status: %d\n", k_status);
            ggml_free(compute_ctx);
            return 1;
        }

        struct ggml_cgraph * v_merge_graph = build_merge_kv_graph(compute_ctx, V_merged, V, V_quant);
        enum ggml_status v_status = ggml_graph_compute_with_ctx(compute_ctx, v_merge_graph, 1);
        if (v_status != GGML_STATUS_SUCCESS) {
            printf("ERROR: V merge computation failed with status: %d\n", v_status);
            ggml_free(compute_ctx);
            return 1;
        }

        struct ggml_cgraph * mask_merge_graph = build_merge_mask_graph(compute_ctx, mask_merged, kq_mask, KQ_mask_quant);
        enum ggml_status mask_status = ggml_graph_compute_with_ctx(compute_ctx, mask_merge_graph, 1);
        if (mask_status != GGML_STATUS_SUCCESS) {
            printf("ERROR: Mask merge computation failed with status: %d\n", mask_status);
            ggml_free(compute_ctx);
            return 1;
        }

        ggml_tensor * K_merged_fp16 = ggml_cast(compute_ctx, K_merged, GGML_TYPE_F16);
        ggml_tensor * V_merged_fp16 = ggml_cast(compute_ctx, V_merged, GGML_TYPE_F16);
        ggml_tensor * mask_merged_fp16 = ggml_cast(compute_ctx, mask_merged, GGML_TYPE_F16);

        print_tensor_summary(K_merged_fp16, "K_merged");
        print_tensor_summary(V_merged_fp16, "V_merged");
        print_tensor_summary(mask_merged_fp16, "mask_merged");

        // Count valid KV cache entries in the merged mask
        if (mask_merged_fp16) {
            valid_count_merged = count_valid_kv_cache_first_row(mask_merged_fp16, 0);
        }

        LOG_INF("+-----------------------+---------------------------------------+--------------------------------------------------+----------+------------+\n");

        // Compute flash attention
        float scale = 1.0f / sqrtf((float)Q->ne[0]); // 1 / sqrt(head_dim)

        //> Build Graph and compute flash attention
        struct ggml_cgraph * flash_result_graph = build_flash_attn_graph(
            compute_ctx, Q, K_merged_fp16, V_merged_fp16, mask_merged_fp16, scale
        );
        ggml_tensor * flash_result = compute_graph(compute_ctx, flash_result_graph, n_threads);

        //> Build Graph and compute flash attention with state
        struct ggml_cgraph * flash_result_with_state_graph = build_flash_attn_with_state_graph(
            compute_ctx, Q, K, V, kq_mask, K_quant, V_quant, KQ_mask_quant, scale
        );
        ggml_tensor * flash_result_with_state = compute_graph(compute_ctx, flash_result_with_state_graph, n_threads);

        ggml_tensor* kqv_out_reshaped = ggml_reshape_tensor(compute_ctx, 
            kqv_out, Q->ne[0], kqv_out->ne[1], Q->ne[2],  kqv_out->ne[3]
        );

        print_tensor_summary(kqv_out_reshaped, "kqv_out_reshaped");
        print_tensor_summary(flash_result, "flash_result");
        print_tensor_summary(flash_result_with_state, "flash_result_state");

        LOG_INF("+-----------------------+---------------------------------------+--------------------------------------------------+----------+------------+\n");

        ggml_tensor * k_fp16_ref = nullptr;
        ggml_tensor * k_quant_part = nullptr;
        if (valid_count_quant != 0) {
            k_fp16_ref = ggml_view_2d(compute_ctx, K, K->ne[0], valid_count_quant, K->nb[1], 0);
            k_quant_part = ggml_view_2d(compute_ctx, K_quant, K_quant->ne[0], valid_count_quant, K_quant->nb[1], 0);
            
            struct ggml_cgraph * gf = ggml_new_graph(compute_ctx);
            ggml_build_forward_expand(gf, k_fp16_ref);
            ggml_build_forward_expand(gf, k_quant_part);
            ggml_graph_compute_with_ctx(compute_ctx, gf, 8);  

            print_tensor_summary(k_fp16_ref, "k_fp16_ref");
            print_tensor_summary(k_quant_part, "k_quant_part");
            LOG_INF("+-----------------------+---------------------------------------+--------------------------------------------------+----------+------------+\n");

            // LOG_INF("K_fp16 : \n");
            // ggml_print_tensor((uint8_t*)k_fp16_ref->data, k_fp16_ref->type, k_fp16_ref->ne, k_fp16_ref->nb, 8);
            // LOG_INF("K_quant : \n");
            // ggml_print_tensor((uint8_t*)k_quant_part->data, k_quant_part->type, k_quant_part->ne, k_quant_part->nb, 8);
        }

        struct ggml_cgraph * substract_graph = build_subtract_graph(compute_ctx, flash_result, flash_result_with_state);
        ggml_tensor * substract_result = compute_graph(compute_ctx, substract_graph, n_threads);

        ggml_print_tensor((uint8_t*)substract_result->data, substract_result->type, substract_result->ne, substract_result->nb, 8);

        // print_kqv_mask(kq_mask);

        // print_kqv_mask(KQ_mask_quant);

        // // > Print the tensor.        
        // LOG_INF("---------------   KQV Output (reshaped)  ---------------\n");
        // ggml_print_tensor((uint8_t*)kqv_out_reshaped->data, kqv_out_reshaped->type, kqv_out_reshaped->ne, kqv_out_reshaped->nb, 8); 
        // LOG_INF("--------------- Standard Flash Attention ---------------\n");
        // ggml_print_tensor((uint8_t*)flash_result->data, flash_result->type, flash_result->ne, flash_result->nb, 8); 
        // LOG_INF("--------------- Flash Attention with State ---------------\n");
        // ggml_print_tensor((uint8_t*)flash_result_with_state->data, flash_result_with_state->type, flash_result_with_state->ne, flash_result_with_state->nb, 8); 

        // Calculate NMSE instead of printing tensors
        double nmse = calculate_nmse(flash_result, kqv_out_reshaped);
        double nmse_state_vs_std = calculate_nmse(flash_result_with_state, flash_result);
        double nmse_with_state = calculate_nmse(flash_result_with_state, kqv_out_reshaped);
        
        if (nmse >= 0.0) {
            LOG_INF("Step %d: Flash Attention vs KQV Output Comparison\n", step);
            print_comparison_stats(flash_result, flash_result_with_state, nmse_state_vs_std);
            print_comparison_stats(flash_result_with_state, kqv_out_reshaped, nmse_with_state);
            nmse_results.push_back(nmse);
            valid_steps.push_back(step);
        } else {
            LOG_ERR("Step %d: Failed to calculate NMSE\n", step);
        }
    }
    
    // Print overall summary
    if (!nmse_results.empty()) {
        LOG_INF("\n========== OVERALL NMSE SUMMARY ==========\n");
        LOG_INF("Total valid steps analyzed: %zu\n", nmse_results.size());
        
        double sum_nmse = 0.0;
        double min_nmse = nmse_results[0];
        double max_nmse = nmse_results[0];
        
        for (size_t i = 0; i < nmse_results.size(); i++) {
            double nmse = nmse_results[i];
            sum_nmse += nmse;
            min_nmse = std::min(min_nmse, nmse);
            max_nmse = std::max(max_nmse, nmse);
            LOG_INF("Step %d: NMSE = %.6e\n", valid_steps[i], nmse);
        }
        
        double mean_nmse = sum_nmse / nmse_results.size();
        
        // Calculate standard deviation
        double var_sum = 0.0;
        for (double nmse : nmse_results) {
            double diff = nmse - mean_nmse;
            var_sum += diff * diff;
        }
        double std_nmse = std::sqrt(var_sum / nmse_results.size());
        
        LOG_INF("\nStatistics:\n");
        LOG_INF("  Mean NMSE: %.6e\n", mean_nmse);
        LOG_INF("  Std NMSE:  %.6e\n", std_nmse);
        LOG_INF("  Min NMSE:  %.6e (Step %d)\n", min_nmse, valid_steps[std::min_element(nmse_results.begin(), nmse_results.end()) - nmse_results.begin()]);
        LOG_INF("  Max NMSE:  %.6e (Step %d)\n", max_nmse, valid_steps[std::max_element(nmse_results.begin(), nmse_results.end()) - nmse_results.begin()]);
        
        // Overall assessment
        LOG_INF("\nOverall Assessment:\n");
        if (mean_nmse < 1e-6) {
            LOG_INF("  🟢 Excellent: All results show excellent agreement\n");
        } else if (mean_nmse < 1e-4) {
            LOG_INF("  🟢 Very Good: Results show very good agreement\n");
        } else if (mean_nmse < 1e-2) {
            LOG_INF("  🟡 Good: Results show good agreement\n");
        } else if (mean_nmse < 1e-1) {
            LOG_INF("  🟡 Acceptable: Results show acceptable agreement\n");
        } else {
            LOG_INF("  🔴 Poor: Results show significant disagreement\n");
        }
        
        // Check consistency
        if (std_nmse / mean_nmse < 0.1) {
            LOG_INF("  📊 Consistency: High (low variance across steps)\n");
        } else if (std_nmse / mean_nmse < 0.5) {
            LOG_INF("  📊 Consistency: Moderate\n");
        } else {
            LOG_INF("  📊 Consistency: Low (high variance across steps)\n");
        }
        
        LOG_INF("==========================================\n");
    } else {
        LOG_ERR("No valid NMSE results obtained\n");
    }
    
    // Free flash attention model
    ggml_free(compute_ctx);

    // Cleanup
    gguf_free(ctx);

    return true;
}

int main(int argc, char** argv) {
    ggml_time_init();

    kqv_tensor_params params;

    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    // Verify torch integration is working
#ifdef LLAMA_TORCH_AVAILABLE
    LOG_INF("PyTorch integration enabled!\n");
    LOG_INF("PyTorch version: %d.%d.%d\n",
            TORCH_VERSION_MAJOR,
            TORCH_VERSION_MINOR,
            TORCH_VERSION_PATCH);

    // Simple test: create a small tensor to verify torch is working
    try {
        torch::Tensor test_tensor = torch::rand({2, 3});
        LOG_INF("PyTorch tensor creation test successful - shape: [%lld, %lld]\n",
                test_tensor.size(0), test_tensor.size(1));
    } catch (const std::exception& e) {
        LOG_ERR("PyTorch tensor creation test failed: %s\n", e.what());
    }
#else
    LOG_INF("PyTorch integration disabled - using ggml fallback\n");
#endif

    if (!read_kqv_tensors(params)) {
        return 1;
    }

    return 0;
}




