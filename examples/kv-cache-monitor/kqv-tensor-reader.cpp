#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

// PyTorch headers (when available)
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>

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

// Build computation graph for flash attention
static struct ggml_cgraph * build_flash_attn_graph(
    ggml_context* ctx,
    ggml_tensor* Q, 
    ggml_tensor* K, 
    ggml_tensor* V, 
    ggml_tensor* mask,
    ggml_tensor* K_quant,
    ggml_tensor* V_quant,
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

    // struct ggml_tensor * result = ggml_flash_attn_mixed(
    //     model.ctx, 
    //     model.Q, 
    //     model.K, 
    //     model.V, 
    //     NULL,
    //     NULL,
    //     model.mask, 
    //     scale, 
    //     max_bias, 
    //     logit_softcap
    // );

    result = ggml_reshape_2d(ctx, result, result->ne[0] * result->ne[1], result->ne[2]);

    ggml_build_forward_expand(gf, result);
    return gf;
}

// Compute flash attention using PyTorch
static struct ggml_tensor * compute_flash_attn(
    ggml_context* ctx,
    ggml_tensor* Q, 
    ggml_tensor* K, 
    ggml_tensor* V, 
    ggml_tensor* mask,
    ggml_tensor* K_quant,
    ggml_tensor* V_quant,
    float scale = 1.0f
) {
    try {
        // Extract dimensions from Q tensor [head_dim, seq_len, n_heads, 1]
        const int64_t head_dim = Q->ne[0];
        const int64_t seq_len = Q->ne[1];
        const int64_t n_heads = Q->ne[2];
        
        // Extract dimensions from K/V tensor [head_dim, kv_len, n_kv_heads, 1]
        const int64_t kv_len = K->ne[1];
        const int64_t n_kv_heads = K->ne[2];
        
        LOG_INF("PyTorch Flash Attention Debug Info:\n");
        LOG_INF("  Q: [%ld,%ld,%ld,%ld] -> [head_dim=%ld, seq_len=%ld, n_heads=%ld], dtype=%s\n",
                Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3], head_dim, seq_len, n_heads, ggml_type_name(Q->type));
        LOG_INF("  K: [%ld,%ld,%ld,%ld] -> [head_dim=%ld, kv_len=%ld, n_kv_heads=%ld], dtype=%s\n",
                K->ne[0], K->ne[1], K->ne[2], K->ne[3], head_dim, kv_len, n_kv_heads, ggml_type_name(K->type));
        LOG_INF("  V: [%ld,%ld,%ld,%ld] -> [head_dim=%ld, kv_len=%ld, n_kv_heads=%ld], dtype=%s\n",
                V->ne[0], V->ne[1], V->ne[2], V->ne[3], head_dim, kv_len, n_kv_heads, ggml_type_name(V->type));
        if (mask) {
            LOG_INF("  Mask: [%ld,%ld,%ld,%ld], dtype=%s\n", mask->ne[0], mask->ne[1], mask->ne[2], mask->ne[3], ggml_type_name(mask->type));
        }
        
        auto torch_options = torch::TensorOptions().dtype(torch::kFloat32);
        
        // Create PyTorch tensors in format [batch_size, n_heads, seq_len, head_dim]
        auto q_torch = torch::zeros({1, n_heads, seq_len, head_dim}, torch_options);
        float* q_torch_data = q_torch.data_ptr<float>();
        
        // Convert Q from ggml format [head_dim, seq_len, n_heads, 1] to torch format [1, n_heads, seq_len, head_dim]
        float* q_data = (float*)Q->data;
        for (int h = 0; h < n_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    int ggml_idx = d + s * head_dim + h * head_dim * seq_len;
                    int torch_idx = h * seq_len * head_dim + s * head_dim + d;
                    q_torch_data[torch_idx] = q_data[ggml_idx];
                }
            }
        }
        
        // Create K tensor in PyTorch format [1, n_kv_heads, kv_len, head_dim]
        auto k_torch = torch::zeros({1, n_kv_heads, kv_len, head_dim}, torch_options);
        float* k_torch_data = k_torch.data_ptr<float>();
        
        // Convert K from ggml format [head_dim, kv_len, n_kv_heads, 1] to torch format
        for (int h = 0; h < n_kv_heads; h++) {
            for (int s = 0; s < kv_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    int ggml_idx = d + s * head_dim + h * head_dim * kv_len;
                    int torch_idx = h * kv_len * head_dim + s * head_dim + d;
                    
                    if (K->type == GGML_TYPE_F16) {
                        k_torch_data[torch_idx] = ggml_fp16_to_fp32(((ggml_fp16_t*)K->data)[ggml_idx]);
                    } else {
                        k_torch_data[torch_idx] = ((float*)K->data)[ggml_idx];
                    }
                }
            }
        }
        
        // Create V tensor in PyTorch format [1, n_kv_heads, kv_len, head_dim]
        auto v_torch = torch::zeros({1, n_kv_heads, kv_len, head_dim}, torch_options);
        float* v_torch_data = v_torch.data_ptr<float>();
        
        // Convert V from ggml format [head_dim, kv_len, n_kv_heads, 1] to torch format
        for (int h = 0; h < n_kv_heads; h++) {
            for (int s = 0; s < kv_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    int ggml_idx = d + s * head_dim + h * head_dim * kv_len;
                    int torch_idx = h * kv_len * head_dim + s * head_dim + d;
                    
                    if (V->type == GGML_TYPE_F16) {
                        v_torch_data[torch_idx] = ggml_fp16_to_fp32(((ggml_fp16_t*)V->data)[ggml_idx]);
                    } else {
                        v_torch_data[torch_idx] = ((float*)V->data)[ggml_idx];
                    }
                }
            }
        }
        
        // Handle mask conversion if present
        torch::Tensor mask_torch;
        if (mask && mask->data) {
            std::vector<float> mask_subset;
            mask_subset.reserve(kv_len);
            
            // Extract mask[0, 0:kv_len] - first query token's mask for all kv tokens
            for (int64_t k = 0; k < kv_len; k++) {
                if (k < mask->ne[0]) {  // Ensure we don't go out of bounds
                    float mask_val = ((float*)mask->data)[k];  // mask[k, 0] in ggml layout
                    mask_subset.push_back(mask_val);
                } else {
                    mask_subset.push_back(-INFINITY);  // Mask out if beyond bounds
                }
            }
            
            // Create PyTorch mask: [1, n_heads, seq_len, kv_len] = [1, 32, 1, 32]
            // Convert -inf to false, finite values to true (PyTorch convention)
            std::vector<uint8_t> bool_mask;
            bool_mask.reserve(kv_len);
            for (float val : mask_subset) {
                bool_mask.push_back(std::isfinite(val) ? 1 : 0);
            }
            
            // Create tensor and expand to [1, n_heads, 1, kv_len]
            auto mask_1d = torch::from_blob(bool_mask.data(), {kv_len}, torch::kUInt8).to(torch::kBool);
            mask_torch = mask_1d.unsqueeze(0).unsqueeze(0).unsqueeze(0); // [1, 1, 1, kv_len]
            mask_torch = mask_torch.expand({1, n_heads, seq_len, kv_len});
            
            LOG_INF("  Final mask tensor shape: [%ld, %ld, %ld, %ld]\n",
                    mask_torch.size(0), mask_torch.size(1), mask_torch.size(2), mask_torch.size(3));
        }
        
        // Handle GQA (Grouped Query Attention) by repeating KV heads to match Q heads
        if (n_heads > n_kv_heads) {
            int repeat_factor = n_heads / n_kv_heads;
            k_torch = k_torch.repeat_interleave(repeat_factor, /*dim=*/1);
            v_torch = v_torch.repeat_interleave(repeat_factor, /*dim=*/1);
            
            LOG_INF("GQA: Repeated KV heads by factor %d (%ld -> %ld heads)\n", 
                    repeat_factor, n_kv_heads, n_heads);
        }
        
        LOG_INF("Final PyTorch tensor shapes:\n");
        LOG_INF("  Q: [%ld,%ld,%ld,%ld]\n", q_torch.size(0), q_torch.size(1), q_torch.size(2), q_torch.size(3));
        LOG_INF("  K: [%ld,%ld,%ld,%ld]\n", k_torch.size(0), k_torch.size(1), k_torch.size(2), k_torch.size(3));
        LOG_INF("  V: [%ld,%ld,%ld,%ld]\n", v_torch.size(0), v_torch.size(1), v_torch.size(2), v_torch.size(3));
        
        // Compute scaled dot product attention using PyTorch
        torch::Tensor torch_result;
        if (mask && mask->data) {
            LOG_INF("Computing attention WITH mask, scale=%.6f\n", scale);
            torch_result = torch::scaled_dot_product_attention(
                q_torch, k_torch, v_torch, mask_torch,
                /*dropout_p=*/0.0,
                /*is_causal=*/false,
                /*scale=*/scale
            );
        } else {
            LOG_INF("Computing attention WITHOUT mask, scale=%.6f\n", scale);
            torch_result = torch::scaled_dot_product_attention(
                q_torch, k_torch, v_torch, torch::Tensor(),
                /*dropout_p=*/0.0,
                /*is_causal=*/false,
                /*scale=*/scale
            );
        }
        
        LOG_INF("PyTorch attention result shape: [%ld, %ld, %ld, %ld]\n",
                torch_result.size(0), torch_result.size(1), torch_result.size(2), torch_result.size(3));
        
        // Convert result back to ggml format
        // PyTorch result: [1, n_heads, seq_len, head_dim] = [1, 32, 1, 128]
        // ggml expected: [head_dim*n_heads, seq_len, 1, 1] = [4096, 1, 1, 1]
        
        // Create output tensor with ggml expected format
        ggml_tensor* result_ggml = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                                      head_dim * n_heads, seq_len, 1, 1);
        
        // Copy data from PyTorch result to ggml format
        // PyTorch layout: [batch, head, seq, dim] -> [1, 32, 1, 128]
        // ggml layout: [dim*head, seq, 1, 1] -> [4096, 1, 1, 1]
        float* result_data = (float*)result_ggml->data;
        const float* torch_data = torch_result.data_ptr<float>();
        
        for (int64_t h = 0; h < n_heads; h++) {
            for (int64_t s = 0; s < seq_len; s++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    // PyTorch index: [0, h, s, d]
                    int64_t torch_idx = h * seq_len * head_dim + s * head_dim + d;
                    // ggml index: [h*head_dim + d, s, 0, 0] -> linear: (h*head_dim + d) + s*4096
                    int64_t ggml_idx = (h * head_dim + d) + s * (head_dim * n_heads);
                    result_data[ggml_idx] = torch_data[torch_idx];
                }
            }
        }
        
        LOG_INF("Converted result to ggml format: [%ld, %ld, %ld, %ld] (total elements: %ld)\n",
                result_ggml->ne[0], result_ggml->ne[1], result_ggml->ne[2], result_ggml->ne[3], 
                ggml_nelements(result_ggml));
        
        LOG_INF("PyTorch Flash Attention computation successful!\n");
        return result_ggml;
        
    } catch (const std::exception& e) {
        LOG_ERR("PyTorch Flash Attention failed: %s\n", e.what());
        LOG_ERR("   Falling back to ggml implementation...\n");
        // Fall through to ggml implementation
    }

    // Fallback to original ggml implementation
    LOG_INF("Using ggml Flash Attention implementation\n");
    struct ggml_cgraph * gf = build_flash_attn_graph(ctx, Q, K, V, mask, K_quant, V_quant, scale);
    
    int n_threads = 12;
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
        LOG_INF("Tensor %s: NULL\n", name.c_str());
        return;
    }
    LOG_INF("%s: shape=[%ld,%ld,%ld,%ld], type=%s, elements=%zu\n", 
            name.c_str(), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
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
    
    // step -> vector of (tensor, name)
    std::map<int, std::vector<std::pair<ggml_tensor*, std::string>>> step_tensor_map;
    for (ggml_tensor* tensor = ggml_get_first_tensor(tensor_ctx); tensor; tensor = ggml_get_next_tensor(tensor_ctx, tensor)) {
        std::string name = tensor->name && tensor->name[0] ? tensor->name : "unnamed";
        LOG_INF("Tensor: %s, shape: %s\n", name.c_str(), ggml_ne_string(tensor).c_str());
        int step = extract_step_from_name(name);
        step_tensor_map[step].emplace_back(tensor, name);
    }

    // Add space for result tensor (estimated)
    struct ggml_init_params ctx_params {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    // create context
    ggml_context* compute_ctx = ggml_init(ctx_params);

    // Output by step
    for (const auto& [step, tensors] : step_tensor_map) {
        LOG_INF("\n==== Step %d ====%s\n", step, (step == -1 ? " (unknown)" : ""));
        
        if (tensors.size() < 4) {
            LOG_INF("Insufficient tensors in step %d (need at least Q, K, V, mask)\n", step);
            continue;
        }
        
        ggml_tensor * kqv_out = tensors[0].first;
        ggml_tensor * Q = tensors[1].first;
        ggml_tensor * K = tensors[2].first;
        ggml_tensor * V = tensors[3].first;
        ggml_tensor * kq_mask = tensors.size() > 4 ? tensors[4].first : nullptr;

        LOG_INF("Tensors count: %zu\n", tensors.size());
        
        LOG_INF("Found tensors - Q: %s, K: %s, V: %s", Q->name, K->name, V->name);
        if (kq_mask) {
            LOG_INF(", Mask: %s", kq_mask->name);
        }
        LOG_INF("\n");
        
        ggml_tensor * K_quant = nullptr;
        ggml_tensor * V_quant = nullptr;
        if (tensors.size() > 5) {
            K_quant = tensors[5].first;
            V_quant = tensors[6].first;
            LOG_INF("Quantized tensors - K_quant: %s, V_quant: %s\n", 
                    K_quant->name, V_quant->name);
        }
        
        // Run flash attention for all steps
        LOG_INF("\nRunning Flash Attention at Step %d\n", step);
        
        // Print input tensor summary (without detailed data)
        print_tensor_summary(Q, "Q (Query)");
        print_tensor_summary(K, "K (Key)");
        print_tensor_summary(V, "V (Value)");
        if (kq_mask) {
            print_tensor_summary(kq_mask, "Mask");
        }
        if (K_quant && V_quant) {
            print_tensor_summary(K_quant, "K_quant");
            print_tensor_summary(V_quant, "V_quant");
        }
        
        // Compute flash attention
        float scale = 1.0f / sqrtf((float)Q->ne[0]); // 1 / sqrt(head_dim)
        
        struct ggml_tensor * flash_result = compute_flash_attn(compute_ctx, Q, K, V, kq_mask, K_quant, V_quant, scale);
        
        if (flash_result) {
            LOG_INF("Flash Attention computation successful!\n");
            ggml_print_tensor_info((uint8_t*)flash_result->data, flash_result->type, 
                                 flash_result->ne, flash_result->nb, "Flash Attention Result", 4);
            
            // Compare with original kqv_out if available
            if (kqv_out && kqv_out->data) {
                LOG_INF("Comparing with original kqv_out:\n");
                ggml_print_tensor_info((uint8_t*)kqv_out->data, kqv_out->type, 
                                     kqv_out->ne, kqv_out->nb, "Original KQV_OUT", 4);
                
                // Calculate difference if same size
                if (ggml_nelements(flash_result) == ggml_nelements(kqv_out) && 
                    flash_result->type == GGML_TYPE_F32 && kqv_out->type == GGML_TYPE_F32) {
                    
                    float* flash_data = (float*)flash_result->data;
                    float* orig_data  = (float*)kqv_out->data;
                    size_t n_elements = ggml_nelements(flash_result);
                    
                    double mse = 0.0;
                    double max_diff = 0.0;
                    for (size_t i = 0; i < n_elements; i++) {
                        double diff = fabs(flash_data[i] - orig_data[i]);
                        mse += diff * diff;
                        max_diff = std::max(max_diff, diff);
                    }
                    mse /= n_elements;
                    
                    LOG_INF("Difference Analysis: MSE: %.10f, Max Diff: %.10f, RMSE: %.10f\n", mse, max_diff, sqrt(mse));
                }
            }
        } else {
            LOG_ERR("Flash Attention computation failed!\n");
        }
        
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

    if (!read_kqv_tensors(params)) {
        return 1;
    }

    return 0;
} 




