#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <memory>
#include <functional>

/**
 * Callback data structure for tracking kq_mask tensors
 */
struct kq_mask_trace_data {
    std::vector<uint8_t> temp_data;
    int step_count = 0;
    std::set<std::string> traced_tensors;
    std::unordered_map<std::string, int> tensor_counts;
    int target_layer = -1; // -1 means monitor all layers, >= 0 means monitor specific layer
    bool prefill_only = false; // whether to print only prefill phase
    bool is_prefill_phase = true; // track current phase
    bool print_mask = false; // whether to print mask visualization
};

static int extract_layer_number(const char* tensor_name) {
    if (!tensor_name) return -1;

    std::string name(tensor_name);

    // Look for kq_mask pattern with layer number
    // Try to find pattern like "kq_mask-N" or extract from names like "blk.N.attn_k"
    size_t kq_pos = name.find("kq_mask");
    if (kq_pos != std::string::npos) {
        size_t dash_pos = kq_pos + 7; // Position after "kq_mask"
        if (dash_pos < name.length() && name[dash_pos] == '-') {
            std::string layer_str = name.substr(dash_pos + 1);
            // Extract only the numeric part
            size_t end_pos = 0;
            while (end_pos < layer_str.length() && std::isdigit(layer_str[end_pos])) {
                end_pos++;
            }
            if (end_pos > 0) {
                try {
                    return std::stoi(layer_str.substr(0, end_pos));
                } catch (...) {
                    return -1;
                }
            }
        }
    }

    return -1;
}

/**
 * Determine mask type based on tensor name
 */
enum MaskType {
    MASK_UNKNOWN = 0,
    MASK_FP16 = 1,
    MASK_QUANTIZED = 2,
    MASK_GENERIC = 3
};

static MaskType get_mask_type(const char* tensor_name) {
    if (!tensor_name) return MASK_UNKNOWN;
    
    std::string name(tensor_name);
    
    // Check for explicit FP16 mask indicators
    if (name.find("kq_mask_fp16") != std::string::npos || 
        name.find("KQ_mask") != std::string::npos ||
        (name.find("kq_mask") != std::string::npos && name.find("quant") == std::string::npos)) {
        return MASK_FP16;
    }
    
    // Check for quantized mask indicators
    if (name.find("kq_mask_quant") != std::string::npos || 
        name.find("mask_quant") != std::string::npos) {
        return MASK_QUANTIZED;
    }
    
    // Generic kq_mask (could be either type)
    if (name.find("kq_mask") != std::string::npos) {
        return MASK_GENERIC;
    }
    
    return MASK_UNKNOWN;
}

static const char* mask_type_to_string(MaskType type) {
    switch (type) {
        case MASK_FP16: return "FP16";
        case MASK_QUANTIZED: return "QUANTIZED";
        case MASK_GENERIC: return "GENERIC";
        default: return "UNKNOWN";
    }
}

static bool is_kq_mask_tensor(const char* tensor_name) {
    return get_mask_type(tensor_name) != MASK_UNKNOWN;
}

static bool should_monitor_tensor(const char* tensor_name, int target_layer) {
    if (!is_kq_mask_tensor(tensor_name)) {
        return false;
    }

    if (target_layer == -1) {
        return true; // monitor all layers
    }

    int layer_num = extract_layer_number(tensor_name);
    return layer_num == target_layer;
}

static void print_tensor_shape_info(const struct ggml_tensor* tensor, const char* tensor_name) {
    if (!tensor) return;

    int layer_num = extract_layer_number(tensor_name);
    MaskType mask_type = get_mask_type(tensor_name);

    LOG("[KQ_MASK-TRACE] Layer %d - %s (%s): shape=[%ld,%ld,%ld,%ld] type=%s elements=%zu\n",
        layer_num >= 0 ? layer_num : -1,
        tensor_name ? tensor_name : "unknown",
        mask_type_to_string(mask_type),
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
        ggml_type_name(tensor->type), ggml_nelements(tensor));
}

/**
 * Print a visualization of the KQV attention mask.
 * Shows which tokens can attend to which other tokens.
 * x = can attend (0 or greater)
 * - = cannot attend (-INFINITY)
 */
static void print_kqv_mask(ggml_tensor* mask, const char* mask_name = nullptr) {
    if (!mask) {
        LOG("[KQ_MASK-TRACE] Mask tensor is null\n");
        return;
    }

    GGML_TENSOR_LOCALS(int64_t, ne_mask, mask, ne)
    GGML_TENSOR_LOCALS(int64_t, nb_mask, mask, nb)

    const char* display_name = mask_name ? mask_name : "Unknown";
    MaskType mask_type = get_mask_type(mask_name);
    
    LOG("\n=== KQV Attention Mask: %s (%s) ===\n", display_name, mask_type_to_string(mask_type));
    LOG("Mask shape: [%lld, %lld, %lld, %lld]\n", ne_mask0, ne_mask1, ne_mask2, ne_mask3);
    LOG("KV tokens →\n");

    // Print column header with numbers
    LOG("     ");
    for (int i = 0; i < ne_mask0 && i < 100; i++) { // Limit to 100 for readability
        LOG("%d", i % 10);
    }
    if (ne_mask0 > 100) {
        LOG("...");
    }
    LOG("\n");

    // Print separator
    LOG("     ");
    for (int i = 0; i < ne_mask0 && i < 100; i++) {
        LOG("=");
    }
    if (ne_mask0 > 100) {
        LOG("...");
    }
    LOG("\n");

    // Get mask data pointer
    const char* mask_nonfp32 = (const char*)mask->data;
    const float* mask_fp32 = (const float*)mask->data;

    ggml_to_float_t to_float = ggml_get_type_traits(mask->type)->to_float;

    float* row_buffer = (float*)malloc(ne_mask0 * sizeof(float));
    
    // Print each row of the mask (limit rows for readability)
    int max_rows = (ne_mask1 > 20) ? 20 : ne_mask1;
    for (int i = 0; i < max_rows; ++i) {
        LOG("%3d |", i); // Row number
        
        if (to_float) {
            to_float(mask_nonfp32 + i * nb_mask1, row_buffer, ne_mask0);
            
            int max_cols = (ne_mask0 > 100) ? 100 : ne_mask0;
            for (int j = 0; j < max_cols; ++j) {
                LOG("%c", (row_buffer[j] == 0.f) ? 'x' : '-');
            }
            if (ne_mask0 > 100) {
                LOG("...");
            }
        } else if (mask->type == GGML_TYPE_F32) {
            int max_cols = (ne_mask0 > 100) ? 100 : ne_mask0;
            for (int j = 0; j < max_cols; ++j) {
                LOG("%c", (mask_fp32[i * ne_mask0 + j] == 0.f) ? 'x' : '-');
            }
            if (ne_mask0 > 100) {
                LOG("...");
            }
        }
        LOG("\n");
    }
    
    if (ne_mask1 > 20) {
        LOG("... (%lld more rows)\n", ne_mask1 - 20);
    }

    free(row_buffer);
    LOG("\n");
}

/**
 * GGML operations callback during the graph execution.
 */
static bool ggml_debug_kq_mask_trace(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (kq_mask_trace_data *) user_data;

    // Skip if we're in decode phase and only prefill is requested
    if (cb_data->prefill_only && !cb_data->is_prefill_phase) {
        return true;
    }

    // Only process kq_mask tensors
    if (!should_monitor_tensor(t->name, cb_data->target_layer)) {
        return true;
    }

    // Check if we've already traced a tensor with the same name
    std::string tensor_name = t->name ? t->name : "unnamed";
    if (cb_data->traced_tensors.find(tensor_name) != cb_data->traced_tensors.end()) {
        return true;
    }
    cb_data->traced_tensors.insert(tensor_name);

    //> ===================================================================================================
    //> Traced target tensor.
    //> ===================================================================================================
    cb_data->step_count++;
    cb_data->tensor_counts[std::string(t->name)]++;

    MaskType mask_type = get_mask_type(t->name);
    
    LOG("\n=== KQ_MASK TENSOR DETECTED ===\n");
    LOG("%s: tensor_name=%s, type=%s\n", __func__, t->name, mask_type_to_string(mask_type));

    // Print tensor shape information
    print_tensor_shape_info(t, t->name);

    // Print source tensor information if available
    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (src0) {
        LOG("[KQ_MASK-TRACE] src[0]: %s, shape=[%ld,%ld,%ld,%ld], type=%s\n",
            src0->name ? src0->name : "unnamed",
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
            ggml_type_name(src0->type));
    }

    if (src1) {
        LOG("[KQ_MASK-TRACE] src[1]: %s, shape=[%ld,%ld,%ld,%ld], type=%s\n",
            src1->name ? src1->name : "unnamed",
            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
            ggml_type_name(src1->type));
    }

    // Print mask visualization if enabled
    if (cb_data->print_mask) {
        // copy the data from the GPU memory if needed
        const bool is_host = ggml_backend_buffer_is_host(t->buffer);

        if (!is_host) {
            auto n_bytes = ggml_nbytes(t);
            cb_data->temp_data.resize(n_bytes);
            ggml_backend_tensor_get(t, cb_data->temp_data.data(), 0, n_bytes);
            
            // Create a temporary tensor with CPU data to print
            ggml_tensor temp_tensor = *t;
            temp_tensor.data = cb_data->temp_data.data();
            print_kqv_mask(&temp_tensor, t->name);
        } else {
            print_kqv_mask(t, t->name);
        }
    }

    LOG("===============================\n\n");

    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    // Get the callback data pointer
    kq_mask_trace_data* cb_data = (kq_mask_trace_data*)params.cb_eval_user_data;

    // Process initial prompt (prefill phase)
    cb_data->is_prefill_phase = true;
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval initial prompt\n", __func__);
        return false;
    }

    // Reset traced tensors after initial prompt processing
    if (cb_data) {
        cb_data->traced_tensors.clear();
    }

    // Switch to decode phase
    cb_data->is_prefill_phase = false;

    // Generate tokens one by one
    for (int i = 0; i < params.n_predict; ++i) {
        LOG("\n\n>>>>>>>>>>>>>>>>>>>> GENERATION STEP %d/%d <<<<<<<<<<<<<<<<<<<\n\n", i + 1, params.n_predict);

        // Sample next token using simple greedy approach
        auto logits = llama_get_logits_ith(ctx, -1);
        auto n_vocab = llama_n_vocab(vocab);

        // Find token with highest probability (greedy sampling)
        llama_token new_token = 0;
        float max_logit = logits[0];
        for (llama_token token_id = 1; token_id < n_vocab; token_id++) {
            if (logits[token_id] > max_logit) {
                max_logit = logits[token_id];
                new_token = token_id;
            }
        }

        // Decode the new token
        if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
            LOG_ERR("%s : failed to eval token %d\n", __func__, i + 1);
            return false;
        }

        // Reset traced tensors after each token decode
        if (cb_data) {
            cb_data->traced_tensors.clear();
        }

        // Add to tokens for potential future use
        tokens.push_back(new_token);
    }

    return true;
}

int main(int argc, char ** argv) {
    kq_mask_trace_data cb_data;

    common_params params;

    // Add custom parameter parsing
    int target_layer = -1; // Default: monitor all layers
    bool prefill_only = false; // Default: print both prefill and decode
    bool print_mask = false; // Default: don't print mask

    // Create new argument list, excluding our custom parameters
    std::vector<char*> new_argv;
    new_argv.push_back(argv[0]); // Keep program name

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            target_layer = std::atoi(argv[i + 1]);
            i++; // Skip next parameter (layer number)
        } else if (strcmp(argv[i], "--prefill-only") == 0) {
            prefill_only = true;
        } else if (strcmp(argv[i], "--print-mask") == 0) {
            print_mask = true;
        } else {
            new_argv.push_back(argv[i]);    //> keep the original parameters
        }
    }

    cb_data.target_layer = target_layer;
    cb_data.prefill_only = prefill_only;
    cb_data.print_mask = print_mask;

    // Set parameters based on the command line arguments
    params.model.path               = "/Users/tianzijie/models/Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q2_K.gguf";
    params.prompt                   = "Hello, how are you?";
    params.n_gpu_layers             = 0;
    params.cache_type_k             = GGML_TYPE_Q4_0;  // Enable mixed cache to see both masks
    params.cache_type_v             = GGML_TYPE_Q4_0;  // Enable mixed cache to see both masks
    params.flash_attn               = true;
    params.n_predict                = 4;

    params.cpuparams.n_threads = 8;

    // Check for help first
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            LOG_ERR("Usage: %s [options] [--layer <layer_number>] [--prefill-only] [--print-mask]\n", argv[0]);
            LOG_ERR("Custom options:\n");
            LOG_ERR("  --layer <n>           Monitor only layer n (0-based). Use -1 or omit to monitor all layers.\n");
            LOG_ERR("  --prefill-only        Print only prefill phase, skip decode phase.\n");
            LOG_ERR("  --print-mask          Print kq_mask attention mask visualization.\n");
            LOG_ERR("Examples:\n");
            LOG_ERR("  %s -m model.gguf -p \"Hello\" --layer 0    # Monitor only layer 0\n", argv[0]);
            LOG_ERR("  %s -m model.gguf -p \"Hello\"              # Monitor all layers\n", argv[0]);
            LOG_ERR("  %s -m model.gguf -p \"Hello\" --prefill-only # Print only prefill phase\n", argv[0]);
            LOG_ERR("  %s -m model.gguf -p \"Hello\" --print-mask   # Print mask visualization\n", argv[0]);
            return 0;
        }
    }

    // Parse remaining parameters using common_params_parse
    if (!common_params_parse(new_argv.size(), new_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug_kq_mask_trace;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
