#include "llama.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "llama-kv-cache-mixed.h"
#include "llama-batch.h"
#include "common.h"

#include <memory>
#include <vector>
#include <random>
#include <cstdio>

// Utility: create a dummy model with one transformer layer and random parameters
static std::shared_ptr<llama_model> make_single_layer_model() {
    llama_model_params params = {};
    std::shared_ptr<llama_model> model = std::make_shared<llama_model>(params);

    model->hparams = llama_hparams();
    model->arch = LLM_ARCH_LLAMA;

    // minimal hyperparameters
    model->hparams.n_layer = 1;
    model->hparams.n_embd_head_k = 32;
    model->hparams.n_embd_head_v = 32;
    model->hparams.n_embd = 128;
    model->hparams.n_ctx_train = 128;
    model->hparams.rope_freq_base_train = 10000.0f;
    model->hparams.rope_freq_scale_train = 1.0f;

    auto & n_head_arr = model->hparams.n_head_arr;
    std::fill(n_head_arr.begin(), n_head_arr.end(), 4);
    auto & n_head_kv_arr = model->hparams.n_head_kv_arr;
    std::fill(n_head_kv_arr.begin(), n_head_kv_arr.end(), 4);
    auto & n_ff_arr = model->hparams.n_ff_arr;
    std::fill(n_ff_arr.begin(), n_ff_arr.end(), 512);

    // randomize a small embedding matrix just for demonstration
    ggml_init_params ctx_params = {16 * 1024 * 1024, nullptr, false};
    ggml_context * ctx = ggml_init(ctx_params);
    model->tok_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model->hparams.n_embd, 16);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> rnd(ggml_nelements(model->tok_embd));
    for (auto & v : rnd) v = dist(rng);
    ggml_backend_tensor_set(model->tok_embd, rnd.data(), 0, rnd.size() * sizeof(float));

    return model;
}

int main() {
    ggml_backend_load_all();

    auto model = make_single_layer_model();

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 64;
    ctx_params.n_batch = 8;
    ctx_params.use_mixed_kv_cache = true; // request mixed kv cache
    ctx_params.flash_attn = true;         // required when using mixed cache
    ctx_params.type_k = GGML_TYPE_F16;
    ctx_params.type_v = GGML_TYPE_F16;

    llama_context * ctx = llama_init_from_model(model.get(), ctx_params);
    if (!ctx) {
        fprintf(stderr, "failed to create llama_context\n");
        return 1;
    }

    llama_kv_cache * base_cache = llama_get_kv_self(ctx);
    auto * mixed_cache = dynamic_cast<llama_kv_cache_mixed*>(base_cache);
    if (!mixed_cache) {
        fprintf(stderr, "mixed kv cache not created\n");
        llama_free(ctx);
        return 1;
    }

    printf("Mixed KV cache created: size=%u, head=%u, used=%u\n",
           mixed_cache->get_size(), mixed_cache->get_head(), mixed_cache->get_used());

    llama_free(ctx);
    return 0;
}
