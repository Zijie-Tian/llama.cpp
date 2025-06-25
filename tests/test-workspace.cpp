// g++ workspace_demo.cpp -I path/to/ggml -pthread -O2 -std=c++17
#include "ggml.h"
#include <vector>
#include <cstdio>

int main() {
    // 1️⃣ 先准备一块“常驻” tensor 内存（仅存元数据和权重）
    const size_t ctx_bytes = 16ull * 1024 * 1024;
    ggml_init_params p = { ctx_bytes, nullptr, /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(p);

    // 随便造两张输入矩阵
    auto * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 64);
    auto * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 128);

    // 2️⃣ 定义一块可复用的 scratch buffer（真正的共享 workspace）
    std::vector<uint8_t> scratch_buf(4 * 128 * 128);   // 64 KiB
    ggml_scratch wsp = { 0, scratch_buf.size(), scratch_buf.data() };

    /* ----------- 第一段算子：矩阵乘 ------------- */
    ggml_set_scratch(ctx, wsp);            // 开始在 scratch 上分配
    auto * Y = ggml_mul_mat(ctx, A, B);    // 结果 Y 驻留在 scratch

    /* ----------- 第二段算子：对 Y 做 GELU ------- */
    ggml_set_scratch(ctx, {0, 0, nullptr}); // 切回默认池；但 Y 仍指向同一块 scratch
    auto * Z = ggml_gelu(ctx, Y);           // Z 在主池，读取 Y 的数据

    /* 3️⃣ 构图并一次性规划临时 work_data（可选，不影响 scratch） */
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, Z);

    ggml_cplan plan = ggml_graph_plan(g, /*n_threads=*/4);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();          // 全图级别 workspace

    ggml_graph_compute(g, &plan);

    printf("Z[0] = %.4f\n", ((float *)Z->data)[0]);

    ggml_free(ctx);
    return 0;
}
