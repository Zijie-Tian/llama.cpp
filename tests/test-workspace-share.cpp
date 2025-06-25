#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdint>
#include <vector>

static void op_write_workspace(ggml_tensor * dst, int ith, int nth, void * wdata, size_t wsize, void * userdata) {
    (void) dst; (void) nth; (void) wsize; (void) userdata;
    if (ith != 0) return;
    int32_t value = *(int32_t *) userdata;
    int32_t * buf = (int32_t *) wdata;
    buf[0] = value;
}

static void op_read_workspace(ggml_tensor * dst, int ith, int nth, void * wdata, size_t wsize, void * userdata) {
    (void) nth; (void) wsize; (void) userdata;
    if (ith != 0) return;
    int32_t v = ((int32_t *) wdata)[0];
    ggml_set_i32(dst, v);
}

int main() {
    struct ggml_init_params params = { 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);

    int32_t val = 42;

    ggml_tensor * dummy_a = ggml_new_i32(ctx, 0);
    ggml_tensor * dummy_b = ggml_new_i32(ctx, 0);
    ggml_tensor * dummy_c = ggml_new_i32(ctx, 0);

    ggml_tensor * args_write[] = { dummy_a, dummy_b, dummy_c };
    ggml_tensor * write = ggml_custom_4d(ctx, GGML_TYPE_I32, 1,1,1,1, args_write, 3, op_write_workspace, 1, &val);

    ggml_tensor * args_read[] = { write, dummy_b, dummy_c };
    ggml_tensor * read = ggml_custom_4d(ctx, GGML_TYPE_I32, 1,1,1,1, args_read, 3, op_read_workspace, 1, nullptr);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, read);

    struct ggml_cplan cplan = ggml_graph_plan(gf, 1, nullptr);
    std::vector<uint8_t> work(cplan.work_size > sizeof(int32_t) ? cplan.work_size : sizeof(int32_t));
    cplan.work_data = work.data();

    ggml_graph_compute(gf, &cplan);

    int32_t result = ggml_get_i32_1d(read, 0);
    printf("workspace value: %d\n", result);

    ggml_free(ctx);

    return result == val ? 0 : 1;
}

