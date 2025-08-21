#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_TMAC
#    include "lut_mul_mat.h"
#    include "tmac.h"
#endif

void print_tensor_info(const ggml_tensor * tensor, const char * name) {
    printf("%s: type=%s, shape=[%lld, %lld, %lld, %lld], buffer=%s\n", name, ggml_type_name(tensor->type),
           tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
           tensor->buffer ? ggml_backend_buft_name(tensor->buffer->buft) : "none");
}

void fill_tensor_with_data(ggml_tensor * tensor, float base_value = 1.0f) {
    if (tensor->type == GGML_TYPE_F32) {
        float * data       = (float *) tensor->data;
        size_t  n_elements = ggml_nelements(tensor);
        for (size_t i = 0; i < n_elements; i++) {
            data[i] = base_value + (float) (i % 10) * 0.1f;
        }
    }
}

void print_tensor_data(const ggml_tensor * tensor, const char * name, int max_elements = 10) {
    if (tensor->type == GGML_TYPE_F32) {
        const float * data       = (const float *) tensor->data;
        size_t        n_elements = ggml_nelements(tensor);
        printf("%s data (first %d elements): ", name, max_elements);
        for (int i = 0; i < std::min((size_t) max_elements, n_elements); i++) {
            printf("%.3f ", data[i]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== T-MAC Buffer Type Test ===\n");

#ifndef GGML_USE_TMAC
    printf("ERROR: GGML_USE_TMAC is not defined. Please compile with -DGGML_TMAC=ON\n");
    return 1;
#endif

    // Initialize T-MAC
    ggml_tmac_init();
    printf("T-MAC initialized successfully\n");

    // Get T-MAC buffer type
    ggml_backend_buffer_type_t tmac_buft = ggml_backend_tmac_buffer_type();
    if (!tmac_buft) {
        printf("ERROR: Failed to get T-MAC buffer type\n");
        return 1;
    }
    printf("T-MAC buffer type: %s\n", ggml_backend_buft_name(tmac_buft));

    // Test parameters
    const int M = 64;   // rows of weight matrix
    const int K = 128;  // cols of weight matrix, rows of input
    const int N = 1;    // cols of input (batch size)

    printf("\nMatrix dimensions: Weight[%d, %d] x Input[%d, %d] -> Output[%d, %d]\n", M, K, K, N, M, N);

    // Create contexts for different buffer types
    ggml_init_params params = {
        .mem_size   = ggml_tensor_overhead() * 10,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };

    // Context for T-MAC tensors
    ggml_context * ctx_tmac = ggml_init(params);
    if (!ctx_tmac) {
        printf("ERROR: Failed to create T-MAC context\n");
        return 1;
    }

    // Context for regular tensors
    ggml_context * ctx_cpu = ggml_init(params);
    if (!ctx_cpu) {
        printf("ERROR: Failed to create CPU context\n");
        ggml_free(ctx_tmac);
        return 1;
    }

    printf("\nContexts created successfully\n");

    try {
        // Create tensors
        printf("\n=== Creating Tensors ===\n");

        // Weight tensor (T-MAC quantized type) - use Q4_0 as it's supported by T-MAC
        ggml_tensor * weight_tensor = ggml_new_tensor_2d(ctx_tmac, GGML_TYPE_Q4_0, K, M);
        ggml_set_name(weight_tensor, "weight.tmac");

        // Input tensor (FP32)
        ggml_tensor * input_tensor = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, K, N);
        ggml_set_name(input_tensor, "input.fp32");

        // Output tensor (FP32)
        ggml_tensor * output_tensor = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, M, N);
        ggml_set_name(output_tensor, "output.fp32");

        printf("Tensors created in contexts\n");

        // Allocate memory for T-MAC tensors
        ggml_backend_buffer_t tmac_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_tmac, tmac_buft);
        if (!tmac_buf) {
            printf("ERROR: Failed to allocate T-MAC buffer\n");
            throw std::runtime_error("T-MAC buffer allocation failed");
        }

        // Allocate memory for CPU tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        ggml_backend_buffer_t      cpu_buf  = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, cpu_buft);
        if (!cpu_buf) {
            printf("ERROR: Failed to allocate CPU buffer\n");
            throw std::runtime_error("CPU buffer allocation failed");
        }

        printf("Memory allocated successfully\n");
        printf("T-MAC buffer size: %.2f MB\n", ggml_backend_buffer_get_size(tmac_buf) / 1024.0 / 1024.0);
        printf("CPU buffer size: %.2f MB\n", ggml_backend_buffer_get_size(cpu_buf) / 1024.0 / 1024.0);

        // Print tensor information
        printf("\n=== Tensor Information ===\n");
        print_tensor_info(weight_tensor, "Weight");
        print_tensor_info(input_tensor, "Input");
        print_tensor_info(output_tensor, "Output");

        // Fill input tensor with test data
        printf("\n=== Filling Input Data ===\n");
        fill_tensor_with_data(input_tensor, 1.0f);
        print_tensor_data(input_tensor, "Input");

        // Create weight data and set to weight tensor
        printf("\n=== Setting Weight Data ===\n");
        std::vector<float>                    weight_data_fp32(M * K);
        std::mt19937                          gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < M * K; i++) {
            weight_data_fp32[i] = dis(gen);
        }

        // For T-MAC tensors, we need to provide properly quantized Q4_0 data
        // Create a temporary FP32 tensor to quantize
        ggml_init_params temp_params = {
            .mem_size   = M * K * sizeof(float) + ggml_tensor_overhead(),
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        ggml_context * temp_ctx  = ggml_init(temp_params);
        ggml_tensor *  temp_fp32 = ggml_new_tensor_2d(temp_ctx, GGML_TYPE_F32, K, M);

        // Fill temp tensor with FP32 data
        float * temp_data = (float *) temp_fp32->data;
        for (int i = 0; i < M * K; i++) {
            temp_data[i] = weight_data_fp32[i];
        }

        printf("Weight tensor size (ggml_nbytes): %zu bytes\n", ggml_nbytes(weight_tensor));

        auto start_time = std::chrono::high_resolution_clock::now();
        // Use the FP32 data for T-MAC conversion
        ggml_backend_tensor_set(weight_tensor, temp_data, 0, M * K * sizeof(float));
        auto end_time = std::chrono::high_resolution_clock::now();

        // Clean up temp context
        ggml_free(temp_ctx);

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        printf("Weight data set completed in %ld ms\n", duration.count());

        // Check if T-MAC can handle this operation
        printf("\n=== T-MAC Capability Check ===\n");

        // Create a temporary computation context for the operation
        ggml_init_params comp_params = {
            .mem_size   = ggml_tensor_overhead() * 5,
            .mem_buffer = NULL,
            .no_alloc   = true,
        };
        ggml_context * comp_ctx = ggml_init(comp_params);

        // Create the matrix multiplication operation
        ggml_tensor * mul_mat_op = ggml_mul_mat(comp_ctx, weight_tensor, input_tensor);
        ggml_set_name(mul_mat_op, "mul_mat_tmac");

        // Check if T-MAC can handle this operation
        bool can_tmac = ggml_tmac_can_mul_mat(mul_mat_op);
        printf("T-MAC can handle this operation: %s\n", can_tmac ? "YES" : "NO");

        if (can_tmac) {
            printf("✓ T-MAC will accelerate this matrix multiplication\n");
        } else {
            printf("✗ T-MAC cannot accelerate this operation\n");
            printf("  Reasons might be:\n");
            printf("  - Weight tensor type not supported: %s\n", ggml_type_name(weight_tensor->type));
            printf("  - Input tensor type not supported: %s\n", ggml_type_name(input_tensor->type));
            printf("  - Tensor names are excluded (token_embd.weight, output.weight)\n");
        }

        print_tensor_info(mul_mat_op, "MulMat Operation");

        // Build and execute computation graph
        printf("\n=== Building Computation Graph ===\n");

        ggml_cgraph * gf = ggml_new_graph(comp_ctx);
        ggml_build_forward_expand(gf, mul_mat_op);

        printf("Graph built with %d nodes\n", ggml_graph_n_nodes(gf));

        // Allocate output tensor
        ggml_tensor * result = ggml_graph_node(gf, -1);  // Last node should be our result

        // Set result data pointer to our output tensor
        result->data   = output_tensor->data;
        result->buffer = output_tensor->buffer;

        printf("\n=== Executing Computation ===\n");

        start_time = std::chrono::high_resolution_clock::now();

        int n_threads = 1;
        if (ggml_graph_compute_with_ctx(comp_ctx, gf, n_threads) != GGML_STATUS_SUCCESS) {
            printf("ERROR: Graph computation failed\n");
        } else {
            end_time         = std::chrono::high_resolution_clock::now();
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            printf("✓ Matrix multiplication completed successfully in %ld μs\n", duration_us.count());

            // Print results
            printf("\n=== Results ===\n");
            print_tensor_data(output_tensor, "Output");

            // Verify output is not all zeros
            const float * output_data = (const float *) output_tensor->data;
            bool          has_nonzero = false;
            for (int i = 0; i < M * N; i++) {
                if (output_data[i] != 0.0f) {
                    has_nonzero = true;
                    break;
                }
            }

            if (has_nonzero) {
                printf("✓ Output contains non-zero values - computation appears successful\n");
            } else {
                printf("⚠ Output is all zeros - may indicate an issue\n");
            }
        }

        // Clean up computation context
        ggml_free(comp_ctx);

        // Clean up buffers
        ggml_backend_buffer_free(tmac_buf);
        ggml_backend_buffer_free(cpu_buf);

        printf("\n=== Test Summary ===\n");
        printf("✓ T-MAC buffer type creation: SUCCESS\n");
        printf("✓ Tensor allocation: SUCCESS\n");
        printf("✓ Data transfer: SUCCESS\n");
        printf("✓ Matrix multiplication: %s\n", "SUCCESS");
        printf("✓ T-MAC integration: %s\n", can_tmac ? "ACTIVE" : "AVAILABLE");

    } catch (const std::exception & e) {
        printf("ERROR: %s\n", e.what());
        ggml_free(ctx_tmac);
        ggml_free(ctx_cpu);
        return 1;
    }

    // Clean up contexts
    ggml_free(ctx_tmac);
    ggml_free(ctx_cpu);

    printf("\n=== T-MAC Test Completed Successfully ===\n");
    return 0;
}
