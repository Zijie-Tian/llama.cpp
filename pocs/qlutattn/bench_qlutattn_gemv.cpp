#include <float.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

// Include QLUTATTN headers
#include "../../ggml/src/ggml-cpu/qlutattn/qlutattn-config.h"
#include "../../ggml/src/ggml-cpu/qlutattn/qlut_ctor.h"
#include "../../ggml/src/ggml-cpu/qlutattn/qlutattn.h"
#include "../../ggml/src/ggml-cpu/qlutattn/tbl.h"

#ifdef __ARM_NEON
#    include <arm_neon.h>
typedef float16_t test_float_type;
#elif defined __AVX2__
#    include <immintrin.h>
typedef float test_float_type;
#else
typedef float test_float_type;
#endif

// ===================================================================================================
// Performance Benchmark Structure
// ===================================================================================================

struct BenchmarkResult {
    double              mean_ms;
    double              std_ms;
    double              min_ms;
    double              max_ms;
    double              throughput_gflops;
    std::vector<double> timings;
};

// Random number generator with fixed seed for reproducibility
static std::mt19937 g_rng(42);

static void fill_random_f32(float * data, size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < n; i++) {
        data[i] = dis(g_rng);
    }
}

static void fill_random_f16(ggml_fp16_t * data, size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < n; i++) {
        data[i] = ggml_fp32_to_fp16(dis(g_rng));
    }
}

// Quantization helper
static void quantize_symmetric(const float * input, uint8_t * output, float * scales, int M, int K, int bits,
                               int group_size) {
    const int max_int  = (1 << (bits - 1)) - 1;
    const int bias     = 1 << (bits - 1);
    const int n_groups = (M * K) / group_size;

    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end   = start + group_size;

        // Find max absolute value in group
        float max_abs = 0.0f;
        for (int i = start; i < end; i++) {
            float abs_val = fabsf(input[i]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }

        // Calculate scale
        scales[g] = max_abs / max_int;
        if (scales[g] < 1e-5f) {
            scales[g] = 1e-5f;
        }

        // Quantize values
        int elem_per_byte = 8 / bits;
        for (int i = 0; i < group_size; i++) {
            int idx   = start + i;
            int q_val = roundf(input[idx] / scales[g]);
            q_val     = std::max(-max_int, std::min(max_int, q_val));
            q_val += bias;  // Convert to unsigned

            // Pack into bytes
            int byte_idx   = (g * group_size + i) / elem_per_byte;
            int bit_offset = ((i % elem_per_byte) * bits);

            if (bit_offset == 0) {
                output[byte_idx] = 0;
            }
            output[byte_idx] |= (q_val << (8 - bits - bit_offset));
        }
    }
}

class QLUTATTNBenchmark {
  private:
    const int n_warmup     = 10;
    const int n_iterations = 100;

    void * aligned_malloc(size_t size) {
        void * ptr = nullptr;
        posix_memalign(&ptr, 64, size);
        return ptr;
    }

    void aligned_free(void * ptr) { free(ptr); }

  public:
    BenchmarkResult compute_statistics(const std::vector<double> & timings) {
        BenchmarkResult result;
        result.timings = timings;

        // Calculate mean
        result.mean_ms = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();

        // Calculate standard deviation
        double sq_sum = std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
        result.std_ms = std::sqrt(sq_sum / timings.size() - result.mean_ms * result.mean_ms);

        // Min and max
        result.min_ms = *std::min_element(timings.begin(), timings.end());
        result.max_ms = *std::max_element(timings.begin(), timings.end());

        return result;
    }

    void benchmark_qlutattn_gemv(int M, int K, int N, int bits) {
        printf("\n================================================================================\n");
        printf("Benchmarking QLUTATTN GeMV: M=%d, K=%d, N=%d, bits=%d\n", M, K, N, bits);
        printf("================================================================================\n");

        // Configuration based on bits
        const int group_size     = 128;
        const int act_group_size = 64;
        const int elem_per_byte  = 8 / bits;

        // Get kernel configuration
        // Initialize config system if needed
        if (!ggml_qlutattn_config_is_initialized()) {
            ggml_qlutattn_config_init();
        }
        const struct qlutattn_kernel_config * kernel_config = ggml_qlutattn_get_config(M, K, bits);
        if (!kernel_config) {
            printf("ERROR: Failed to get kernel config for M=%d, K=%d, bits=%d\n", M, K, bits);
            return;
        }

        // Allocate memory for weights and activations
        float *       weights_f32    = (float *) aligned_malloc(M * K * sizeof(float));
        float *       activation_f32 = (float *) aligned_malloc(N * K * sizeof(float));
        ggml_fp16_t * activation_f16 = (ggml_fp16_t *) aligned_malloc(N * K * sizeof(ggml_fp16_t));

        // Initialize with random data
        fill_random_f32(weights_f32, M * K);
        fill_random_f32(activation_f32, N * K);
        fill_random_f16(activation_f16, N * K);

        // Quantize weights
        size_t weight_size = (M * K) / elem_per_byte;
        size_t scale_size  = (M * K) / group_size;

        uint8_t *         quantized_weights = (uint8_t *) aligned_malloc(weight_size);
        float *           scales_f32        = (float *) aligned_malloc(scale_size * sizeof(float));
        test_float_type * scales            = (test_float_type *) aligned_malloc(scale_size * sizeof(test_float_type));

        // Perform quantization
        quantize_symmetric(weights_f32, quantized_weights, scales_f32, M, K, bits, group_size);

        // Convert scales to appropriate type
        for (size_t i = 0; i < scale_size; i++) {
#ifdef __ARM_NEON
            scales[i] = (float16_t) scales_f32[i];
#else
            scales[i] = scales_f32[i];
#endif
        }

        // Allocate LUT structures
        size_t qlut_size       = K * N * 16;  // 16 entries per group for g=4
        size_t lut_scales_size = K / act_group_size * N;
        size_t lut_biases_size = K / act_group_size * N;

        int8_t *          QLUT       = (int8_t *) aligned_malloc(qlut_size * sizeof(int8_t));
        test_float_type * LUT_Scales = (test_float_type *) aligned_malloc(lut_scales_size * sizeof(test_float_type));
        test_float_type * LUT_Biases = (test_float_type *) aligned_malloc(lut_biases_size * sizeof(test_float_type));

        // Allocate output
        float *           output_f32 = (float *) aligned_malloc(M * N * sizeof(float));
        test_float_type * output     = (test_float_type *) aligned_malloc(M * N * sizeof(test_float_type));

        // Initialize output to zero
        memset(output_f32, 0, M * N * sizeof(float));
        memset(output, 0, M * N * sizeof(test_float_type));

        // Warmup runs (including LUT construction)
        printf("Running %d warmup iterations (including LUT construction)...\n", n_warmup);
        for (int iter = 0; iter < n_warmup; iter++) {
            // Build LUT for each batch
            for (int n = 0; n < N; n++) {
                void * B          = activation_f16 + n * K;
                void * qlut       = QLUT + n * K * 16;
                void * lut_scales = LUT_Scales + n * K / act_group_size;
                void * lut_biases = LUT_Biases + n * K / act_group_size;

                ggml::cpu::qlutattn::qlutattn_lut_ctor_int8_g4(B, lut_scales, lut_biases, qlut, K, kernel_config);
            }

            // Reset output
            memset(output, 0, M * N * sizeof(test_float_type));

            // Call QGEMM for each output row
            ggml::cpu::qlutattn::qgemm_lut_int8_g4(quantized_weights, QLUT, scales, LUT_Scales, LUT_Biases, output,
                                                   kernel_config->bm, K, N, kernel_config);
        }

        // Benchmark runs (including LUT construction)
        printf("Running %d benchmark iterations (including LUT construction)...\n", n_iterations);
        std::vector<double> timings;

        for (int iter = 0; iter < n_iterations; iter++) {
            // Reset output
            memset(output, 0, M * N * sizeof(test_float_type));

            auto start = std::chrono::high_resolution_clock::now();

            // Build LUT for each batch (included in timing)
            for (int n = 0; n < N; n++) {
                void * B          = activation_f16 + n * K;
                void * qlut       = QLUT + n * K * 16;
                void * lut_scales = LUT_Scales + n * K / act_group_size;
                void * lut_biases = LUT_Biases + n * K / act_group_size;

                ggml::cpu::qlutattn::qlutattn_lut_ctor_int8_g4(B, lut_scales, lut_biases, qlut, K, kernel_config);
            }

            // Call QGEMM
            ggml::cpu::qlutattn::qgemm_lut_int8_g4(quantized_weights, QLUT, scales, LUT_Scales, LUT_Biases, output,
                                                   kernel_config->bm, K, N, kernel_config);

            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> elapsed = end - start;
            timings.push_back(elapsed.count());
        }

        // Calculate statistics
        BenchmarkResult result = compute_statistics(timings);

        // Calculate GFLOPS (2 * M * K * N for GEMV)
        double flops             = 2.0 * M * K * N;
        result.throughput_gflops = (flops / 1e9) / (result.mean_ms / 1000.0);

        // Compute reference result for accuracy check
        printf("Computing reference result...\n");
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += weights_f32[m * K + k] * activation_f32[n * K + k];
                }
                output_f32[m * N + n] = sum;
            }
        }

        // Convert output to float for comparison
        float * output_converted = (float *) aligned_malloc(M * N * sizeof(float));
        for (int i = 0; i < M * N; i++) {
#ifdef __ARM_NEON
            output_converted[i] = (float) output[i];
#else
            output_converted[i] = output[i];
#endif
        }

        // Calculate NMSE
        double nmse    = 0.0;
        double norm_sq = 0.0;
        for (int i = 0; i < M * N; i++) {
            double diff = output_converted[i] - output_f32[i];
            nmse += diff * diff;
            norm_sq += output_f32[i] * output_f32[i];
        }
        nmse = (norm_sq > 0) ? nmse / norm_sq : 0;

        // Print results
        printf("\n--- Results ---\n");
        printf("QLUTATTN (%d-bit):\n", bits);
        printf("  Mean:       %.3f ms\n", result.mean_ms);
        printf("  Std Dev:    %.3f ms\n", result.std_ms);
        printf("  Min:        %.3f ms\n", result.min_ms);
        printf("  Max:        %.3f ms\n", result.max_ms);
        printf("  Throughput: %.2f GFLOPS\n", result.throughput_gflops);
        printf("  NMSE:       %.8f\n", nmse);

        // Cleanup
        aligned_free(weights_f32);
        aligned_free(activation_f32);
        aligned_free(activation_f16);
        aligned_free(quantized_weights);
        aligned_free(scales_f32);
        aligned_free(scales);
        aligned_free(QLUT);
        aligned_free(LUT_Scales);
        aligned_free(LUT_Biases);
        aligned_free(output_f32);
        aligned_free(output);
        aligned_free(output_converted);
    }

    void run_benchmark_suite() {
        // Test different matrix sizes
        struct TestCase {
            int M, K, N, bits;
        };

        std::vector<TestCase> test_cases = {
            // Small matrices (for debugging)
            { 128,  128,  1, 2 },
            { 128,  128,  1, 4 },

            // Medium matrices
            { 256,  256,  1, 2 },
            { 256,  256,  1, 4 },
            { 512,  512,  1, 2 },
            { 512,  512,  1, 4 },

            // Large matrices (typical LLM sizes)
            { 1024, 1024, 1, 2 },
            { 1024, 1024, 1, 4 },
            { 2048, 2048, 1, 2 },
            { 2048, 2048, 1, 4 },

            // Batch processing
            { 1024, 1024, 4, 2 },
            { 1024, 1024, 4, 4 },
            { 512,  512,  8, 2 },
            { 512,  512,  8, 4 },
        };

        printf("\n");
        printf("================================================================================\n");
        printf("                     QLUTATTN GeMV Performance Benchmark                        \n");
        printf("================================================================================\n");
        printf("Configuration:\n");
        printf("  Warmup iterations:   %d\n", n_warmup);
        printf("  Timing iterations:   %d\n", n_iterations);
        printf("  Group size:          128\n");
        printf("  Activation group:    64\n");
        printf("  LUT group size (g):  4\n");
        printf("================================================================================\n");

        for (const auto & test : test_cases) {
            try {
                // Skip very large matrices if they might cause memory issues
                size_t memory_needed =
                    (size_t) test.M * test.K * sizeof(float) * 4 + (size_t) test.M * test.N * sizeof(float) * 2;
                if (memory_needed > 1ULL * 1024 * 1024 * 1024) {  // Skip if > 1GB
                    printf("\nSkipping M=%d, K=%d, N=%d (too large)\n", test.M, test.K, test.N);
                    continue;
                }

                benchmark_qlutattn_gemv(test.M, test.K, test.N, test.bits);
            } catch (const std::exception & e) {
                printf("Error in test case M=%d, K=%d, N=%d, bits=%d: %s\n", test.M, test.K, test.N, test.bits,
                       e.what());
            }
        }

        printf("\n");
        printf("================================================================================\n");
        printf("                           Benchmark Complete                                   \n");
        printf("================================================================================\n");
    }
};

int main(int argc, char ** argv) {
    try {
        QLUTATTNBenchmark benchmark;
        benchmark.run_benchmark_suite();
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    }
}
