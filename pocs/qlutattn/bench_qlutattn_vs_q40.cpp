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
// Q4_0 structures from ggml
// ===================================================================================================
#define QK4_0 32

typedef struct {
    ggml_fp16_t d;              // delta
    uint8_t     qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;

static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK8_0 32

typedef struct {
    float  d;          // delta
    float  s;          // d * sum(qs[i])
    int8_t qs[QK8_0];  // quants
} block_q8_0;

static_assert(sizeof(block_q8_0) == 2 * sizeof(float) + QK8_0, "wrong q8_0 block size/padding");

// ===================================================================================================
// Performance Benchmark Structure
// ===================================================================================================
struct BenchmarkResult {
    double              mean_ms;
    double              std_ms;
    double              min_ms;
    double              max_ms;
    double              throughput_gflops;
    double              throughput_gbps;  // GB/s for memory bandwidth
    std::vector<double> timings;
    double              accuracy_nmse;
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

// Quantization helper for QLUTATTN
static void quantize_symmetric(const float * input, uint8_t * output, float * scales, int M, int K, int bits,
                               int group_size) {
    const int max_int  = (1 << (bits - 1)) - 1;
    const int bias     = 1 << (bits - 1);
    const int n_groups = (M * K) / group_size;

    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end   = start + group_size;

        float max_abs = 0.0f;
        for (int i = start; i < end; i++) {
            float abs_val = fabsf(input[i]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }

        scales[g] = max_abs / max_int;
        if (scales[g] < 1e-5f) {
            scales[g] = 1e-5f;
        }

        int elem_per_byte = 8 / bits;
        for (int i = 0; i < group_size; i++) {
            int idx   = start + i;
            int q_val = roundf(input[idx] / scales[g]);
            q_val     = std::max(-max_int, std::min(max_int, q_val));
            q_val += bias;

            int byte_idx   = (g * group_size + i) / elem_per_byte;
            int bit_offset = ((i % elem_per_byte) * bits);

            if (bit_offset == 0) {
                output[byte_idx] = 0;
            }
            output[byte_idx] |= (q_val << (8 - bits - bit_offset));
        }
    }
}

// Quantize to Q4_0 format
static void quantize_q4_0(const float * src, block_q4_0 * dst, int64_t n) {
    assert(n % QK4_0 == 0);
    const int nb = n / QK4_0;

    for (int j = 0; j < nb; j++) {
        float amax = 0.0f;
        float vmax = 0.0f;

        for (int i = 0; i < QK4_0; i++) {
            const float v = src[j * QK4_0 + i];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                vmax = v;
            }
        }

        const float d  = vmax / -8;
        const float id = d ? 1.0f / d : 0.0f;

        dst[j].d = ggml_fp32_to_fp16(d);

        for (int i = 0; i < QK4_0 / 2; ++i) {
            const float x0 = src[j * QK4_0 + i * 2 + 0] * id;
            const float x1 = src[j * QK4_0 + i * 2 + 1] * id;

            const uint8_t xi0 = std::min(15, (int) (x0 + 8.5f));
            const uint8_t xi1 = std::min(15, (int) (x1 + 8.5f));

            dst[j].qs[i] = xi0 | (xi1 << 4);
        }
    }
}

// Quantize to Q8_0 format for activation
static void quantize_q8_0(const float * src, block_q8_0 * dst, int64_t n) {
    assert(n % QK8_0 == 0);
    const int nb = n / QK8_0;

    for (int j = 0; j < nb; j++) {
        float amax = 0.0f;
        for (int i = 0; i < QK8_0; i++) {
            const float v = src[j * QK8_0 + i];
            amax          = std::max(amax, fabsf(v));
        }

        const float d  = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;

        dst[j].d = d;

        int sum = 0;
        for (int i = 0; i < QK8_0; i++) {
            const int v  = roundf(src[j * QK8_0 + i] * id);
            dst[j].qs[i] = std::max(-128, std::min(127, v));
            sum += dst[j].qs[i];
        }
        dst[j].s = d * sum;
    }
}

class ComparativeBenchmark {
  private:
    const int n_warmup     = 10;
    const int n_iterations = 100;

    void * aligned_malloc(size_t size) {
        void * ptr = nullptr;
        posix_memalign(&ptr, 64, size);
        return ptr;
    }

    void aligned_free(void * ptr) { free(ptr); }

    BenchmarkResult compute_statistics(const std::vector<double> & timings) {
        BenchmarkResult result;
        result.timings = timings;

        result.mean_ms = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();

        double sq_sum = std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
        result.std_ms = std::sqrt(sq_sum / timings.size() - result.mean_ms * result.mean_ms);

        result.min_ms = *std::min_element(timings.begin(), timings.end());
        result.max_ms = *std::max_element(timings.begin(), timings.end());

        return result;
    }

  public:
    void benchmark_comparison(int M, int K, int N = 1) {
        printf("\n================================================================================\n");
        printf("Comparative Benchmark: M=%d, K=%d, N=%d\n", M, K, N);
        printf("================================================================================\n");

        // Allocate memory for weights and activations
        float * weights_f32    = (float *) aligned_malloc(M * K * sizeof(float));
        float * activation_f32 = (float *) aligned_malloc(N * K * sizeof(float));
        float * output_ref     = (float *) aligned_malloc(M * N * sizeof(float));

        // Initialize with random data
        fill_random_f32(weights_f32, M * K);
        fill_random_f32(activation_f32, N * K);

        // Compute reference result (FP32)
        memset(output_ref, 0, M * N * sizeof(float));
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += weights_f32[m * K + k] * activation_f32[n * K + k];
                }
                output_ref[m * N + n] = sum;
            }
        }

        // Variables to store timings for comparison
        double mean_time_qlutattn = 0, mean_time_q40 = 0, mean_time_fp32 = 0;

        // ===================================================================================================
        // Benchmark 1: QLUTATTN 4-bit
        // ===================================================================================================
        {
            printf("\n--- QLUTATTN 4-bit ---\n");

            const int bits           = 4;
            const int group_size     = 128;
            const int act_group_size = 64;
            const int elem_per_byte  = 8 / bits;

            // Initialize config system if needed
            if (!ggml_qlutattn_config_is_initialized()) {
                ggml_qlutattn_config_init();
            }
            const struct qlutattn_kernel_config * kernel_config = ggml_qlutattn_get_config(M, K, bits);
            if (!kernel_config) {
                printf("ERROR: Failed to get kernel config\n");
                goto benchmark_q40;
            }

            ggml_fp16_t * activation_f16 = (ggml_fp16_t *) aligned_malloc(N * K * sizeof(ggml_fp16_t));
            fill_random_f16(activation_f16, N * K);

            size_t weight_size = (M * K) / elem_per_byte;
            size_t scale_size  = (M * K) / group_size;

            uint8_t *         quantized_weights = (uint8_t *) aligned_malloc(weight_size);
            float *           scales_f32        = (float *) aligned_malloc(scale_size * sizeof(float));
            test_float_type * scales = (test_float_type *) aligned_malloc(scale_size * sizeof(test_float_type));

            quantize_symmetric(weights_f32, quantized_weights, scales_f32, M, K, bits, group_size);

            for (size_t i = 0; i < scale_size; i++) {
#ifdef __ARM_NEON
                scales[i] = (float16_t) scales_f32[i];
#else
                scales[i] = scales_f32[i];
#endif
            }

            size_t qlut_size       = K * N * 16;
            size_t lut_scales_size = K / act_group_size * N;
            size_t lut_biases_size = K / act_group_size * N;

            int8_t *          QLUT = (int8_t *) aligned_malloc(qlut_size * sizeof(int8_t));
            test_float_type * LUT_Scales =
                (test_float_type *) aligned_malloc(lut_scales_size * sizeof(test_float_type));
            test_float_type * LUT_Biases =
                (test_float_type *) aligned_malloc(lut_biases_size * sizeof(test_float_type));
            test_float_type * output = (test_float_type *) aligned_malloc(M * N * sizeof(test_float_type));

            // Warmup (including LUT construction)
            for (int iter = 0; iter < n_warmup; iter++) {
                // Build LUT
                for (int n = 0; n < N; n++) {
                    void * B          = activation_f16 + n * K;
                    void * qlut       = QLUT + n * K * 16;
                    void * lut_scales = LUT_Scales + n * K / act_group_size;
                    void * lut_biases = LUT_Biases + n * K / act_group_size;

                    ggml::cpu::qlutattn::qlutattn_lut_ctor_int8_g4(B, lut_scales, lut_biases, qlut, K, kernel_config);
                }

                memset(output, 0, M * N * sizeof(test_float_type));
                ggml::cpu::qlutattn::qgemm_lut_int8_g4(quantized_weights, QLUT, scales, LUT_Scales, LUT_Biases, output,
                                                       kernel_config->bm, K, N, kernel_config);
            }

            // Benchmark (including LUT construction)
            std::vector<double> timings;
            for (int iter = 0; iter < n_iterations; iter++) {
                memset(output, 0, M * N * sizeof(test_float_type));

                auto start = std::chrono::high_resolution_clock::now();

                // Build LUT (included in timing)
                for (int n = 0; n < N; n++) {
                    void * B          = activation_f16 + n * K;
                    void * qlut       = QLUT + n * K * 16;
                    void * lut_scales = LUT_Scales + n * K / act_group_size;
                    void * lut_biases = LUT_Biases + n * K / act_group_size;

                    ggml::cpu::qlutattn::qlutattn_lut_ctor_int8_g4(B, lut_scales, lut_biases, qlut, K, kernel_config);
                }

                // Perform matrix multiplication
                ggml::cpu::qlutattn::qgemm_lut_int8_g4(quantized_weights, QLUT, scales, LUT_Scales, LUT_Biases, output,
                                                       kernel_config->bm, K, N, kernel_config);

                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double, std::micro> elapsed = end - start;
                timings.push_back(elapsed.count());
            }

            BenchmarkResult result    = compute_statistics(timings);
            double          mean_time = result.mean_ms;
            double          std_dev   = result.std_ms;
            double          min_time  = result.min_ms;
            double          max_time  = result.max_ms;

            // Calculate accuracy
            float * output_f32 = (float *) aligned_malloc(M * N * sizeof(float));
            for (int i = 0; i < M * N; i++) {
#ifdef __ARM_NEON
                output_f32[i] = (float) output[i];
#else
                output_f32[i] = output[i];
#endif
            }

            double nmse = 0.0, norm_sq = 0.0;
            for (int i = 0; i < M * N; i++) {
                double diff = output_f32[i] - output_ref[i];
                nmse += diff * diff;
                norm_sq += output_ref[i] * output_ref[i];
            }
            result.accuracy_nmse = (norm_sq > 0) ? nmse / norm_sq : 0;

            printf("  Latency:      %.1f us (±%.1f us)\n", mean_time, std_dev);
            printf("  Min/Max:      %.1f / %.1f us\n", min_time, max_time);
            printf("  NMSE:         %.8f\n", result.accuracy_nmse);

            // Store for comparison
            mean_time_qlutattn = mean_time;

            aligned_free(activation_f16);
            aligned_free(quantized_weights);
            aligned_free(scales_f32);
            aligned_free(scales);
            aligned_free(QLUT);
            aligned_free(LUT_Scales);
            aligned_free(LUT_Biases);
            aligned_free(output);
            aligned_free(output_f32);
        }

// ===================================================================================================
// Benchmark 2: Standard Q4_0
// ===================================================================================================
benchmark_q40:
        {
            printf("\n--- Standard Q4_0 (with Q8_0 activation quantization) ---\n");

            // Quantize weights to Q4_0
            int          nb_weights = (M * K) / QK4_0;
            block_q4_0 * q4_weights = (block_q4_0 *) aligned_malloc(nb_weights * sizeof(block_q4_0));
            quantize_q4_0(weights_f32, q4_weights, M * K);

            // Allocate Q8_0 activation buffer
            int          nb_activation = (K) / QK8_0;  // per activation vector
            block_q8_0 * q8_activation = (block_q8_0 *) aligned_malloc(N * nb_activation * sizeof(block_q8_0));

            // Get Q4_0 vec_dot function
            const auto * funcs = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0);

            float * output_q40 = (float *) aligned_malloc(M * N * sizeof(float));

            // Warmup (including activation quantization)
            for (int iter = 0; iter < n_warmup; iter++) {
                // Quantize activations for all N vectors
                for (int n = 0; n < N; n++) {
                    quantize_q8_0(activation_f32 + n * K, q8_activation + n * nb_activation, K);
                }

                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        funcs->vec_dot(K, &output_q40[m * N + n], 0, q4_weights + m * K / QK4_0, 0,
                                       q8_activation + n * nb_activation, 0, 1);
                    }
                }
            }

            // Benchmark (including activation quantization)
            std::vector<double> timings;
            for (int iter = 0; iter < n_iterations; iter++) {
                auto start = std::chrono::high_resolution_clock::now();

                // Quantize activations (included in timing)
                for (int n = 0; n < N; n++) {
                    quantize_q8_0(activation_f32 + n * K, q8_activation + n * nb_activation, K);
                }

                // Perform matrix multiplication
                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        funcs->vec_dot(K, &output_q40[m * N + n], 0, q4_weights + m * K / QK4_0, 0,
                                       q8_activation + n * nb_activation, 0, 1);
                    }
                }

                auto                                      end     = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::micro> elapsed = end - start;
                timings.push_back(elapsed.count());
            }

            BenchmarkResult result_q40    = compute_statistics(timings);
            double          mean_time_q40 = result_q40.mean_ms;
            double          std_dev_q40   = result_q40.std_ms;
            double          min_time_q40  = result_q40.min_ms;
            double          max_time_q40  = result_q40.max_ms;

            // Calculate accuracy
            double nmse = 0.0, norm_sq = 0.0;
            for (int i = 0; i < M * N; i++) {
                double diff = output_q40[i] - output_ref[i];
                nmse += diff * diff;
                norm_sq += output_ref[i] * output_ref[i];
            }
            result_q40.accuracy_nmse = (norm_sq > 0) ? nmse / norm_sq : 0;

            printf("  Latency:      %.1f us (±%.1f us)\n", mean_time_q40, std_dev_q40);
            printf("  Min/Max:      %.1f / %.1f us\n", min_time_q40, max_time_q40);
            printf("  NMSE:         %.8f\n", result_q40.accuracy_nmse);

            // Store for comparison
            mean_time_q40 = mean_time_q40;

            aligned_free(q4_weights);
            aligned_free(q8_activation);
            aligned_free(output_q40);
        }

        // ===================================================================================================
        // Benchmark 3: FP32 Reference
        // ===================================================================================================
        {
            printf("\n--- FP32 Reference ---\n");

            float * output_fp32 = (float *) aligned_malloc(M * N * sizeof(float));

            // Warmup
            for (int iter = 0; iter < n_warmup; iter++) {
                memset(output_fp32, 0, M * N * sizeof(float));
                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; k++) {
                            sum += weights_f32[m * K + k] * activation_f32[n * K + k];
                        }
                        output_fp32[m * N + n] = sum;
                    }
                }
            }

            // Benchmark
            std::vector<double> timings;
            for (int iter = 0; iter < n_iterations; iter++) {
                memset(output_fp32, 0, M * N * sizeof(float));

                auto start = std::chrono::high_resolution_clock::now();

                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; k++) {
                            sum += weights_f32[m * K + k] * activation_f32[n * K + k];
                        }
                        output_fp32[m * N + n] = sum;
                    }
                }

                auto                                      end     = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::micro> elapsed = end - start;
                timings.push_back(elapsed.count());
            }

            BenchmarkResult result_fp32    = compute_statistics(timings);
            double          mean_time_fp32 = result_fp32.mean_ms;
            double          std_dev_fp32   = result_fp32.std_ms;
            double          min_time_fp32  = result_fp32.min_ms;
            double          max_time_fp32  = result_fp32.max_ms;

            printf("  Latency:      %.1f us (±%.1f us)\n", mean_time_fp32, std_dev_fp32);
            printf("  Min/Max:      %.1f / %.1f us\n", min_time_fp32, max_time_fp32);
            printf("  NMSE:         0.00000000 (reference)\n");

            // Store for comparison
            mean_time_fp32 = mean_time_fp32;

            aligned_free(output_fp32);

            // Print speedup comparisons at the end
            if (mean_time_qlutattn > 0 && mean_time_q40 > 0 && mean_time_fp32 > 0) {
                printf("\n--- Performance Comparison ---\n");
                printf("  QLUTATTN speedup vs FP32: %.2fx\n", mean_time_fp32 / mean_time_qlutattn);
                printf("  QLUTATTN speedup vs Q4_0: %.2fx\n", mean_time_q40 / mean_time_qlutattn);
                printf("  Q4_0 speedup vs FP32:     %.2fx\n", mean_time_fp32 / mean_time_q40);
            }
        }

        // Cleanup
        aligned_free(weights_f32);
        aligned_free(activation_f32);
        aligned_free(output_ref);
    }

    void run_benchmark_suite() {
        printf("\n");
        printf("================================================================================\n");
        printf("           QLUTATTN vs Q4_0 vs FP32 Performance Comparison                      \n");
        printf("================================================================================\n");
        printf("Configuration:\n");
        printf("  Warmup iterations:   %d\n", n_warmup);
        printf("  Timing iterations:   %d\n", n_iterations);
        printf("  QLUTATTN group size: 128\n");
        printf("  Q4_0 block size:     32\n");
        printf("================================================================================\n");

        // Test different matrix sizes
        struct TestCase {
            int M, K, N;
        };

        std::vector<TestCase> test_cases = {
            // Small matrices
            { 128,  128,  1 },
            { 256,  256,  1 },

            // Medium matrices
            { 512,  512,  1 },
            { 1024, 1024, 1 },

            // Large matrices (typical LLM sizes)
            { 2048, 2048, 1 },
            { 4096, 4096, 1 },

            // Batch processing (if memory allows)
            { 1024, 1024, 4 },
            { 512,  512,  8 },
        };

        for (const auto & test : test_cases) {
            try {
                // Check memory requirements
                size_t memory_needed =
                    (size_t) test.M * test.K * sizeof(float) * 4 + (size_t) test.M * test.N * sizeof(float) * 4;
                if (memory_needed > 2ULL * 1024 * 1024 * 1024) {
                    printf("\nSkipping M=%d, K=%d, N=%d (requires >2GB memory)\n", test.M, test.K, test.N);
                    continue;
                }

                benchmark_comparison(test.M, test.K, test.N);
            } catch (const std::exception & e) {
                printf("Error in test case M=%d, K=%d, N=%d: %s\n", test.M, test.K, test.N, e.what());
            }
        }

        printf("\n");
        printf("================================================================================\n");
        printf("                           Benchmark Complete                                   \n");
        printf("================================================================================\n");

        // Print summary
        printf("\nSummary:\n");
        printf("- QLUTATTN uses LUT-based computation with 4-bit quantization (including LUT construction)\n");
        printf("- Q4_0 uses standard GGML 4-bit quantization with group size 32\n");
        printf("- FP32 is the unquantized reference implementation\n");
        printf("- Latency is measured in microseconds (us), lower is better\n");
        printf("- NMSE measures accuracy loss compared to FP32\n");
        printf("- Speedup shows performance improvement relative to other methods\n");
    }
};

int main(int argc, char ** argv) {
    try {
        ComparativeBenchmark benchmark;
        benchmark.run_benchmark_suite();
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    }
}
