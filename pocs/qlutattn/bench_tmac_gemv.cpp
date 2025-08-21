#define GGML_USE_TMAC 1
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
#include "ggml-cpu/tmac/lut_mul_mat.h"
#include "ggml-cpu/tmac/tmac.h"
#include "ggml.h"

#ifdef __ARM_NEON
#    include <arm_neon.h>
typedef float16_t tmac_float_type;
#elif defined __AVX2__
#    include <immintrin.h>
typedef float tmac_float_type;
#else
typedef float tmac_float_type;
#endif

// ===================================================================================================
// Utility functions from test_tmac_simple.cpp
// ===================================================================================================

typedef uint16_t ggml_half;

#define GGML_FP16_TO_FP32(x) ggml_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_fp32_to_fp16(x)

constexpr size_t kAllocAlignment = 64;

static void * aligned_malloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, kAllocAlignment);
#else
    void * ptr = nullptr;
    posix_memalign(&ptr, kAllocAlignment, size);
    return ptr;
#endif
}

static void aligned_free(void * ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Quantization block structures
#define QKLUTATTN_W2G128 128

typedef struct {
    ggml_half d;                         // scale
    ggml_half m;                         // min
    uint8_t   qs[QKLUTATTN_W2G128 / 4];  // 8-bit quants
} block_qlutattn_w2g128;

#define QKLUTATTN_W4G128 128

typedef struct {
    ggml_half d;                         // scale
    ggml_half m;                         // min
    uint8_t   qs[QKLUTATTN_W4G128 / 2];  // 8-bit quants
} block_qlutattn_w4g128;

// Random number generator with fixed seed for reproducibility
static std::mt19937 g_rng(42);

static void fill_tensor_f32(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    float *                               data       = (float *) dst->data;
    size_t                                n_elements = ggml_nelements(dst);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = dis(g_rng);
    }
}

static void set_tensor_f32(ggml_tensor * dst, float value) {
    float * data       = (float *) dst->data;
    size_t  n_elements = ggml_nelements(dst);
    for (size_t i = 0; i < n_elements; i++) {
        data[i] = value;
    }
}

// Quantization functions
static void pseudo_symmetric_quantize_f32(const float * input, int8_t * quantized, float * scales, float * zeros, int n,
                                          int n_bit, int q_group_size) {
    int num_groups;
    if (q_group_size > 0) {
        if (n % q_group_size != 0) {
            GGML_ASSERT(0);
        }
        num_groups = n / q_group_size;
    } else {
        num_groups   = 1;
        q_group_size = n;
    }

    const int max_int = (1 << (n_bit - 1)) - 1;
    const int min_int = -max_int;

    for (int g = 0; g < num_groups; ++g) {
        int start_idx = g * q_group_size;
        int end_idx   = start_idx + q_group_size;

        float max_abs_val = -FLT_MAX;

        for (int i = start_idx; i < end_idx; ++i) {
            float abs_val = fabsf(input[i]);
            if (abs_val > max_abs_val) {
                max_abs_val = abs_val;
            }
        }

        scales[g] = max_abs_val / max_int;
        zeros[g]  = 0.0f;

        for (int i = start_idx; i < end_idx; ++i) {
            int quantized_val = (int) roundf(input[i] / scales[g]);
            quantized_val     = quantized_val < min_int ? min_int : (quantized_val > max_int ? max_int : quantized_val);
            quantized[i]      = (int8_t) quantized_val;
        }
    }
}

void quantize_row_qlutattn_w4g128_pg_ref(block_qlutattn_w4g128 * GGML_RESTRICT y, const float * GGML_RESTRICT x,
                                         int64_t k) {
    const int qk             = QKLUTATTN_W4G128 / 2;
    const int nelem_per_byte = 128 / qk;
    assert(k % QKLUTATTN_W4G128 == 0);
    const int nb = k / 128;

    float  scale[nb];
    float  zero[nb];
    int8_t quantized[nb * 128];
    
    pseudo_symmetric_quantize_f32(x, quantized, scale, zero, k, 4, 128);

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < qk; j++) {
            const uint8_t x0 = (uint8_t) (quantized[i * 128 + j * nelem_per_byte + 0] + (1 << (4 - 1)));
            const uint8_t x1 = (uint8_t) (quantized[i * 128 + j * nelem_per_byte + 1] + (1 << (4 - 1)));

            y[i].qs[j] = (x0 << 4) | (x1 << 0);
        }

        y[i].d = GGML_FP32_TO_FP16(scale[i]);
        y[i].m = GGML_FP32_TO_FP16(zero[i]);
    }
}

void quantize_row_qlutattn_w2g128_pg_ref(block_qlutattn_w2g128 * GGML_RESTRICT y, const float * GGML_RESTRICT x,
                                         int64_t k) {
    const int qk             = QKLUTATTN_W2G128 / 4;
    const int nelem_per_byte = 128 / qk;
    assert(k % QKLUTATTN_W2G128 == 0);
    const int nb = k / 128;

    float  scale[nb];
    float  zero[nb];
    int8_t quantized[nb * 128];
    
    pseudo_symmetric_quantize_f32(x, quantized, scale, zero, k, 2, 128);

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < qk; j++) {
            const uint8_t x0 = (uint8_t) (quantized[i * 128 + j * nelem_per_byte + 0] + (1 << (2 - 1)));
            const uint8_t x1 = (uint8_t) (quantized[i * 128 + j * nelem_per_byte + 1] + (1 << (2 - 1)));
            const uint8_t x2 = (uint8_t) (quantized[i * 128 + j * nelem_per_byte + 2] + (1 << (2 - 1)));
            const uint8_t x3 = (uint8_t) (quantized[i * 128 + j * nelem_per_byte + 3] + (1 << (2 - 1)));

            y[i].qs[j] = (x0 << 6) | (x1 << 4) | (x2 << 2) | (x3 << 0);
        }

        y[i].d = GGML_FP32_TO_FP16(scale[i]);
        y[i].m = GGML_FP32_TO_FP16(zero[i]);
    }
}

// ===================================================================================================
// Performance Benchmark Structure
// ===================================================================================================

struct BenchmarkResult {
    double mean_ms;
    double std_ms;
    double min_ms;
    double max_ms;
    double throughput_gflops;
    std::vector<double> timings;
};

class TMACBenchmark {
private:
    ggml_context * main_ctx;
    ggml_context * tmac_ctx;
    ggml_backend_buffer_t tmac_buf;
    ggml_backend_buffer_type_t tmac_buft;
    
    // Configuration
    const int n_warmup = 10;
    const int n_iterations = 100;
    const int n_threads = 12;
    
public:
    TMACBenchmark() {
        // Initialize T-MAC
        ggml_tmac_init();
        printf("✓ T-MAC initialized successfully\n");

        // Get T-MAC buffer type
        tmac_buft = ggml_backend_tmac_buffer_type();
        if (!tmac_buft) {
            throw std::runtime_error("Failed to get T-MAC buffer type");
        }

        // Create contexts
        struct ggml_init_params main_params = {
            .mem_size   = 2ULL * 1024 * 1024 * 1024,  // 2GB
            .mem_buffer = NULL,
            .no_alloc   = false,
        };

        main_ctx = ggml_init(main_params);
        if (!main_ctx) {
            throw std::runtime_error("Failed to initialize main GGML context");
        }

        ggml_init_params tmac_params = {
            .mem_size   = 2ULL * 1024 * 1024 * 1024,  // 2GB
            .mem_buffer = NULL,
            .no_alloc   = true,
        };

        tmac_ctx = ggml_init(tmac_params);
        if (!tmac_ctx) {
            throw std::runtime_error("Failed to create T-MAC context");
        }
    }
    
    ~TMACBenchmark() {
        if (main_ctx) ggml_free(main_ctx);
        if (tmac_ctx) ggml_free(tmac_ctx);
        if (tmac_buf) ggml_backend_buffer_free(tmac_buf);
    }

    BenchmarkResult compute_statistics(const std::vector<double>& timings) {
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

    void benchmark_gemv_size(int M, int K, int N, int nbits) {
        printf("\n" "="*80 "\n");
        printf("Benchmarking TMAC GeMV: M=%d, K=%d, N=%d, bits=%d\n", M, K, N, nbits);
        printf("="*80 "\n");
        
        const int64_t group_size = 128;
        
        // Create and initialize tensors
        ggml_tensor * tensor_f32 = ggml_new_tensor_2d(main_ctx, GGML_TYPE_F32, K, M);
        fill_tensor_f32(tensor_f32);
        
        ggml_tensor * activation = ggml_new_tensor_2d(main_ctx, GGML_TYPE_F32, K, N);
        fill_tensor_f32(activation);
        
        // Prepare quantized weights based on bit width
        int nelem_per_byte = 8 / nbits;
        size_t weight_size = M * K / group_size * group_size / nelem_per_byte * sizeof(uint8_t) +
                            M * K / group_size * sizeof(float) * 2;
        uint8_t * tmac_qs = (uint8_t *) aligned_malloc(weight_size);
        
        if (nbits == 4) {
            block_qlutattn_w4g128 * qweights_block = 
                (block_qlutattn_w4g128 *) aligned_malloc(M * K / group_size * sizeof(block_qlutattn_w4g128));
            
            quantize_row_qlutattn_w4g128_pg_ref(qweights_block, (float *) tensor_f32->data, M * K);
            
            // Pack weights for T-MAC
            for (int i = 0; i < M * K / group_size; i++) {
                for (int j = 0; j < group_size / nelem_per_byte; j++) {
                    tmac_qs[i * (group_size / nelem_per_byte) + j] = qweights_block[i].qs[j];
                }
            }
            
            // Pack scales
            float * scale_ptr = (float *) (tmac_qs + M * K / group_size * (group_size / nelem_per_byte));
            for (int i = 0; i < M * K / group_size; i++) {
                scale_ptr[i] = GGML_FP16_TO_FP32(qweights_block[i].d);
            }
            
            // Pack zero points
            float * zp_ptr = (float *) (tmac_qs + M * K / group_size * (group_size / nelem_per_byte) + 
                                        M * K / group_size * sizeof(float));
            for (int i = 0; i < M * K / group_size; i++) {
                zp_ptr[i] = GGML_FP16_TO_FP32(qweights_block[i].m);
            }
            
            aligned_free(qweights_block);
        } else if (nbits == 2) {
            block_qlutattn_w2g128 * qweights_block = 
                (block_qlutattn_w2g128 *) aligned_malloc(M * K / group_size * sizeof(block_qlutattn_w2g128));
            
            quantize_row_qlutattn_w2g128_pg_ref(qweights_block, (float *) tensor_f32->data, M * K);
            
            // Pack weights for T-MAC
            for (int i = 0; i < M * K / group_size; i++) {
                for (int j = 0; j < group_size / nelem_per_byte; j++) {
                    tmac_qs[i * (group_size / nelem_per_byte) + j] = qweights_block[i].qs[j];
                }
            }
            
            // Pack scales
            float * scale_ptr = (float *) (tmac_qs + M * K / group_size * (group_size / nelem_per_byte));
            for (int i = 0; i < M * K / group_size; i++) {
                scale_ptr[i] = GGML_FP16_TO_FP32(qweights_block[i].d);
            }
            
            // Pack zero points
            float * zp_ptr = (float *) (tmac_qs + M * K / group_size * (group_size / nelem_per_byte) + 
                                        M * K / group_size * sizeof(float));
            for (int i = 0; i < M * K / group_size; i++) {
                zp_ptr[i] = GGML_FP16_TO_FP32(qweights_block[i].m);
            }
            
            aligned_free(qweights_block);
        }
        
        // Create T-MAC tensor
        ggml_type tmac_type = (nbits == 4) ? GGML_TYPE_TMAC_W4G128_1 : GGML_TYPE_TMAC_W2G128_1;
        ggml_tensor * tensor = ggml_new_tensor_2d(tmac_ctx, tmac_type, K, M);
        ggml_set_name(tensor, "tmac_tensor");
        
        // Allocate T-MAC buffer
        ggml_backend_buffer_t tmac_buf = ggml_backend_alloc_ctx_tensors_from_buft(tmac_ctx, tmac_buft);
        if (!tmac_buf) {
            throw std::runtime_error("Failed to allocate T-MAC buffer");
        }
        
        // Convert weights to T-MAC format
        ggml_backend_tmac_convert_weight(tensor, tmac_qs, 0, ggml_backend_buft_get_alloc_size(tmac_buft, tensor));
        
        // Create computation graphs
        struct ggml_cgraph * gf_f32 = ggml_new_graph(main_ctx);
        ggml_tensor * mul_f32 = ggml_mul_mat(main_ctx, tensor_f32, activation);
        ggml_build_forward_expand(gf_f32, mul_f32);
        
        struct ggml_cgraph * gf_tmac = ggml_new_graph(tmac_ctx);
        ggml_tensor * mul_tmac = ggml_mul_mat(tmac_ctx, tensor, activation);
        ggml_build_forward_expand(gf_tmac, mul_tmac);
        
        ggml_backend_buffer_t tmac_buf2 = ggml_backend_alloc_ctx_tensors_from_buft(tmac_ctx, tmac_buft);
        if (!tmac_buf2) {
            throw std::runtime_error("Failed to allocate T-MAC buffer for output");
        }
        
        // Warmup runs
        printf("Running %d warmup iterations...\n", n_warmup);
        for (int i = 0; i < n_warmup; i++) {
            ggml_graph_compute_with_ctx(main_ctx, gf_f32, n_threads);
            ggml_graph_compute_with_ctx(tmac_ctx, gf_tmac, n_threads);
        }
        
        // Benchmark FP32
        printf("Benchmarking FP32 baseline...\n");
        std::vector<double> timings_f32;
        for (int i = 0; i < n_iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            ggml_graph_compute_with_ctx(main_ctx, gf_f32, n_threads);
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> elapsed = end - start;
            timings_f32.push_back(elapsed.count());
        }
        
        // Benchmark T-MAC
        printf("Benchmarking T-MAC...\n");
        std::vector<double> timings_tmac;
        for (int i = 0; i < n_iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            ggml_graph_compute_with_ctx(tmac_ctx, gf_tmac, n_threads);
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> elapsed = end - start;
            timings_tmac.push_back(elapsed.count());
        }
        
        // Calculate statistics
        BenchmarkResult result_f32 = compute_statistics(timings_f32);
        BenchmarkResult result_tmac = compute_statistics(timings_tmac);
        
        // Calculate GFLOPS (2 * M * K * N for GEMV)
        double flops = 2.0 * M * K * N;
        result_f32.throughput_gflops = (flops / 1e9) / (result_f32.mean_ms / 1000.0);
        result_tmac.throughput_gflops = (flops / 1e9) / (result_tmac.mean_ms / 1000.0);
        
        // Calculate accuracy (NMSE)
        ggml_tensor * result_f32_tensor = ggml_graph_node(gf_f32, -1);
        ggml_tensor * result_tmac_tensor = ggml_graph_node(gf_tmac, -1);
        
        float * result_f32_data = (float *) result_f32_tensor->data;
        float * result_tmac_data = (float *) result_tmac_tensor->data;
        
        double nmse = 0.0;
        double norm_sq = 0.0;
        for (int i = 0; i < M * N; i++) {
            double diff = result_tmac_data[i] - result_f32_data[i];
            nmse += diff * diff;
            norm_sq += result_f32_data[i] * result_f32_data[i];
        }
        nmse = (norm_sq > 0) ? nmse / norm_sq : 0;
        
        // Print results
        printf("\n--- Results ---\n");
        printf("FP32 Baseline:\n");
        printf("  Mean:       %.3f ms\n", result_f32.mean_ms);
        printf("  Std Dev:    %.3f ms\n", result_f32.std_ms);
        printf("  Min:        %.3f ms\n", result_f32.min_ms);
        printf("  Max:        %.3f ms\n", result_f32.max_ms);
        printf("  Throughput: %.2f GFLOPS\n", result_f32.throughput_gflops);
        
        printf("\nT-MAC (%d-bit):\n", nbits);
        printf("  Mean:       %.3f ms\n", result_tmac.mean_ms);
        printf("  Std Dev:    %.3f ms\n", result_tmac.std_ms);
        printf("  Min:        %.3f ms\n", result_tmac.min_ms);
        printf("  Max:        %.3f ms\n", result_tmac.max_ms);
        printf("  Throughput: %.2f GFLOPS\n", result_tmac.throughput_gflops);
        
        printf("\nSpeedup:    %.2fx\n", result_f32.mean_ms / result_tmac.mean_ms);
        printf("NMSE:       %.8f\n", nmse);
        
        // Cleanup
        aligned_free(tmac_qs);
        ggml_backend_buffer_free(tmac_buf);
        ggml_backend_buffer_free(tmac_buf2);
    }
    
    void run_benchmark_suite() {
        // Test different matrix sizes
        struct TestCase {
            int M, K, N, bits;
        };
        
        std::vector<TestCase> test_cases = {
            // Small matrices
            {128, 128, 1, 4},
            {128, 128, 1, 2},
            
            // Medium matrices
            {256, 256, 1, 4},
            {256, 256, 1, 2},
            {512, 512, 1, 4},
            {512, 512, 1, 2},
            
            // Large matrices (typical LLM sizes)
            {1024, 1024, 1, 4},
            {1024, 1024, 1, 2},
            {2048, 2048, 1, 4},
            {2048, 2048, 1, 2},
            {4096, 4096, 1, 4},
            {4096, 4096, 1, 2},
            
            // Batch processing
            {1024, 1024, 8, 4},
            {1024, 1024, 8, 2},
            {2048, 2048, 4, 4},
            {2048, 2048, 4, 2},
        };
        
        printf("\n");
        printf("================================================================================\n");
        printf("                        T-MAC GeMV Performance Benchmark                        \n");
        printf("================================================================================\n");
        printf("Configuration:\n");
        printf("  Warmup iterations:   %d\n", n_warmup);
        printf("  Timing iterations:   %d\n", n_iterations);
        printf("  Number of threads:   %d\n", n_threads);
        printf("================================================================================\n");
        
        for (const auto& test : test_cases) {
            try {
                benchmark_gemv_size(test.M, test.K, test.N, test.bits);
            } catch (const std::exception& e) {
                printf("Error in test case M=%d, K=%d, N=%d, bits=%d: %s\n", 
                       test.M, test.K, test.N, test.bits, e.what());
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
        TMACBenchmark benchmark;
        benchmark.run_benchmark_suite();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    }
}