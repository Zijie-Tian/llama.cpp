#include <ggml-cpu.h>
#include <ggml.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(_MSC_VER)
#    pragma warning(disable : 4244 4267)  // possible loss of data
#endif

// Configuration
constexpr int kVecSize    = 1 << 18;  // 256K elements
constexpr int kWarmupRuns = 5;
constexpr int kBenchRuns  = 100;

// Copy-pasted from ggml.c
#define QK4_0 32

typedef struct {
    float   d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;

static_assert(sizeof(block_q4_0) == sizeof(float) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK8_0 32

typedef struct {
    float  d;          // delta
    int8_t qs[QK8_0];  // quants
} block_q8_0;

static_assert(sizeof(block_q8_0) == sizeof(float) + QK8_0, "wrong q8_0 block size/padding");

// Random float generation
static float drawFromGaussianPdf(std::mt19937 & rndm) {
    constexpr double kScale           = 1. / (1. + std::mt19937::max());
    constexpr double kTwoPiTimesScale = 6.28318530717958647692 * kScale;
    static float     lastX;
    static bool      haveX = false;
    if (haveX) {
        haveX = false;
        return lastX;
    }
    auto r   = sqrt(-2 * log(1 - kScale * rndm()));
    auto phi = kTwoPiTimesScale * rndm();
    lastX    = r * sin(phi);
    haveX    = true;
    return r * cos(phi);
}

static void fillRandomGaussianFloats(std::vector<float> & values, std::mt19937 & rndm, float mean = 0,
                                     float stddev = 1.0f) {
    for (auto & v : values) {
        v = mean + stddev * drawFromGaussianPdf(rndm);
    }
}

// Benchmarking statistics
struct BenchmarkStats {
    double min_latency      = std::numeric_limits<double>::max();
    double max_latency      = 0;
    double total_latency    = 0;
    double total_latency_sq = 0;
    int    count            = 0;

    void record(double latency_us) {
        min_latency = std::min(min_latency, latency_us);
        max_latency = std::max(max_latency, latency_us);
        total_latency += latency_us;
        total_latency_sq += latency_us * latency_us;
        count++;
    }

    double mean() const { return count > 0 ? total_latency / count : 0; }

    double stddev() const {
        if (count <= 1) {
            return 0;
        }
        double mean_val = mean();
        double variance = (total_latency_sq / count) - (mean_val * mean_val);
        return variance > 0 ? sqrt(variance) : 0;
    }

    void print(const char * name) const {
        printf("\n=== %s Latency Statistics ===\n", name);
        printf("  Iterations:     %d\n", count);
        printf("  Min latency:    %.3f us\n", min_latency);
        printf("  Max latency:    %.3f us\n", max_latency);
        printf("  Mean latency:   %.3f us\n", mean());
        printf("  Stddev:         %.3f us\n", stddev());
        printf("  Throughput:     %.2f GFLOPS\n", compute_gflops());
    }

    double compute_gflops() const {
        // For Q4_0 x FP32 dot product:
        // Each block processes QK4_0 elements
        // Operations per block: QK4_0 multiplications + (QK4_0-1) additions ≈ 2*QK4_0 ops
        // Total blocks: kVecSize / QK4_0
        double ops_per_dot  = 2.0 * kVecSize;
        double time_seconds = mean() * 1e-6;
        return (ops_per_dot / time_seconds) * 1e-9;
    }
};

// Scalar implementation for reference
static float scalar_dot_q4_0_f32(int n, const block_q4_0 * x, const float * y) {
    const static float kValues[16] = { -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f,
                                       0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f };
    float              sum         = 0;
    for (int i = 0; i < n; ++i) {
        float block_sum = 0;
        for (int j = 0; j < QK4_0 / 2; ++j) {
            uint8_t v  = x[i].qs[j];
            int     i0 = v & 0xf;
            int     i1 = v >> 4;
            block_sum += y[2 * j] * kValues[i0] + y[2 * j + 1] * kValues[i1];
        }
        sum += x[i].d * block_sum;
        y += QK4_0;
    }
    return sum;
}

// Q4_0 x Q8_0 dot product
static float scalar_dot_q4_0_q8_0(int n, const block_q4_0 * x, const block_q8_0 * y) {
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        int block_sum = 0;
        for (int j = 0; j < QK8_0 / 2; ++j) {
            uint8_t v  = x[i].qs[j];
            int     i0 = (int8_t) (v & 0xf) - 8;
            int     i1 = (int8_t) (v >> 4) - 8;
            block_sum += i0 * y[i].qs[2 * j] + i1 * y[i].qs[2 * j + 1];
        }
        sum += x[i].d * y[i].d * block_sum;
    }
    return sum;
}

int main(int argc, char ** argv) {
    printf("Q4_0 End-to-End Latency Benchmark\n");
    printf("==================================\n");
    printf("Vector size: %d elements\n", kVecSize);
    printf("Block size:  %d (QK4_0)\n", QK4_0);
    printf("Warmup runs: %d\n", kWarmupRuns);
    printf("Bench runs:  %d\n", kBenchRuns);

    // Parse command line arguments
    bool use_q8_0 = argc > 1 && atoi(argv[1]) == 1;
    bool use_ggml = argc > 2 && atoi(argv[2]) == 1;

    printf("\nConfiguration:\n");
    printf("  Second operand: %s\n", use_q8_0 ? "Q8_0" : "FP32");
    printf("  Implementation: %s\n", use_ggml ? "GGML" : "Scalar");

    // Initialize random generator
    std::mt19937 rndm(1234);

    // Allocate vectors
    std::vector<float> x_fp32(kVecSize);
    std::vector<float> y_fp32(kVecSize);

    int                     n_blocks = kVecSize / QK4_0;
    std::vector<block_q4_0> x_q4_0(n_blocks);
    std::vector<block_q8_0> y_q8_0(n_blocks);

    // Get GGML function pointers
    const auto * q4_0_funcs = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0);
    const auto * q8_0_funcs = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);

    BenchmarkStats stats;

    printf("\nRunning benchmark...\n");

    for (int iter = 0; iter < kWarmupRuns + kBenchRuns; ++iter) {
        // Generate random data
        fillRandomGaussianFloats(x_fp32, rndm, 0, 1.0f);
        fillRandomGaussianFloats(y_fp32, rndm, 0, 1.0f);

        // Quantize x to Q4_0
        q4_0_funcs->from_float(x_fp32.data(), x_q4_0.data(), kVecSize);

        // Quantize y to Q8_0 if needed
        if (use_q8_0) {
            q8_0_funcs->from_float(y_fp32.data(), y_q8_0.data(), kVecSize);
        }

        // Benchmark the dot product
        auto t_start = std::chrono::high_resolution_clock::now();

        float result = 0;
        if (use_ggml) {
            // Use GGML implementation
            if (use_q8_0) {
                q4_0_funcs->vec_dot(kVecSize, &result, 0, x_q4_0.data(), 0, y_q8_0.data(), 0, 1);
            } else {
                // For Q4_0 x FP32, we need to use Q8_0 as intermediary since GGML requires both operands quantized
                // Quantize y to Q8_0 temporarily for the dot product
                std::vector<block_q8_0> y_q8_0_temp(n_blocks);
                q8_0_funcs->from_float(y_fp32.data(), y_q8_0_temp.data(), kVecSize);
                q4_0_funcs->vec_dot(kVecSize, &result, 0, x_q4_0.data(), 0, y_q8_0_temp.data(), 0, 1);
            }
        } else {
            // Use scalar implementation
            if (use_q8_0) {
                result = scalar_dot_q4_0_q8_0(n_blocks, x_q4_0.data(), y_q8_0.data());
            } else {
                result = scalar_dot_q4_0_f32(n_blocks, x_q4_0.data(), y_fp32.data());
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();

        // Calculate latency in microseconds
        double latency_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

        // Skip warmup runs
        if (iter >= kWarmupRuns) {
            stats.record(latency_us);
            if ((iter - kWarmupRuns + 1) % 10 == 0) {
                printf("  Progress: %d/%d (latest: %.2f us, result: %.6f)\n", iter - kWarmupRuns + 1, kBenchRuns,
                       latency_us, result);
            }
        }
    }

    // Print statistics
    char name[256];
    snprintf(name, sizeof(name), "Q4_0 x %s (%s)", use_q8_0 ? "Q8_0" : "FP32", use_ggml ? "GGML" : "Scalar");
    stats.print(name);

    // Additional analysis
    printf("\n=== Performance Analysis ===\n");
    double bytes_per_q4_0_block = sizeof(block_q4_0);
    double bytes_per_q8_0_block = sizeof(block_q8_0);
    double bytes_per_fp32       = sizeof(float);

    double total_bytes;
    if (use_q8_0) {
        total_bytes = n_blocks * (bytes_per_q4_0_block + bytes_per_q8_0_block);
    } else {
        total_bytes = n_blocks * bytes_per_q4_0_block + kVecSize * bytes_per_fp32;
    }

    double bandwidth_gbps = (total_bytes / stats.mean()) / 1e3;  // GB/s
    printf("  Memory bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("  Bytes per FLOP:   %.3f\n", total_bytes / (2.0 * kVecSize));

    return 0;
}
