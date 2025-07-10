#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <random>
#include <cmath>
#include <vector>
#include <float.h>

#ifndef CLAMP
#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#endif

static void pseudo_quantize_qlutattn_f32(
    const float* input,
    uint8_t* quantized,
    float* scales,
    float* zeros,
    int n,
    int n_bit,
    int q_group_size
) {
    int num_groups;
    if (q_group_size > 0) {
        if (n % q_group_size != 0) {
            GGML_ASSERT(0);
        }
        num_groups = n / q_group_size;
    } else if (q_group_size == -1) {
        num_groups = 1;
        q_group_size = n;
    } else {
        num_groups = 1;
        q_group_size = n;
    }

    //> [0, 2^n_bit - 1]
    const int max_int = (1 << n_bit) - 1;
    const int min_int = 0;

    for (int g = 0; g < num_groups; ++g) {
        int start_idx = g * q_group_size;
        int end_idx = start_idx + q_group_size;

        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;

        for (int i = start_idx; i < end_idx; ++i) {
            if (input[i] > max_val) max_val = input[i];
            if (input[i] < min_val) min_val = input[i];
        }

        scales[g] = (max_val - min_val < 1e-5f ? 1e-5f : (max_val - min_val)) / max_int;
        float zeros_int = CLAMP(-roundf(min_val / scales[g]), 0.0f, (float)max_int);
        zeros[g] = (zeros_int - (1 << (n_bit - 1))) * scales[g];

        for (int i = start_idx; i < end_idx; ++i) {
            int quantized_val = (int)roundf(input[i] / scales[g]) + (int)zeros_int;
            quantized_val = quantized_val < min_int ? min_int : (quantized_val > max_int ? max_int : quantized_val);
            quantized[i] = (uint8_t)quantized_val;
        }
    }
}

// Test helper functions
static void dequantize_per_group(
    const uint8_t* quantized,
    float* output,
    const float* scales,
    const float* zeros,
    int n,
    int n_bit,
    int q_group_size
) {
    int num_groups;
    if (q_group_size > 0) {
        num_groups = n / q_group_size;
    } else {
        num_groups = 1;
        q_group_size = n;
    }
    
    for (int g = 0; g < num_groups; ++g) {
        int start_idx = g * q_group_size;
        int end_idx = start_idx + q_group_size;
        
        for (int i = start_idx; i < end_idx; ++i) {
            output[i] = (quantized[i] - (1 << (n_bit - 1))) * scales[g] + zeros[g];
        }
    }
}

static void dequantize_per_channel(
    const uint8_t* quantized,
    float* output,
    const float* scales,
    const float* zeros,
    int n,
    int n_bit,
    int n_rows,
    int q_group_size
) {
    int num_heads = n / n_rows;
    int groups_per_channel = (q_group_size > 0) ? n_rows / q_group_size : 1;
    if (q_group_size <= 0) q_group_size = n_rows;
    
    for (int h = 0; h < num_heads; ++h) {
        for (int g = 0; g < groups_per_channel; ++g) {
            int group_idx = h * groups_per_channel + g;
            int start_offset = g * q_group_size;
            
            for (int i = 0; i < q_group_size; ++i) {
                int idx = h * n_rows + start_offset + i;
                output[idx] = (quantized[idx] - (1 << (n_bit - 1))) * scales[group_idx] + zeros[group_idx];
            }
        }
    }
}

static float calculate_mse(const float* a, const float* b, int n) {
    float mse = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = a[i] - b[i];
        mse += diff * diff;
    }
    return mse / n;
}

void gen_test_data(float* input, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int h = 0; h < n; ++h) {
        input[h] = dist(gen);
    }
}

void print_data(const float* data, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%7.4f", data[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }
    if (n % 16 != 0) {
        printf("\n");
    }
    printf("--------------------------------------------------------------------\n");
}

void print_data_uint8(const uint8_t* data, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%3u (%08b)", (unsigned int)data[i], (unsigned int)data[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }
    if (n % 16 != 0) {
        printf("\n");
    }
    printf("--------------------------------------------------------------------\n");
}


static void print_stats(const char* name, const float* data, int n) {
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    float sum = 0.0f;
    
    for (int i = 0; i < n; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        sum += data[i];
    }
    
    printf("%s stats: min=%.6f, max=%.6f, mean=%.6f\n", name, min_val, max_val, sum/n);
}

int main() {
    printf("Testing per-channel quantization function\n");
    printf("=========================================\n\n");
    
    // Test configuration
    const int n_heads = 4;
    const int head_dim = 128;
    const int n_bits = 4;
    const int test_cases[] = {64, 32, 16, -1}; // Different group sizes, -1 means full channel
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int test_idx = 0; test_idx < sizeof(test_cases)/sizeof(test_cases[0]); ++test_idx) {
        int group_size = test_cases[test_idx];
        printf("Test case %d: group_size=%d\n", test_idx + 1, group_size);
        printf("----------------------------------\n");
        
        // Calculate sizes
        const int n = n_heads * head_dim;
        const int total_groups = (group_size > 0) ? n / group_size : 1;
        
        // Allocate buffers
        std::vector<float> input(n);
        std::vector<uint8_t> quantized(n);
        std::vector<float> scales(total_groups);
        std::vector<float> zeros(total_groups);
        std::vector<float> dequantized(n);
        
        // Generate test data with different patterns for each head
        gen_test_data(input.data(), n);

        // print_data(input.data(), n);
        
        // // Print input stats
        // print_stats("Input", input.data(), n);
        
        // Quantize
        pseudo_quantize_qlutattn_f32(
            input.data(),
            quantized.data(),
            scales.data(),
            zeros.data(),
            n,
            n_bits,
            group_size
        );

        // print_data_uint8(quantized.data(), n);
        
        // Dequantize
        dequantize_per_group(
            quantized.data(),
            dequantized.data(),
            scales.data(),
            zeros.data(),
            n,
            n_bits,
            group_size
        );

        // print_data(dequantized.data(), n);
        
        // Print dequantized stats
        print_stats("Dequantized", dequantized.data(), n);
        
        // Calculate error metrics
        float mse = calculate_mse(input.data(), dequantized.data(), n);
        float rmse = sqrtf(mse);
        printf("Quantization error: MSE=%.6f, RMSE=%.6f\n", mse, rmse);
        
        // Check per-head errors
        printf("Per-head errors:\n");
        for (int h = 0; h < n_heads; ++h) {
            float head_mse = calculate_mse(
                &input[h * head_dim],
                &dequantized[h * head_dim],
                head_dim
            );
            printf("  Head %d: MSE=%.6f, RMSE=%.6f\n", h, head_mse, sqrtf(head_mse));
        }
        
        // Verify quantized values are within bounds
        bool values_valid = true;
        for (int i = 0; i < n; ++i) {
            if (quantized[i] > ((1 << n_bits) - 1)) {
                printf("ERROR: Quantized value %d at index %d exceeds max value\n", quantized[i], i);
                values_valid = false;
            }
        }
        
        if (values_valid) {
            printf("✓ All quantized values are within valid range [0, %d]\n", (1 << n_bits) - 1);
        }
        
        // Print sample data comparison for the first test case
        if (test_idx == 0) {
            printf("\nSample data comparison (first 16 values from each head):\n");
            printf("Head | Index | Original    | Dequantized | Diff       | Quantized\n");
            printf("-----|-------|-------------|-------------|------------|----------\n");
            
            for (int h = 0; h < n_heads; ++h) {
                for (int i = 0; i < 16; ++i) {
                    int idx = h * head_dim + i;
                    float diff = dequantized[idx] - input[idx];
                    printf("  %d  |  %3d  | %11.6f | %11.6f | %10.6f | %3d\n", 
                           h, i, input[idx], dequantized[idx], diff, quantized[idx]);
                }
                if (h < n_heads - 1) {
                    printf("-----|-------|-------------|-------------|------------|----------\n");
                }
            }
        }
        
        printf("\n");
    }
    
    return 0;
}