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
#include <algorithm>

// Use fixed seed for reproducible results
static std::mt19937 g_rng(42);

struct tensor_stats {
    double mean = 0.0;
    double std_dev = 0.0;
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    size_t elements = 0;
};

static void fill_tensor_f32(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    float *                               data       = (float *) dst->data;
    size_t                                n_elements = ggml_nelements(dst);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = dis(g_rng);
    }
}

static void fill_tensor_f16(ggml_tensor * dst, float min_val = -1.0f, float max_val = 1.0f) {
    ggml_fp16_t *                         data       = (ggml_fp16_t *) dst->data;
    size_t                                n_elements = ggml_nelements(dst);
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n_elements; i++) {
        data[i] = ggml_fp32_to_fp16(dis(g_rng));
    }
}

static void print_tensor_info(const char * name, ggml_tensor * tensor) {
    printf("%s: [%ld, %ld, %ld, %ld] type=%s, elements=%ld\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2],
           tensor->ne[3], ggml_type_name(tensor->type), ggml_nelements(tensor));
}

static struct ggml_tensor * compute_graph(
    ggml_context* ctx,
    struct ggml_cgraph * gf,
    int n_threads = 12
) {
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    return ggml_graph_node(gf, -1);
}

// Convert tensor to float array for calculations
static std::vector<float> tensor_to_float_array(ggml_tensor* tensor) {
    size_t n_elements = ggml_nelements(tensor);
    std::vector<float> data(n_elements);
    
    if (tensor->type == GGML_TYPE_F32) {
        memcpy(data.data(), tensor->data, n_elements * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t* fp16_data = (ggml_fp16_t*)tensor->data;
        for (size_t i = 0; i < n_elements; i++) {
            data[i] = ggml_fp16_to_fp32(fp16_data[i]);
        }
    } else {
        auto tt = ggml_get_type_traits(tensor->type);
        if (tt && tt->to_float) {
            tt->to_float(tensor->data, data.data(), n_elements);
        } else {
            fprintf(stderr, "Unsupported tensor type: %s\n", ggml_type_name(tensor->type));
        }
    }
    
    return data;
}

// Compute tensor statistics
static tensor_stats compute_tensor_stats(const std::vector<float>& data) {
    tensor_stats stats;
    stats.elements = data.size();
    
    if (data.empty()) {
        return stats;
    }
    
    // Compute mean
    double sum = 0.0;
    stats.min_val = data[0];
    stats.max_val = data[0];
    
    for (size_t i = 0; i < data.size(); i++) {
        double val = data[i];
        sum += val;
        stats.min_val = std::min(stats.min_val, val);
        stats.max_val = std::max(stats.max_val, val);
    }
    
    stats.mean = sum / data.size();
    
    // Compute standard deviation
    double var_sum = 0.0;
    for (size_t i = 0; i < data.size(); i++) {
        double diff = data[i] - stats.mean;
        var_sum += diff * diff;
    }
    stats.std_dev = std::sqrt(var_sum / data.size());
    
    return stats;
}

// Calculate NMSE (Normalized Mean Square Error)
static double calculate_nmse(const std::vector<float>& original, const std::vector<float>& quantized) {
    if (original.size() != quantized.size() || original.empty()) {
        return -1.0;
    }
    
    double mse = 0.0;
    double signal_power = 0.0;
    
    for (size_t i = 0; i < original.size(); i++) {
        double diff = original[i] - quantized[i];
        mse += diff * diff;
        signal_power += original[i] * original[i];
    }
    
    mse /= original.size();
    signal_power /= original.size();
    
    if (signal_power == 0.0) {
        return -1.0;
    }
    
    return mse / signal_power;
}

// Calculate SQNR (Signal-to-Quantization-Noise Ratio) in dB
static double calculate_sqnr(const std::vector<float>& original, const std::vector<float>& quantized) {
    double nmse = calculate_nmse(original, quantized);
    if (nmse <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return -10.0 * std::log10(nmse);
}

// Print comparison table
static void print_comparison_table(
    const char* tensor_name,
    const tensor_stats& orig_stats,
    const tensor_stats& quant_stats,
    double nmse,
    double sqnr,
    const char* quant_type
) {
    // Determine quality assessment
    const char* assessment;
    const char* quality_icon;
    
    if (nmse < 1e-4) {
        assessment = "Excellent (< 1e-4)";
        quality_icon = "🟢";
    } else if (nmse < 1e-3) {
        assessment = "Very Good (< 1e-3)";
        quality_icon = "🟡";
    } else if (nmse < 1e-2) {
        assessment = "Good (< 1e-2)";
        quality_icon = "🟠";
    } else if (nmse < 1e-1) {
        assessment = "Fair (< 1e-1)";
        quality_icon = "🔴";
    } else {
        assessment = "Poor (>= 1e-1)";
        quality_icon = "🔴";
    }
    
    printf("\n");
    printf("===== Quantization Analysis for %s =====\n", tensor_name);
    printf("Quantization Type: %s\n\n", quant_type);
    
    printf("+-------------------------+------------+------------+------------+--------------+--------------------+\n");
    printf("| %-23s | %-10s | %-10s | %-10s | %-12s | %-18s |\n", 
            "Tensor Statistics", "Elements", "Mean", "Std Dev", "Min/Max", "Metrics");
    printf("+-------------------------+------------+------------+------------+--------------+--------------------+\n");
    printf("| %-23s | %-10zu | %10.6f | %10.6f | %5.3f/%-5.3f | %-18s |\n",
            "Original (FP16)", orig_stats.elements, orig_stats.mean, orig_stats.std_dev, 
            orig_stats.min_val, orig_stats.max_val, "");
    printf("| %-23s | %-10zu | %10.6f | %10.6f | %5.3f/%-5.3f | %-18s |\n",
            "Dequantized", quant_stats.elements, quant_stats.mean, quant_stats.std_dev,
            quant_stats.min_val, quant_stats.max_val, "");
    printf("+-------------------------+------------+------------+------------+--------------+--------------------+\n");
    printf("| %-23s | %-10s | %-10s | %-10s | %-12s | %s %-15s |\n",
            "Quantization Quality", "-", "-", "-", "-", quality_icon, "");
    printf("| %-23s | %-10s | %-10s | %-10s | %-12s | %.6e       |\n",
            "NMSE", "-", "-", "-", "-", nmse);
    printf("| %-23s | %-10s | %-10s | %-10s | %-12s | %.2f dB         |\n",
            "SQNR", "-", "-", "-", "-", sqnr);
    printf("| %-23s | %-10s | %-10s | %-10s | %-12s | %-18s |\n",
            "Quality Assessment", "-", "-", "-", "-", assessment);
    printf("+-------------------------+------------+------------+------------+--------------+--------------------+\n");
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                printf("                                      ..., \n");
                i2 = ne[2] - n;
            }
            printf("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    printf("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                printf("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    printf("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) printf(", ");
                }
                printf("],\n");
            }
            printf("                                      ],\n");
        }
        printf("                                     ]\n");
        printf("                                     sum = %f\n", sum);
    }
}

// Test different quantization types
static void test_quantization(ggml_context* ctx, ggml_tensor* original, ggml_type quant_type, const char* type_name) {
    // Create quantized tensor
    ggml_tensor * quantized = ggml_new_tensor_4d(ctx, quant_type, 
        original->ne[0], original->ne[1], original->ne[2], original->ne[3]);
    ggml_set_name(quantized, "quantized");
    
    // Create dequantized tensor (always F32 for comparison)
    ggml_tensor * dequantized = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
        original->ne[0], original->ne[1], original->ne[2], original->ne[3]);
    ggml_set_name(dequantized, "dequantized");
    
    // Build computation graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    
    // Quantize: original -> quantized
    ggml_tensor * quant_op = ggml_cpy(ctx, original, quantized);
    ggml_build_forward_expand(gf, quant_op);
    
    // Dequantize: quantized -> dequantized
    ggml_tensor * dequant_op = ggml_cpy(ctx, quantized, dequantized);
    ggml_build_forward_expand(gf, dequant_op);
    
    // Execute graph
    compute_graph(ctx, gf, 1);

    // ggml_print_tensor((uint8_t *)original->data, original->type, original->ne, original->nb, 3);

    // printf("----------------------------------------\n");

    // ggml_print_tensor((uint8_t *)dequantized->data, dequantized->type, dequantized->ne, dequantized->nb, 3);

    // Convert tensors to float arrays
    std::vector<float> orig_data = tensor_to_float_array(original);
    std::vector<float> dequant_data = tensor_to_float_array(dequantized);
    
    // Compute statistics
    tensor_stats orig_stats = compute_tensor_stats(orig_data);
    tensor_stats dequant_stats = compute_tensor_stats(dequant_data);
    
    // Calculate quantization metrics
    double nmse = calculate_nmse(orig_data, dequant_data);
    double sqnr = calculate_sqnr(orig_data, dequant_data);
    
    // Print results in table format
    print_comparison_table("K", orig_stats, dequant_stats, nmse, sqnr, type_name);
}

int main() {    
    struct ggml_init_params params = {
        .mem_size   = 1024*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize GGML context\n");
        return 1;
    }

    const int64_t head_dim   = 128;
    const int64_t seq_len    = 1;
    const int64_t n_heads    = 4;
    const int64_t n_kv_heads = 4;
    const int64_t kv_len     = 4;

    // Create tensors
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads,   1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads, 1);

    // Fill with FIXED reproducible data
    printf("\nGenerating fixed test data (seed=42)...\n");
    fill_tensor_f32(q, -0.8f, 0.8f);
    fill_tensor_f16(k, -0.6f, 0.6f);
    fill_tensor_f16(v, -0.7f, 0.7f);

    printf("\nTensor Information:\n");
    print_tensor_info("Q", q);
    print_tensor_info("K", k);
    print_tensor_info("V", v);

    // Test different quantization types
    printf("\n=====================================");
    printf("\n    Quantization Accuracy Analysis   ");
    printf("\n=====================================\n");
    
    // Test Q4_0 quantization
    
    // test_quantization(ctx, k, GGML_TYPE_QLUTATTN_W1G128_K, "QLUTATTN_W1G128_K");
    test_quantization(ctx, k, GGML_TYPE_QLUTATTN_W2G128_K, "QLUTATTN_W2G128_K");
    test_quantization(ctx, k, GGML_TYPE_QLUTATTN_W4G128_K, "QLUTATTN_W4G128_K");
    
    // Test Q4_1 quantization if you want to compare
    // test_quantization(ctx, k, GGML_TYPE_Q4_1, "Q4_1");
    
    // Test Q5_0 quantization if you want to compare
    // test_quantization(ctx, k, GGML_TYPE_Q5_0, "Q5_0");
    
    // Test Q8_0 quantization if you want to compare
    // test_quantization(ctx, k, GGML_TYPE_Q8_0, "Q8_0");

    // Clean up
    ggml_free(ctx);
    
    return 0;
}