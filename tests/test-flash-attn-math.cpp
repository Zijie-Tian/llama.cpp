#include <iostream>
#include <cmath>
#include <vector>

// 手工验证flash attention的在线算法
void test_manual_flash_attention() {
    std::cout << "=== Manual Flash Attention Test ===" << std::endl;
    
    // 测试数据
    std::vector<float> Q = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> K0 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> K1 = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> V0 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> V1 = {5.0f, 6.0f, 7.0f, 8.0f};
    
    float scale = 1.0f / sqrt(4.0f);
    
    // 计算QK scores
    float QK0 = Q[0] * K0[0] + Q[1] * K0[1] + Q[2] * K0[2] + Q[3] * K0[3];
    float QK1 = Q[0] * K1[0] + Q[1] * K1[1] + Q[2] * K1[2] + Q[3] * K1[3];
    
    QK0 *= scale;
    QK1 *= scale;
    
    std::cout << "QK0 = " << QK0 << ", QK1 = " << QK1 << std::endl;
    
    // 标准softmax方法
    float max_score = std::max(QK0, QK1);
    float exp0 = exp(QK0 - max_score);
    float exp1 = exp(QK1 - max_score);
    float sum_exp = exp0 + exp1;
    
    float w0 = exp0 / sum_exp;
    float w1 = exp1 / sum_exp;
    
    std::cout << "Standard softmax:" << std::endl;
    std::cout << "  max_score = " << max_score << std::endl;
    std::cout << "  exp0 = " << exp0 << ", exp1 = " << exp1 << std::endl;
    std::cout << "  w0 = " << w0 << ", w1 = " << w1 << std::endl;
    
    std::vector<float> standard_result(4);
    for (int i = 0; i < 4; i++) {
        standard_result[i] = w0 * V0[i] + w1 * V1[i];
    }
    
    std::cout << "Standard result: [" << standard_result[0] << ", " << standard_result[1] 
              << ", " << standard_result[2] << ", " << standard_result[3] << "]" << std::endl;
    
    // 在线flash attention方法
    std::cout << "\\nOnline Flash Attention:" << std::endl;
    
    // 第一步：处理第一个token
    float M = QK0;  // 当前最大值
    float S = exp(QK0 - M);  // 当前累积和 
    std::vector<float> O(4);
    for (int i = 0; i < 4; i++) {
        O[i] = V0[i] * exp(QK0 - M);  // 累积输出
    }
    
    std::cout << "After step 1 (token 0):" << std::endl;
    std::cout << "  M = " << M << ", S = " << S << std::endl;
    std::cout << "  O = [" << O[0] << ", " << O[1] << ", " << O[2] << ", " << O[3] << "]" << std::endl;
    
    // 第二步：处理第二个token
    float new_score = QK1;
    float M_old = M;
    
    if (new_score > M) {
        M = new_score;
        float scale_factor = exp(M_old - M);
        S = S * scale_factor + exp(new_score - M);
        for (int i = 0; i < 4; i++) {
            O[i] = O[i] * scale_factor + V1[i] * exp(new_score - M);
        }
    } else {
        S = S + exp(new_score - M);
        for (int i = 0; i < 4; i++) {
            O[i] = O[i] + V1[i] * exp(new_score - M);
        }
    }
    
    std::cout << "After step 2 (token 1):" << std::endl;
    std::cout << "  M = " << M << ", S = " << S << std::endl;
    std::cout << "  O (unnormalized) = [" << O[0] << ", " << O[1] << ", " << O[2] << ", " << O[3] << "]" << std::endl;
    
    // 最终归一化
    for (int i = 0; i < 4; i++) {
        O[i] /= S;
    }
    
    std::cout << "  O (normalized) = [" << O[0] << ", " << O[1] << ", " << O[2] << ", " << O[3] << "]" << std::endl;
    
    // 比较结果
    float max_diff = 0.0f;
    for (int i = 0; i < 4; i++) {
        float diff = std::abs(standard_result[i] - O[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    std::cout << "\\nDifference between standard and online: " << max_diff << std::endl;
    if (max_diff < 1e-5) {
        std::cout << "✅ PASS: Online flash attention matches standard!" << std::endl;
    } else {
        std::cout << "❌ FAIL: Results don't match!" << std::endl;
    }
}

int main() {
    test_manual_flash_attention();
    return 0;
}