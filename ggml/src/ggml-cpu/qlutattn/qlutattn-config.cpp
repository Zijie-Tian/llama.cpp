#include "qlutattn-config.h"
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>

//> ===================================================================================================
//> Thread-safe global config management for QLUTATTN
//> ===================================================================================================

namespace {
    // Use singleton pattern for thread-safe global access
    class QlutattnConfigManager {
    private:
        std::unordered_map<std::string, qlutattn_kernel_config> configs;
        mutable std::mutex config_mutex;
        bool initialized = false;
        
        // Generate key from M, K, bits
        static std::string make_key(int M, int K, int bits) {
            return "M" + std::to_string(M) + "_K" + std::to_string(K) + "_b" + std::to_string(bits);
        }
        
        // Initialize default configs
        void init_default_configs() {
            // Default test config for 128x128 with different bit widths
            qlutattn_kernel_config config_4bit = {
                .g                = 4,
                .ngroups_per_elem = 2,  // 8/4 = 2
                .q_group_size     = 128,
                .act_group_size   = 64,
                .has_scale        = true,
                .kfactor          = 16,
                .bits             = 4,
                .actk             = 16,  // 64/4 = 16
                .has_zero_point   = false,
                .one_scale        = true,
                .bm               = 256,
                .simd_n_in        = 16,
                .simd_n_out       = 16,
                .chunk_n          = 8
            };
            
            qlutattn_kernel_config config_2bit = {
                .g                = 4,
                .ngroups_per_elem = 2,  // 8/4 = 2 (even for 2-bit, we process 4 bits at a time)
                .q_group_size     = 128,
                .act_group_size   = 64,
                .has_scale        = true,
                .kfactor          = 16,
                .bits             = 2,
                .actk             = 16,
                .has_zero_point   = false,
                .one_scale        = true,
                .bm               = 256,
                .simd_n_in        = 16,
                .simd_n_out       = 16,
                .chunk_n          = 8
            };
            
            qlutattn_kernel_config config_1bit = {
                .g                = 4,
                .ngroups_per_elem = 2,
                .q_group_size     = 128,
                .act_group_size   = 64,
                .has_scale        = true,
                .kfactor          = 16,
                .bits             = 1,
                .actk             = 16,
                .has_zero_point   = false,
                .one_scale        = true,
                .bm               = 256,
                .simd_n_in        = 16,
                .simd_n_out       = 16,
                .chunk_n          = 8
            };
            
            // Register default configs for common cases
            // These can be overridden by calling register_config
            configs[make_key(1, 1, 4)] = config_4bit;  // For testing
            configs[make_key(128, 128, 4)] = config_4bit;
            configs[make_key(128, 128, 2)] = config_2bit;
            configs[make_key(128, 128, 1)] = config_1bit;
            
            // Add more default configs as needed
            // Can also auto-tune and cache configs dynamically
        }
        
    public:
        static QlutattnConfigManager& instance() {
            static QlutattnConfigManager inst;
            return inst;
        }
        
        void init() {
            std::lock_guard<std::mutex> lock(config_mutex);
            if (!initialized) {
                init_default_configs();
                initialized = true;
            }
        }
        
        const qlutattn_kernel_config* get_config(int M, int K, int bits) const {
            std::lock_guard<std::mutex> lock(config_mutex);
            
            if (!initialized) {
                return nullptr;
            }
            
            std::string key = make_key(M, K, bits);
            auto it = configs.find(key);
            
            if (it != configs.end()) {
                return &it->second;
            }
            
            // Try to find a compatible config (e.g., any M/K with same bits)
            // This is a fallback for testing
            for (const auto& pair : configs) {
                if (pair.second.bits == bits) {
                    return &pair.second;
                }
            }
            
            return nullptr;
        }
        
        bool register_config(int M, int K, int bits, const qlutattn_kernel_config* config) {
            if (!config) return false;
            
            std::lock_guard<std::mutex> lock(config_mutex);
            
            std::string key = make_key(M, K, bits);
            
            // Check if already exists
            if (configs.find(key) != configs.end()) {
                return false;
            }
            
            configs[key] = *config;
            return true;
        }
        
        bool is_initialized() const {
            std::lock_guard<std::mutex> lock(config_mutex);
            return initialized;
        }
        
        void cleanup() {
            std::lock_guard<std::mutex> lock(config_mutex);
            configs.clear();
            initialized = false;
        }
    };
}

//> ===================================================================================================
//> C interface implementation
//> ===================================================================================================

extern "C" {

void ggml_qlutattn_config_init(void) {
    QlutattnConfigManager::instance().init();
}

const struct qlutattn_kernel_config * ggml_qlutattn_get_config(int M, int K, int bits) {
    return QlutattnConfigManager::instance().get_config(M, K, bits);
}

bool ggml_qlutattn_register_config(int M, int K, int bits, const struct qlutattn_kernel_config * config) {
    return QlutattnConfigManager::instance().register_config(M, K, bits, config);
}

bool ggml_qlutattn_config_is_initialized(void) {
    return QlutattnConfigManager::instance().is_initialized();
}

void ggml_qlutattn_config_cleanup(void) {
    QlutattnConfigManager::instance().cleanup();
}

} // extern "C"