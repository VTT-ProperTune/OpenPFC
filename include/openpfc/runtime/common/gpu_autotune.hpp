// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file gpu_autotune.hpp
 * @brief GPU kernel auto-tuning framework
 *
 * This file provides a framework for automatically tuning GPU kernel launch
 * parameters based on device architecture and problem size. The framework
 * supports both CUDA and HIP backends with JSON caching and fallback defaults.
 */

#pragma once

#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING

#include <string>
#include <vector>
#include <functional>
#include <utility>
#include <cstddef>
#include <map>
#include <mutex>

#ifdef OpenPFC_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef OpenPFC_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace pfc::gpu {

/**
 * @brief Structure storing optimal kernel launch parameters.
 *
 * This structure contains the optimal configuration for a specific kernel
 * type, including block dimensions, shared memory usage, and valid problem
 * size ranges.
 */
struct KernelConfig {
  std::string kernel_name;                      ///< Name identifier for the kernel
  int block_size_x{256};                        ///< Block dimension in X
  int block_size_y{1};                          ///< Block dimension in Y
  int block_size_z{1};                          ///< Block dimension in Z
  size_t shared_memory_bytes{0};                ///< Shared memory allocation
  int min_grid_size{0};                         ///< Minimum grid size
  int max_grid_size{0};                         ///< Maximum grid size
  std::pair<size_t, size_t> problem_size_range{0, SIZE_MAX}; ///< Valid problem size range [min, max]
};

/**
 * @brief Structure defining tunable parameter ranges for a kernel.
 *
 * This structure contains the ranges of parameters that will be tested
 * during the auto-tuning process for a specific kernel.
 */
struct KernelTuneParams {
  std::vector<int> candidate_block_sizes{64, 128, 256, 512, 1024}; ///< List of 1D block sizes to test
  std::vector<dim3> candidate_block_dims{{32,4,1}, {16,16,1}, {8,8,4}}; ///< List of 3D block dimensions to test
  size_t min_problem_size{1000};                 ///< Minimum problem size for tuning
  size_t max_problem_size{10000000};             ///< Maximum problem size for tuning
};

/**
 * @brief Main class for GPU kernel auto-tuning.
 *
 * This class provides a singleton pattern with dynamic cache directory
 * resolution for managing GPU kernel configurations. It supports multiple
 * tuning modes, device detection, and JSON-based cache persistence.
 *
 * The cache directory is resolved dynamically from environment variables
 * each time it is accessed, enabling proper test isolation and runtime
 * reconfiguration.
 */
class AutoTuner {
public:
  /**
   * @brief Singleton accessor.
   *
   * @return Reference to the singleton AutoTuner instance.
   * @note Thread-safe initialization.
   */
  static AutoTuner& instance() {
    static AutoTuner inst;
    return inst;
  }

  // Delete copy and move operations
  AutoTuner(const AutoTuner&) = delete;
  AutoTuner& operator=(const AutoTuner&) = delete;
  AutoTuner(AutoTuner&&) = delete;
  AutoTuner& operator=(AutoTuner&&) = delete;

  /**
   * @brief Returns optimal configuration for the given kernel and problem size.
   *
   * This method retrieves the optimal configuration from cache or returns
   * fallback defaults if cache is unavailable. The problem_size parameter
   * is used for cache lookup and validation against the problem_size_range.
   *
   * @param kernel_name Identifier for the kernel.
   * @param problem_size Size of the problem (used for cache lookup).
   * @return KernelConfig containing optimal parameters.
   * @throws std::runtime_error if mode is "cache_only" and no cache exists.
   */
  KernelConfig get_config(const std::string& kernel_name, size_t problem_size);

  /**
   * @brief Benchmarks and stores optimal configuration for the kernel.
   *
   * This method runs the kernel with various parameter configurations
   * and selects the best-performing one. The selected configuration is
   * cached and persisted to disk.
   *
   * @param kernel_name Identifier for the kernel.
   * @param kernel Function that launches the kernel with given config.
   * @param problem_size Size of the problem for benchmarking.
   * @throws std::invalid_argument if kernel_name is not in registry.
   */
  void tune_kernel(const std::string& kernel_name,
                   std::function<void(const KernelConfig&)> kernel,
                   size_t problem_size);

  /**
   * @brief Persists current cache to JSON file.
   *
   * @param filepath Path to output cache file.
   */
  void save_cache(const std::string& filepath);

  /**
   * @brief Loads cache from JSON file.
   *
   * @param filepath Path to input cache file.
   * @throws std::runtime_error if file cannot be opened or parsed.
   */
  void load_cache(const std::string& filepath);

  /**
   * @brief Test helper to directly add configuration to cache.
   *
   * @param kernel_name Identifier for the kernel.
   * @param config Configuration to store.
   */
  void set_cache_config(const std::string& kernel_name, const KernelConfig& config);

  /**
   * @brief Test helper to reset tuner to initial state.
   *
   * Clears cache and re-reads environment variables. Cache directory
   * is resolved from current environment after reset.
   */
  void reset();
  /// Fallback default configurations.
  static inline const KernelConfig kDefaultConfigs[] = {
    {"add_scalar", 256, 1, 1, 0, 0, 0, {0, SIZE_MAX}},
    {"multiply_scalar", 256, 1, 1, 0, 0, 0, {0, SIZE_MAX}},
    {"for_each_interior_3d", 32, 4, 1, 0, 0, 0, {0, SIZE_MAX}},
    {"gather", 256, 1, 1, 0, 0, 0, {0, SIZE_MAX}},
    {"scatter", 256, 1, 1, 0, 0, 0, {0, SIZE_MAX}}
  };
  /// Number of default configurations.
  static inline const size_t kNumDefaultConfigs = sizeof(kDefaultConfigs) / sizeof(KernelConfig);
  /// Private constructor for singleton pattern.
  AutoTuner();

  /**
   * @brief Get cache directory dynamically from environment.
   *
   * @return Cache directory path from OPENPFC_GPU_AUTOTUNE_CACHE_DIR
   *         environment variable, or "." if not set.
   */
  std::string get_cache_dir() const;

  std::string mode_;                              ///< Current tuning mode ("auto", "cache_only", "fallback_only")
  bool disabled_;                                 ///< Whether auto-tuning is disabled
  std::string device_id_;                         ///< Unique GPU identifier
  std::string device_arch_;                       ///< GPU compute capability
  std::map<std::string, KernelConfig> cache_;     ///< Cached kernel configurations
  std::map<std::string, KernelTuneParams> registry_; ///< Kernel parameter registries
  mutable std::mutex mutex_;                      ///< Thread safety mutex

  /// Initialize device information (ID and architecture).
  void initialize_device_info();
  /// Initialize kernel registry with default parameters.
  void initialize_registry();
  /**
   * @brief Get environment variable with default value.
   *
   * @param name Environment variable name.
   * @param default_val Default value if variable is not set.
   * @return Environment variable value or default.
   */
  std::string get_env_var(const char* name, const std::string& default_val) const;
  /**
   * @brief Find default configuration for a kernel.
   *
   * @param kernel_name Kernel identifier.
   * @return Default configuration or generic default if not found.
   */
  KernelConfig find_default_config(const std::string& kernel_name);
  /**
   * @brief Check if configuration is valid for given problem size.
   *
   * @param config Configuration to validate.
   * @param problem_size Problem size to check against.
   * @return true if problem_size is within config's valid range.
   */
  bool is_valid_config(const KernelConfig& config, size_t problem_size) const;
};

/**
 * @brief Returns unique GPU identifier (PCI bus ID).
 *
 * @return GPU device ID or "unknown" if detection fails.
 */
std::string get_device_id();

/**
 * @brief Returns GPU compute capability.
 *
 * For CUDA, returns "sm_XX" format (e.g., "sm_80").
 * For HIP, returns architecture name (e.g., "gfx90a").
 *
 * @return GPU architecture or "unknown" if detection fails.
 */
std::string get_device_architecture();

} // namespace pfc::gpu

// Include implementation
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <nlohmann/json.hpp>

#ifdef OpenPFC_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef OpenPFC_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace pfc::gpu {
inline AutoTuner::AutoTuner() {
  disabled_ = get_env_var("OPENPFC_GPU_AUTOTUNE_DISABLE", "0") == "1";
  mode_ = get_env_var("OPENPFC_GPU_AUTOTUNE_MODE", "auto");
  initialize_device_info();
  initialize_registry();

  if (!disabled_ && mode_ != "fallback_only") {
    std::string cache_file = get_cache_dir() + "/" + device_id_ + "_autotune_cache.json";
    try {
      load_cache(cache_file);
    } catch (...) {
      // Cache load failed, will use defaults or auto-tune
    }
  }
}

inline std::string AutoTuner::get_cache_dir() const {
  return get_env_var("OPENPFC_GPU_AUTOTUNE_CACHE_DIR", ".");
}

inline KernelConfig AutoTuner::get_config(const std::string& kernel_name, size_t problem_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (disabled_) {
    return find_default_config(kernel_name);
  }

  auto it = cache_.find(kernel_name);
  if (it != cache_.end()) {
    const auto& config = it->second;
    if (is_valid_config(config, problem_size)) {
      return config;
    }
  }

  if (mode_ == "cache_only") {
    throw std::runtime_error("No cached configuration for kernel: " + kernel_name);
  }

  if (mode_ == "fallback_only") {
    return find_default_config(kernel_name);
  }

  // mode_ == "auto": return default for now, tuning happens separately
  return find_default_config(kernel_name);
}

inline void AutoTuner::tune_kernel(const std::string& kernel_name,
                             std::function<void(const KernelConfig&)> kernel,
                             size_t problem_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto reg_it = registry_.find(kernel_name);
  if (reg_it == registry_.end()) {
    throw std::invalid_argument("Unknown kernel name: " + kernel_name);
  }

  const auto& params = reg_it->second;
  KernelConfig best_config;
  double best_time = std::numeric_limits<double>::max();

  // Benchmark 1D block sizes
  for (int block_size : params.candidate_block_sizes) {
    KernelConfig config{kernel_name, block_size, 1, 1, 0, 0, 0,
                        {problem_size, problem_size}};

    // Warmup runs
    for (int i = 0; i < 3; ++i) kernel(config);

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) kernel(config);
    auto end = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(end - start).count();
    if (time < best_time) {
      best_time = time;
      best_config = config;
    }
  }

  cache_[kernel_name] = best_config;

  std::string cache_file = get_cache_dir() + "/" + device_id_ + "_autotune_cache.json";
  save_cache(cache_file);
}

inline void AutoTuner::save_cache(const std::string& filepath) {
  std::lock_guard<std::mutex> lock(mutex_);

  nlohmann::json j;
  j["device_id"] = device_id_;
  j["device_architecture"] = device_arch_;

  nlohmann::json kernels_json;
  for (const auto& [name, config] : cache_) {
    nlohmann::json config_json;
    config_json["block_size_x"] = config.block_size_x;
    config_json["block_size_y"] = config.block_size_y;
    config_json["block_size_z"] = config.block_size_z;
    config_json["shared_memory_bytes"] = config.shared_memory_bytes;
    config_json["min_grid_size"] = config.min_grid_size;
    config_json["max_grid_size"] = config.max_grid_size;
    config_json["problem_size_range"] = {config.problem_size_range.first,
                                          config.problem_size_range.second};
    kernels_json[name] = config_json;
  }
  j["kernels"] = kernels_json;

  std::ofstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open cache file: " + filepath);
  }
  file << j.dump(2);
}

inline void AutoTuner::load_cache(const std::string& filepath) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open cache file: " + filepath);
  }

  nlohmann::json j;
  try {
    file >> j;
  } catch (const nlohmann::json::parse_error& e) {
    throw std::runtime_error("Invalid JSON in cache file: " + std::string(e.what()));
  }

  // Validate device ID and architecture
  if (j.contains("device_id") && j["device_id"] != device_id_) {
    // Device mismatch, could warn or invalidate cache
  }

  if (j.contains("kernels")) {
    for (auto& [name, config_json] : j["kernels"].items()) {
      KernelConfig config;
      config.kernel_name = name;
      config.block_size_x = config_json.value("block_size_x", 256);
      config.block_size_y = config_json.value("block_size_y", 1);
      config.block_size_z = config_json.value("block_size_z", 1);
      config.shared_memory_bytes = config_json.value("shared_memory_bytes", 0);
      config.min_grid_size = config_json.value("min_grid_size", 0);
      config.max_grid_size = config_json.value("max_grid_size", 0);
      if (config_json.contains("problem_size_range") &&
          config_json["problem_size_range"].is_array() &&
          config_json["problem_size_range"].size() == 2) {
        config.problem_size_range = {config_json["problem_size_range"][0],
                                      config_json["problem_size_range"][1]};
      }
      cache_[name] = config;
    }
  }
}

inline void AutoTuner::set_cache_config(const std::string& kernel_name, const KernelConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_[kernel_name] = config;
}

inline void AutoTuner::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
  disabled_ = get_env_var("OPENPFC_GPU_AUTOTUNE_DISABLE", "0") == "1";
  mode_ = get_env_var("OPENPFC_GPU_AUTOTUNE_MODE", "auto");
}

inline void AutoTuner::initialize_device_info() {
  device_id_ = get_device_id();
  device_arch_ = get_device_architecture();
}

inline void AutoTuner::initialize_registry() {
  registry_["add_scalar"] = KernelTuneParams{};
  registry_["multiply_scalar"] = KernelTuneParams{};
  registry_["for_each_interior_3d"] = KernelTuneParams{};
  registry_["gather"] = KernelTuneParams{};
  registry_["scatter"] = KernelTuneParams{};
  registry_["kobayashi_fd_3d"] = KernelTuneParams{};
}

inline std::string AutoTuner::get_env_var(const char* name, const std::string& default_val) const {
  const char* val = std::getenv(name);
  return val ? val : default_val;
}

inline KernelConfig AutoTuner::find_default_config(const std::string& kernel_name) {
  for (size_t i = 0; i < kNumDefaultConfigs; ++i) {
    if (kDefaultConfigs[i].kernel_name == kernel_name) {
      return kDefaultConfigs[i];
    }
  }
  // Return generic default if not found
  return {kernel_name, 256, 1, 1, 0, 0, 0, {0, SIZE_MAX}};
}

inline bool AutoTuner::is_valid_config(const KernelConfig& config, size_t problem_size) const {
  return problem_size >= config.problem_size_range.first &&
         problem_size <= config.problem_size_range.second;
}

#ifdef OpenPFC_ENABLE_CUDA
inline std::string get_device_id() {
  int device;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return "unknown";
  }
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return "unknown";
  }
  return std::to_string(prop.pciBusID);
}

inline std::string get_device_architecture() {
  int device;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return "unknown";
  }
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return "unknown";
  }
  return "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
}
#endif

#ifdef OpenPFC_ENABLE_HIP
inline std::string get_device_id() {
  int device;
  hipError_t err = hipGetDevice(&device);
  if (err != hipSuccess) {
    return "unknown";
  }
  hipDeviceProp_t prop;
  err = hipGetDeviceProperties(&prop, device);
  if (err != hipSuccess) {
    return "unknown";
  }
  return std::to_string(prop.pciDeviceId);
}

inline std::string get_device_architecture() {
  int device;
  hipError_t err = hipGetDevice(&device);
  if (err != hipSuccess) {
    return "unknown";
  }
  hipDeviceProp_t prop;
  err = hipGetDeviceProperties(&prop, device);
  if (err != hipSuccess) {
    return "unknown";
  }
  return std::string(prop.gcnArch);
}
#endif

} // namespace pfc::gpu

#endif // OpenPFC_ENABLE_GPU_AUTOTUNING
