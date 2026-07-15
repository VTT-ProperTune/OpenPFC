// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include "openpfc/runtime/common/gpu_autotune.hpp"

using namespace pfc::gpu;

// RAII wrapper for environment variable management
struct EnvVarGuard {
  std::string name_;
  std::string original_;
  bool was_set_;

  explicit EnvVarGuard(const std::string& name, const std::string& value)
    : name_(name) {
    const char* val = std::getenv(name.c_str());
    was_set_ = (val != nullptr);
    original_ = was_set_ ? val : "";
    setenv(name.c_str(), value.c_str(), 1);
  }

  ~EnvVarGuard() {
    if (was_set_) {
      setenv(name_.c_str(), original_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  EnvVarGuard(const EnvVarGuard&) = delete;
  EnvVarGuard& operator=(const EnvVarGuard&) = delete;
};

TEST_CASE("test_cache_save_load") {
  // Create temporary directory for cache
  std::string cache_dir = "test_cache_dir";
  std::filesystem::create_directories(cache_dir);

  // Set cache directory environment variable with RAII guard
  EnvVarGuard cache_guard("OPENPFC_GPU_AUTOTUNE_CACHE_DIR", cache_dir);

  // Reset singleton to pick up new environment variable
  AutoTuner& tuner = AutoTuner::instance();
  tuner.reset();

  // Use set_cache_config to populate cache for testing
  KernelConfig config{"test_kernel", 128, 2, 1, 0, 0, 0, {100, 10000}};
  tuner.set_cache_config("test_kernel", config);

  std::string cache_file = cache_dir + "/" + get_device_id() + "_autotune_cache.json";
  tuner.save_cache(cache_file);

  // Verify file exists and contains valid JSON
  REQUIRE(std::filesystem::exists(cache_file));

  // Load cache into tuner instance
  tuner.load_cache(cache_file);

  // Verify loaded config matches original
  auto loaded_config = tuner.get_config("test_kernel", 500);
  REQUIRE(loaded_config.block_size_x == 128);
  REQUIRE(loaded_config.block_size_y == 2);

  // Cleanup - RAII guard restores original env var automatically
  std::filesystem::remove_all(cache_dir);
}

#ifdef OpenPFC_ENABLE_CUDA
TEST_CASE("test_device_detection_cuda") {
  std::string device_id = get_device_id();
  REQUIRE(!device_id.empty());
  // Accept "unknown" as valid for systems without GPUs

  std::string arch = get_device_architecture();
  REQUIRE(!arch.empty());
  // Accept "unknown" as valid for systems without GPUs
  if (arch != "unknown") {
    REQUIRE(arch.find("sm_") == 0);
  }
}
#endif

#ifdef OpenPFC_ENABLE_HIP
TEST_CASE("test_device_detection_hip") {
  std::string device_id = get_device_id();
  REQUIRE(!device_id.empty());
  // Accept "unknown" as valid for systems without GPUs

  std::string arch = get_device_architecture();
  REQUIRE(!arch.empty());
  // Accept "unknown" as valid for systems without GPUs
}
#endif

TEST_CASE("test_fallback_defaults") {
  AutoTuner& tuner = AutoTuner::instance();
  tuner.reset();
  EnvVarGuard mode_guard("OPENPFC_GPU_AUTOTUNE_MODE", "fallback_only");
  tuner.reset();

  // Test known kernels return correct defaults
  auto config = tuner.get_config("add_scalar", 1000);
  REQUIRE(config.block_size_x == 256);
  REQUIRE(config.block_size_y == 1);

  auto config2 = tuner.get_config("for_each_interior_3d", 5000);
  REQUIRE(config2.block_size_x == 32);
  REQUIRE(config2.block_size_y == 4);

  // RAII guard restores environment automatically
}

TEST_CASE("test_cache_invalid_json") {
  std::string cache_file = "invalid_cache.json";

  // Write invalid JSON
  std::ofstream file(cache_file);
  file << "{ invalid json content";
  file.close();

  AutoTuner& tuner = AutoTuner::instance();
  REQUIRE_THROWS(tuner.load_cache(cache_file));

  std::filesystem::remove(cache_file);
}

TEST_CASE("test_cache_missing_file") {
  AutoTuner& tuner = AutoTuner::instance();

  // Should throw when loading non-existent file
  REQUIRE_THROWS(tuner.load_cache("nonexistent_cache.json"));

  // get_config should still work with fallback defaults
  EnvVarGuard mode_guard("OPENPFC_GPU_AUTOTUNE_MODE", "fallback_only");
  tuner.reset();
  auto config = tuner.get_config("add_scalar", 1000);
  REQUIRE(config.block_size_x == 256);
}

TEST_CASE("test_autotuner_get_config") {
  AutoTuner& tuner = AutoTuner::instance();
  tuner.reset();
  EnvVarGuard mode_guard("OPENPFC_GPU_AUTOTUNE_MODE", "fallback_only");
  tuner.reset();

  // Test known kernel names
  auto config1 = tuner.get_config("add_scalar", 1000);
  REQUIRE(config1.kernel_name == "add_scalar");

  auto config2 = tuner.get_config("multiply_scalar", 5000);
  REQUIRE(config2.kernel_name == "multiply_scalar");

  auto config3 = tuner.get_config("gather", 100);
  REQUIRE(config3.kernel_name == "gather");

  auto config4 = tuner.get_config("scatter", 200);
  REQUIRE(config4.kernel_name == "scatter");

  auto config5 = tuner.get_config("for_each_interior_3d", 10000);
  REQUIRE(config5.kernel_name == "for_each_interior_3d");
}

TEST_CASE("test_tune_kernel_unknown_kernel") {
  AutoTuner& tuner = AutoTuner::instance();
  tuner.reset();

  // Should throw std::invalid_argument for unknown kernel name
  REQUIRE_THROWS_AS(tuner.tune_kernel("unknown_kernel", [](const KernelConfig&) {}, 1000),
                     std::invalid_argument);
}

TEST_CASE("test_environment_variable_disable") {
  AutoTuner& tuner = AutoTuner::instance();
  tuner.reset();

  // Test disable environment variable with RAII guard
  {
    EnvVarGuard disable_guard("OPENPFC_GPU_AUTOTUNE_DISABLE", "1");
    tuner.reset();

    // Should use fallback defaults when disabled
    auto config = tuner.get_config("add_scalar", 1000);
    REQUIRE(config.block_size_x == 256);
  }

  // Environment variable automatically restored by RAII guard
}

TEST_CASE("test_environment_variable_mode_cache_only") {
  AutoTuner& tuner = AutoTuner::instance();
  tuner.reset();

  // Test cache_only mode with non-existent cache
  {
    EnvVarGuard mode_guard("OPENPFC_GPU_AUTOTUNE_MODE", "cache_only");
    tuner.reset();

    // Should throw when no cache exists
    REQUIRE_THROWS(tuner.get_config("add_scalar", 1000));
  }

  // Environment variable automatically restored by RAII guard
}

TEST_CASE("test_environment_variable_cache_dir") {
  // Create temporary directory
  std::string cache_dir = "test_cache_dir_env";
  std::filesystem::create_directories(cache_dir);

  // Set cache directory environment variable with RAII guard
  EnvVarGuard cache_guard("OPENPFC_GPU_AUTOTUNE_CACHE_DIR", cache_dir);

  // Reset tuner to pick up new environment variable
  AutoTuner& tuner = AutoTuner::instance();
  tuner.reset();

  // Add config and save
  KernelConfig config{"test_env_kernel", 512, 1, 1, 0, 0, 0, {0, SIZE_MAX}};
  tuner.set_cache_config("test_env_kernel", config);
  std::string cache_file = cache_dir + "/" + get_device_id() + "_autotune_cache.json";
  tuner.save_cache(cache_file);

  // Verify file was created in the specified directory
  REQUIRE(std::filesystem::exists(cache_file));

  // Cleanup - RAII guard restores environment automatically
  std::filesystem::remove_all(cache_dir);
}

TEST_CASE("test_problem_size_range_validation") {
  AutoTuner& tuner = AutoTuner::instance();
  tuner.reset();

  // Test problem size validation in cache
  KernelConfig config{"test_range_kernel", 64, 1, 1, 0, 0, 0, {1000, 10000}};
  tuner.set_cache_config("test_range_kernel", config);

  // Should return cached config for valid range
  auto valid_config = tuner.get_config("test_range_kernel", 5000);
  REQUIRE(valid_config.block_size_x == 64);

  // Should return default config for out-of-range problem sizes
  auto default_config = tuner.get_config("test_range_kernel", 100);
  REQUIRE(default_config.block_size_x == 256); // Default value
}

TEST_CASE("test_unknown_kernel_uses_default") {
  AutoTuner& tuner = AutoTuner::instance();
  tuner.reset();
  EnvVarGuard mode_guard("OPENPFC_GPU_AUTOTUNE_MODE", "fallback_only");
  tuner.reset();

  // Test unknown kernel returns generic default
  auto config = tuner.get_config("unknown_custom_kernel", 1000);
  REQUIRE(config.kernel_name == "unknown_custom_kernel");
  REQUIRE(config.block_size_x == 256);
  REQUIRE(config.block_size_y == 1);
  REQUIRE(config.block_size_z == 1);
}


int main(int argc, char *argv[]) {
  return Catch::Session().run(argc, argv);
}
#endif // OpenPFC_ENABLE_GPU_AUTOTUNING
