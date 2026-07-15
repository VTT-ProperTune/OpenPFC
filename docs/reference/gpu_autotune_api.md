<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# GPU Auto-Tuning API Reference

## Namespace `pfc::gpu`

The GPU auto-tuning functionality is provided through the `pfc::gpu` namespace, which contains all auto-tuning related types, classes, and functions.

### `KernelConfig`

Structure storing optimal kernel launch parameters.

**Fields:**
- `std::string kernel_name` - Name identifier for the kernel
- `int block_size_x` - Block dimension in X (default: 256)
- `int block_size_y` - Block dimension in Y (default: 1)
- `int block_size_z` - Block dimension in Z (default: 1)
- `size_t shared_memory_bytes` - Shared memory allocation (default: 0)
- `int min_grid_size` - Minimum grid size (default: 0)
- `int max_grid_size` - Maximum grid size (default: 0)
- `std::pair<size_t, size_t> problem_size_range` - Valid problem size range [min, max]

### `KernelTuneParams`

Structure defining tunable parameter ranges for a kernel.

**Fields:**
- `std::vector<int> candidate_block_sizes` - List of 1D block sizes to test
- `std::vector<dim3> candidate_block_dims` - List of 3D block dimensions to test
- `size_t min_problem_size` - Minimum problem size for tuning
- `size_t max_problem_size` - Maximum problem size for tuning

### `AutoTuner`

Main class for GPU kernel auto-tuning. Uses singleton pattern with dynamic cache directory resolution.

**Singleton Accessor:**
```cpp
static AutoTuner& instance();
```
- Returns reference to the singleton AutoTuner instance
- Thread-safe initialization
- Cache directory is resolved dynamically from environment variables

**Constructor (private):**
```cpp
AutoTuner();
```
- Private constructor for singleton pattern
- Reads environment variables but does not store cache directory
- Use singleton accessor instead of direct construction

**Methods:**

```cpp
KernelConfig get_config(const std::string& kernel_name, size_t problem_size);
```
- Returns optimal configuration for the given kernel and problem size
- Thread-safe
- Throws `std::runtime_error` if mode is "cache_only" and no cache exists
- `kernel_name` - Identifier for the kernel
- `problem_size` - Size of the problem (used for cache lookup)

```cpp
void tune_kernel(const std::string& kernel_name,
                 std::function<void(const KernelConfig&)> kernel,
                 size_t problem_size);
```
- Benchmarks and stores optimal configuration for the kernel
- Thread-safe
- Throws `std::invalid_argument` if kernel_name is not in registry
- `kernel_name` - Identifier for the kernel
- `kernel` - Function that launches the kernel with given config
- `problem_size` - Size of the problem for benchmarking

```cpp
void save_cache(const std::string& filepath);
```
- Persists current cache to JSON file
- Thread-safe
- `filepath` - Path to output cache file

```cpp
void load_cache(const std::string& filepath);
```
- Loads cache from JSON file
- Thread-safe
- Throws `std::runtime_error` if file cannot be opened or parsed
- `filepath` - Path to input cache file

```cpp
void set_cache_config(const std::string& kernel_name, const KernelConfig& config);
```
- Test helper to directly add configuration to cache
- Thread-safe
- `kernel_name` - Identifier for the kernel
- `config` - Configuration to store

```cpp
void reset();
```
- Test helper to reset tuner to initial state
- Clears cache and re-reads environment variables
- Thread-safe
- Cache directory is resolved from current environment after reset

**Static Members:**
- `static const KernelConfig kDefaultConfigs[]` - Fallback default configurations
- `static const size_t kNumDefaultConfigs` - Number of default configurations

**Deleted Members:**
- Copy constructor and assignment operator deleted
- Move constructor and assignment operator deleted

### Free Functions

```cpp
std::string get_device_id();
```
- Returns unique GPU identifier (PCI bus ID)
- Returns "unknown" if device detection fails

```cpp
std::string get_device_architecture();
```
- Returns GPU compute capability (e.g., "sm_80" for CUDA, "gfx90a" for HIP)
- Returns "unknown" if device detection fails

## Build Configuration

The auto-tuning infrastructure is controlled by the `OpenPFC_ENABLE_GPU_AUTOTUNING` CMake option. When enabled, the `OpenPFC_ENABLE_GPU_AUTOTUNING` preprocessor definition is set, allowing the header-only implementation to be compiled into the project.
