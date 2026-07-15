<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# GPU Kernel Auto-Tuning

## Overview

OpenPFC provides a GPU kernel auto-tuning framework that automatically selects optimal kernel launch parameters for your specific GPU architecture and problem sizes. This framework replaces hardcoded kernel launch parameters with runtime-optimized configurations, enabling portable, high-performance GPU execution across diverse GPU systems from desktop GPUs to exascale clusters like LUMI.

## Architecture

The auto-tuner uses a singleton pattern (`pfc::gpu::AutoTuner::instance()`) to ensure a single shared instance across all kernel launch sites. This centralizes tuning logic and avoids duplicate device detection and cache management.

The cache directory is resolved dynamically from the `OPENPFC_GPU_AUTOTUNE_CACHE_DIR` environment variable each time it is accessed, enabling proper test isolation and runtime reconfiguration.

## Environment Variables

- `OPENPFC_GPU_AUTOTUNE_CACHE_DIR` - Directory path for storing cache files (default: current directory)
- `OPENPFC_GPU_AUTOTUNE_MODE` - Tuning mode: "auto", "cache_only", or "fallback_only" (default: "auto")
- `OPENPFC_GPU_AUTOTUNE_DISABLE` - Set to "1" to disable auto-tuning and use hardcoded defaults

## Cache File Format

Cache files are stored as JSON with the following structure:

```json
{
  "device_id": "0000:07:00.0",
  "device_architecture": "sm_80",
  "kernels": {
    "add_scalar": {
      "block_size_x": 256,
      "block_size_y": 1,
      "block_size_z": 1,
      "shared_memory_bytes": 0,
      "min_grid_size": 0,
      "max_grid_size": 0,
      "problem_size_range": [1000, 10000000]
    }
  }
}
```

## Usage

The auto-tuner is automatically used in GPU kernels when `OpenPFC_ENABLE_GPU_AUTOTUNING` is enabled. No code changes are required for basic usage.

## Tuning Modes

- **auto**: Perform tuning if no cache exists, otherwise use cached values
- **cache_only**: Only use cached configurations; throws if cache not found
- **fallback_only**: Never use cache; always use hardcoded fallback defaults

## Fallback Defaults

When cache is unavailable or disabled, the following defaults are used:

- `add_scalar`: 256 threads
- `multiply_scalar`: 256 threads
- `for_each_interior_3d`: 32×4×1 threads
- `gather`: 256 threads
- `scatter`: 256 threads

## Advanced Usage

For custom kernels, you can use the `pfc::gpu::AutoTuner` singleton directly:

```cpp
#include "openpfc/runtime/common/gpu_autotune.hpp"

// Get configuration for a problem size
auto config = pfc::gpu::AutoTuner::instance().get_config("my_kernel", problem_size);
dim3 block(config.block_size_x, config.block_size_y, config.block_size_z);

// Or perform tuning
auto& tuner = pfc::gpu::AutoTuner::instance();
tuner.tune_kernel("my_kernel", [&](const pfc::gpu::KernelConfig& cfg) {
  my_kernel<<<grid, dim3(cfg.block_size_x, cfg.block_size_y, cfg.block_size_z)>>>(...);
}, problem_size);
```

## Build Configuration

To enable GPU auto-tuning, build OpenPFC with the `OpenPFC_ENABLE_GPU_AUTOTUNING` CMake option set to ON. This option must be configured at build time as it controls the preprocessor definition used throughout the codebase.
