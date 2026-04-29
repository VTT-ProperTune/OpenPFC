<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: GPU-enabled builds and `App` programs

This page ties together **CUDA** or **HIP** OpenPFC builds, **HeFFTe** with device FFT, and **JSON/TOML** that selects a GPU backend. Read **[`INSTALL.md`](../../INSTALL.md)** (CUDA/HIP sections) and **[`../build_cpu_gpu.md`](../build_cpu_gpu.md)** first; use **[`../class_tour.md`](../class_tour.md)** for where **`CpuFft`** vs device FFT types sit in headers.

## When you need a GPU build

- You want **`tungsten_cuda`**, **`tungsten_hip`**, **`allen_cahn_cuda`**, **`allen_cahn_hip`**, or CUDA/HIP code paths in your own target.
- You will set **`plan_options`** / **`backend`** to **`cuda`** or ROCm in a config file (**[`examples/fft_backend_selection.toml`](../../examples/fft_backend_selection.toml)**).

CPU-only workflows do **not** require this tutorial—use the default CPU tree and **`tungsten`** / **`examples/`** as in **[`../quickstart.md`](../quickstart.md)**.

## Prerequisites (summary)

- **Separate build directory** per variant (e.g. **`build-gpu`** vs **`build-cpu`**) — see **[`../build_cpu_gpu.md`](../build_cpu_gpu.md)**.
- **HeFFTe** built with the matching backend (e.g. **`…-cuda`** prefix on **`CMAKE_PREFIX_PATH`**), per **[`INSTALL.md`](../../INSTALL.md)** §3.
- **CUDA toolkit** or **ROCm** on **`PATH`**, and **`CMAKE_CUDA_ARCHITECTURES`** (CUDA) matching your hardware.

## Configure and build (illustrative)

CUDA (adjust prefix and architecture):

```bash
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cuda:$CMAKE_PREFIX_PATH
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOpenPFC_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=native \
      -S . -B build-gpu
cmake --build build-gpu -j"$(nproc)"
```

HIP uses **`-DOpenPFC_ENABLE_HIP=ON`** and ROCm-aware HeFFTe; see **[`INSTALL.md`](../../INSTALL.md)** §9 and **[`INSTALL.LUMI.md`](../INSTALL.LUMI.md)** for Cray/ROCm clusters.

## Running shipped apps

From **`build-gpu/`**, use the GPU binary with the **same JSON** layout as CPU, but ensure the file requests the right **`backend`** in **`plan_options`** (mirror **`fft_backend_selection.toml`**).

```bash
cd build-gpu
mpirun -n 4 ./apps/tungsten/tungsten_cuda ../apps/tungsten/inputs_json/tungsten_single_seed.json
```

**Binary names** and availability: **[`../applications.md`](../applications.md)**. If **`tungsten_cuda`** is missing, the configure step did not enable CUDA or did not find the toolkit.

## GPU-aware MPI (optional, clusters)

Device buffers may require **GPU-aware MPI** and environment variables (e.g. **`MPICH_GPU_SUPPORT_ENABLED=1`** on some Cray stacks). **`App`** can log hints when compile-time support is enabled — see **`include/openpfc/frontend/ui/app.hpp`** and **[`INSTALL.LUMI.md`](../INSTALL.LUMI.md)**.

## Your own `App` project

The CMake pattern in **[`custom_app_minimal.md`](custom_app_minimal.md)** is unchanged: link **`OpenPFC`** built with GPU options. Consumers must use a **GPU-enabled** install prefix and compatible **`nlohmann_json`**. Validation and **`model.params`** behave like CPU (**[`../parameter_validation.md`](../parameter_validation.md)**).

## See also

- **[`../example_run_output.md`](../example_run_output.md)** — log shape for **`App`** runs  
- **[`../configuration.md`](../configuration.md)** — config sections  
- **[`../troubleshooting.md`](../troubleshooting.md)** — FFT / MPI / device issues  
