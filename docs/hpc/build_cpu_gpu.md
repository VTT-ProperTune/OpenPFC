<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# CPU vs GPU build trees (quick reference)

Use separate build directories for CPU and CUDA (or HIP) so you never mix object files or cached flags between toolchains.

| Variant | Suggested directory | Defining CMake options |
|--------|---------------------|-------------------------|
| CPU | `build-cpu` | Default: CUDA and HIP off. MPI, FFTW, and HeFFTe (CPU/FFTW) required. |
| CUDA | `build-gpu` | `-DOpenPFC_ENABLE_CUDA=ON`, `-DCMAKE_CUDA_ARCHITECTURES=` matching the GPU (e.g. `86` for compute capability 8.6). HeFFTe must be built with CUDA enabled. |
| HIP | `build-hip` | `-DOpenPFC_ENABLE_HIP=ON`. HeFFTe with ROCm; see [INSTALL.md](../INSTALL.md) §9. |

Why two trees: The CUDA flag changes which translation units and libraries are built; reusing one build dir after toggling GPU options often leaves stale `CMakeCache.txt` entries. A clean configure per variant avoids that.

Authors’ workflow (condensed):

HeFFTe lives under `$HOME/opt/heffte/2.4.1-cpu` or `2.4.1-cuda` (see [INSTALL.md](../INSTALL.md) §3). Point OpenPFC at the right prefix with `CMAKE_PREFIX_PATH` (or a personal module that sets it).

```bash
# CPU (after module load gcc + openmpi, and HeFFTe installed to 2.4.1-cpu)
export CC=$(which gcc)
export CXX=$(which g++)
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cpu:$CMAKE_PREFIX_PATH
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build-cpu
cmake --build build-cpu -j"$(nproc)"

# CUDA (HeFFTe with CUDA at 2.4.1-cuda; nvcc on PATH)
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cuda:$CMAKE_PREFIX_PATH
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOpenPFC_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=native \
      -S . -B build-gpu
cmake --build build-gpu -j"$(nproc)"
```

Full prerequisites (modules, HeFFTe layout, optional FetchContent fallback into `build/` only) are in [INSTALL.md](../INSTALL.md) §3 and §5–§9.

## See also

- [`tutorials/gpu_app_quickstart.md`](tutorials/gpu_app_quickstart.md) — JSON `backend`, `tungsten_cuda`, cluster notes  
- [`example_run_output.md`](example_run_output.md) — what GPU `App` logs look like compared to CPU  
