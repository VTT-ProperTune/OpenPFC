<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Building OpenPFC on tohtori (GCC + Open MPI + optional CUDA)

This note complements the generic instructions in [INSTALL.md](../../INSTALL.md). It matches the pinned paths in [`cmake/toolchains/tohtori-gcc11-openmpi.cmake`](../../cmake/toolchains/tohtori-gcc11-openmpi.cmake) and the usual VTT **tohtori** module layout (`gcc/11.2.0`, `openmpi/4.1.1` under `/share/apps/…`).

## 1. CPU build (reference)

In an interactive shell:

```bash
module load gcc/11.2.0
module load openmpi/4.1.1
export CC=$(which gcc) CXX=$(which g++)
which mpicc mpicxx   # both should resolve under the same Open MPI prefix
```

Install HeFFTe (FFTW CPU backend) under e.g. `$HOME/opt/heffte/2.4.1-cpu` as in [INSTALL.md](../../INSTALL.md) §3, then configure OpenPFC with that prefix on `CMAKE_PREFIX_PATH` or `-DHeffte_DIR=.../lib64/cmake/Heffte`.

For CMake without loading modules (IDE, batch scripts), use the toolchain file:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/toolchains/tohtori-gcc11-openmpi.cmake \
  ...
```

## 2. CUDA toolkit: load a `cuda` module

**Before** configuring a CUDA-enabled OpenPFC build, load a CUDA module so `nvcc` and the toolkit match your driver and CMake can detect `CUDAToolkit`:

```bash
module load cuda/12.9    # or: module avail cuda   # pick a version where `nvcc --version` works
which nvcc
nvcc --version
```

The site may default to `cuda/13.0` or similar; use `module avail cuda` and choose a version that installs a real toolkit (not only driver stubs).

Without this, CMake may still find `/usr/local/cuda/bin/nvcc`, but your `PATH` and library layout are easier to reason about if they come from one module stack.

## 3. CUDA-capable HeFFTe (required only for CUDA spectral FFT)

OpenPFC links HeFFTe for FFT/decomposition. A **CPU-only** HeFFTe install (e.g. `2.4.1-cpu`) is sufficient for CPU FFTs and for CUDA finite-difference / kernel-only applications that do not use HeFFTe’s cuFFT backend (for example `wave2d_cuda`).

For **CUDA spectral FFT** paths (Tungsten CUDA spectral binaries, CUDA FFT runtime, CUDA FFT tests), HeFFTe must be configured with the CUDA backend (`-DHeffte_ENABLE_CUDA=ON` when building HeFFTe). Install it to a **separate** prefix, e.g.:

```text
$HOME/opt/heffte/2.4.1-cuda
```

If you maintain a private Lmod module (e.g. `heffte/2.4.1-cuda`), it should set `Heffte_DIR` to the CMake package directory:

```text
$HOME/opt/heffte/2.4.1-cuda/lib64/cmake/Heffte
```

and load a **GCC** and **Open MPI** build that matches that HeFFTe (often GCC 12.x + `openmpi/4.1.1-cuda` for GPU-aware MPI).

**Important:** If both CPU and CUDA HeFFTe installs exist under `$HOME/opt/heffte/`, OpenPFC’s autodetection may pick the **CPU** tree first. For CUDA spectral builds, set the package location explicitly on the **first** `cmake` invocation:

```bash
-DHeffte_DIR=$HOME/opt/heffte/2.4.1-cuda/lib64/cmake/Heffte
```

(or set environment variable `HEFFTE_DIR` to that same directory before configuring).

## 4. Example: configure with CUDA

After `module load` (GCC, Open MPI, CUDA), a CUDA finite-difference build can use the CPU HeFFTe prefix and will report `OpenPFC_ENABLE_CUDA_SPECTRAL = OFF` if CUDA HeFFTe is not found. To enable CUDA spectral FFT targets as well, configure with CUDA HeFFTe installed:

```bash
export CC=$(which gcc) CXX=$(which g++)
cmake -S . -B build-gpu \
  -DOpenPFC_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DHeffte_DIR=$HOME/opt/heffte/2.4.1-cuda/lib64/cmake/Heffte \
  -DCMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cuda \
  ...
```

Use `CMAKE_CUDA_ARCHITECTURES` appropriate for your GPU (e.g. `86` for Ampere). CMake 3.22 on this system is supported for CUDA: OpenPFC teaches older CMake releases the `--std=c++20` nvcc flag so CUDA targets can include OpenPFC’s C++20 public headers without requiring native “CUDA20” dialect support from CMake itself.

## 5. wave2d: VTK output and CUDA binary

CPU binaries `wave2d_fd_manual` / `wave2d_fd` and the CUDA app `wave2d_cuda` accept optional VTK flags (see [`apps/wave2d/README.md`](../../apps/wave2d/README.md)):

```bash
mpirun -n 2 ./wave2d_fd_manual 96 96 200 0.02 neumann \
  --vtk $PWD/wave2d-run-out/cpu/u_%04d.vti --vtk-every 50

mpirun -n 2 ./wave2d_cuda 96 96 200 0.02 neumann \
  --vtk $PWD/wave2d-run-out/gpu/u_%04d.vti --vtk-every 50
```

Open the generated `.pvti` files in ParaView (multi-rank) or the single `.vti` if `mpirun -n 1`.

## See also

- [INSTALL.md](../../INSTALL.md) — supported stack, HeFFTe build, troubleshooting  
- [tohtori OpenMPI environment](../../.cursor/rules/tohtori-openmpi-environment.mdc) (Cursor rule: fixed paths when modules are not loaded)  
- [GPU path decision](gpu_path_decision.md) — whether to enable CUDA/HIP at all  
