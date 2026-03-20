<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Installing OpenPFC

This document is the **supported** source build guide. OpenPFC is **routinely tested** with **GCC 11.2.0** in a module environment. Other compilers may work, but if something breaks, try matching this stack first.

## 1. Environment modules (recommended on clusters)

Load a recent GCC, OpenMPI, and (for GPU) CUDA **before** configuring anything:

```bash
module load gcc/11.2.0
module load openmpi          # e.g. openmpi/4.1.1 — use `module avail openmpi` on your site
module load cuda/12.9        # for GPU — run `module avail cuda` and pick a version where `nvcc --version` works
```

Verify:

```bash
g++ --version    # expect 11.2.x when using gcc/11.2.0
mpicc --version
nvcc --version   # after loading CUDA, for GPU builds
```

**Important:** CMake may still pick `/usr/bin/gcc` if it was run before modules were loaded or if the cache is stale. HeFFTe must be built and consumed with the **same** toolchain. After loading `gcc/11.2.0`, set compilers explicitly when configuring OpenPFC (and when building HeFFTe):

```bash
export CC=$(which gcc)
export CXX=$(which g++)
```

(or pass `-DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++)` to `cmake`.)

**Stale CMake cache:** If an earlier configure picked the wrong compiler, `build-*/CMakeCache.txt` may still point at `/usr/bin/gcc`. Remove the build directory and reconfigure, or pass `-DCMAKE_C_COMPILER` and `-DCMAKE_CXX_COMPILER` explicitly on every `cmake` invocation.

**CUDA note:** Load a `cuda` module that points at an **installed** toolkit (check with `nvcc --version` after loading). If your site’s `cuda/13` (or similar) sets `PATH` but `nvcc` is missing, pick another module version that matches a real install (e.g. `module avail cuda`), or set `-DCMAKE_CUDA_COMPILER=/path/to/nvcc` / `CUDAToolkit_ROOT` explicitly.

## 2. Other dependencies

- **CMake** 3.15+
- **FFTW** (development packages) — required for HeFFTe’s CPU backend
- **nlohmann/json** — optional system install; otherwise CMake may fetch it during configuration
- **MPI** — OpenMPI is commonly used; CMake must find `MPI_CXX` (wrappers from the `openmpi` module are usually enough)
- **toml++** and **Catch2** — if not found on the system, CMake may download them (e.g. when building tests). Ensure network access on first configure, or install/provide packages your site supports.
- **Doxygen** — optional; if present, documentation generation is enabled by default (disable with `-DOpenPFC_BUILD_DOCUMENTATION=OFF` if you do not need it)

## 3. Build and install HeFFTe (required)

OpenPFC **does not** download HeFFTe. You must build and install [HeFFTe](https://github.com/icl-utk-edu/heffte) yourself.

- **Releases:** <https://github.com/icl-utk-edu/heffte/releases> (current recommended: **v2.4.1**)
- **Upstream install guide:** <https://icl-utk-edu.github.io/heffte/md_doxygen_installation.html>

Example: install with **FFTW + CUDA** under `$HOME/opt/heffte/2.4.1` (suitable for both CPU and GPU OpenPFC builds). Load `gcc/11.2.0`, `openmpi`, and **`cuda`** first so `nvcc` is available. Use the **same** `CC`/`CXX` as for OpenPFC (see §1).

```bash
export VER=2.4.1
export CC=$(which gcc)
export CXX=$(which g++)
wget -q https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${VER}.tar.gz
tar xf v${VER}.tar.gz
cmake -S heffte-${VER} -B heffte-${VER}-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_INSTALL_PREFIX=$HOME/opt/heffte/${VER} \
  -DHeffte_ENABLE_FFTW=ON \
  -DHeffte_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build heffte-${VER}-build -j"$(nproc)"
cmake --install heffte-${VER}-build
```

Replace `80` with your GPU’s compute capability, use `native` with CMake 3.24+, or pass several architectures at once (e.g. `75;80;86`) if the install must run on multiple GPU generations.

For **CPU-only** clusters, omit CUDA flags (still set `CC`/`CXX`):

```bash
cmake -S heffte-${VER} -B heffte-${VER}-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_INSTALL_PREFIX=$HOME/opt/heffte/${VER} \
  -DHeffte_ENABLE_FFTW=ON \
  -DHeffte_ENABLE_CUDA=OFF
```

Point CMake at the installation (adjust if your site uses `lib64`):

```bash
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1:$CMAKE_PREFIX_PATH
# or, explicitly:
# cmake ... -DHeffte_DIR=$HOME/opt/heffte/2.4.1/lib64/cmake/Heffte
```

Use the directory that contains `HeffteConfig.cmake` (`lib` vs `lib64` depends on the HeFFTe install).

## 4. Get OpenPFC sources

```bash
git clone https://github.com/VTT-ProperTune/OpenPFC.git
cd OpenPFC
```

## 5. Configure and build OpenPFC (CPU)

With modules + `CMAKE_PREFIX_PATH` set as above, pass the **same** compilers as for HeFFTe:

```bash
export CC=$(which gcc)
export CXX=$(which g++)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -S . -B build-cpu
cmake --build build-cpu -j"$(nproc)"
```

**Minimal configure (optional):** By default, OpenPFC may enable code coverage and (if Doxygen is installed) documentation. For binaries only:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DOpenPFC_ENABLE_CODE_COVERAGE=OFF \
      -DOpenPFC_BUILD_DOCUMENTATION=OFF \
      -S . -B build-cpu
```

## 6. Configure and build OpenPFC (CUDA)

**HeFFTe requirement:** GPU OpenPFC needs HeFFTe built with **`-DHeffte_ENABLE_CUDA=ON`**. Use the same install prefix as in §3 and keep `CMAKE_PREFIX_PATH` set.

Load the **CUDA** module so `nvcc` is on `PATH`, then configure with explicit host compilers:

```bash
export CC=$(which gcc)
export CXX=$(which g++)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DOpenPFC_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -S . -B build-gpu
cmake --build build-gpu -j"$(nproc)"
```

Match `CMAKE_CUDA_ARCHITECTURES` to your GPU (or `native` on CMake 3.24+). To find your GPU's compute capability, run:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. 8.6 → use 86
```

**Minimal configure (optional):** As with the CPU build (§5), you can disable optional features:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DOpenPFC_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DOpenPFC_ENABLE_CODE_COVERAGE=OFF \
      -DOpenPFC_BUILD_DOCUMENTATION=OFF \
      -S . -B build-gpu
```

**If CUDA is missing:** If you pass `-DOpenPFC_ENABLE_CUDA=ON` but CMake cannot find the CUDA toolkit, configuration **still succeeds** with a **warning** and CUDA support is turned **off**. Always check the configuration summary: `OpenPFC_ENABLE_CUDA` should be **ON** for a true GPU build. Fix `PATH`, `CMAKE_CUDA_COMPILER`, or `CUDAToolkit_ROOT`, then reconfigure from a clean build directory.

## 7. Install prefix (optional)

```bash
export CC=$(which gcc)
export CXX=$(which g++)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DCMAKE_INSTALL_PREFIX=$HOME/opt/openpfc \
      -S . -B build
cmake --build build -j"$(nproc)"
cmake --install build
```

(Keep `CMAKE_PREFIX_PATH` including HeFFTe when running this `cmake`.)

## 8. Alternative: Nix

For a reproducible environment (including HeFFTe), see [nix/README.md](nix/README.md).

## 9. AMD GPU (HIP)

For ROCm / HIP builds, load a recent GCC, OpenMPI, and **ROCm** before configuring anything (see §1 for compiler notes). Many clusters provide a ROCm module:

```bash
module load gcc/11.2.0
module load openmpi          # e.g. openmpi/4.1.1
module load rocm/6.4.0       # for GPU — run `module avail rocm` and pick a version
```

If ROCm is not in a module, ensure its bin directory is on `PATH` (e.g. `export PATH=/opt/rocm-6.4.0/bin:$PATH`).

Verify:

```bash
g++ --version
mpicc --version
hipcc --version   # after loading ROCm or setting PATH
rocm-smi          # optional: list AMD GPUs
```

**CMAKE_PREFIX_PATH for ROCm:** CMake finds HIP via `find_package(HIP)`. If HIP is not found, set `CMAKE_PREFIX_PATH` to your ROCm installation (e.g. `-DCMAKE_PREFIX_PATH=/opt/rocm` or `/opt/rocm-6.4.0`) so that `HIPConfig.cmake` is found.

### 9.1. Build and install HeFFTe with ROCm

OpenPFC GPU (HIP) needs HeFFTe built with **`-DHeffte_ENABLE_ROCM=ON`**. Use the same host compilers as for OpenPFC (§1). Example: install with **FFTW + ROCm** under `$HOME/opt/heffte/2.4.1-rocm`. Ensure ROCm is on `PATH` and, if needed, set `CMAKE_PREFIX_PATH` so HeFFTe can find rocFFT/HIP.

```bash
export VER=2.4.1
export CC=$(which gcc)
export CXX=$(which g++)
# Ensure ROCm is on PATH; optionally:
export CMAKE_PREFIX_PATH=/opt/rocm-6.4.0:$CMAKE_PREFIX_PATH
wget -q https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${VER}.tar.gz -O v${VER}.tar.gz
tar xf v${VER}.tar.gz
cmake -S heffte-${VER} -B heffte-${VER}-build-rocm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_INSTALL_PREFIX=$HOME/opt/heffte/${VER}-rocm \
  -DHeffte_ENABLE_FFTW=ON \
  -DHeffte_ENABLE_ROCM=ON
cmake --build heffte-${VER}-build-rocm -j"$(nproc)"
cmake --install heffte-${VER}-build-rocm
```

Optionally set **`-DCMAKE_HIP_ARCHITECTURES=<arch>`** to match your GPU (e.g. `gfx90a` for MI210, `gfx1100` for some RDNA3). Use `rocm-smi` or your vendor docs to get the architecture code.

Point CMake at this installation when building OpenPFC (see §3 for `lib` vs `lib64`):

```bash
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-rocm:$CMAKE_PREFIX_PATH
```

### 9.2. Configure and build OpenPFC (HIP)

Load the **ROCm** module (or set `PATH`) so `hipcc` and HIP are available. Set **`CMAKE_PREFIX_PATH`** to include both the HeFFTe ROCm install and your ROCm installation, so OpenPFC can find HeFFTe and `find_package(HIP)` succeeds:

```bash
export CC=$(which gcc)
export CXX=$(which g++)
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-rocm:/opt/rocm-6.4.0:$CMAKE_PREFIX_PATH
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DOpenPFC_ENABLE_HIP=ON \
      -S . -B build-hip
cmake --build build-hip -j"$(nproc)"
```

Adjust the ROCm path in `CMAKE_PREFIX_PATH` if your install is elsewhere (e.g. `/opt/rocm`). Optionally add **`-DCMAKE_HIP_ARCHITECTURES=<arch>`** to match your AMD GPU.

**Minimal configure (optional):** To disable code coverage and documentation (avoids gcov link issues with the HIP toolchain):

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
      -DOpenPFC_ENABLE_HIP=ON \
      -DOpenPFC_ENABLE_CODE_COVERAGE=OFF \
      -DOpenPFC_BUILD_DOCUMENTATION=OFF \
      -S . -B build-hip
```

**If HIP is not found:** If you pass `-DOpenPFC_ENABLE_HIP=ON` but CMake does not find HIP, configuration can still succeed with a **warning** and HIP will be disabled. Check the configuration summary and ensure `CMAKE_PREFIX_PATH` includes the ROCm installation so that `HIPConfig.cmake` is found. Then reconfigure from a clean build directory if needed.

CMake will warn if HeFFTe was built without ROCm support when HIP is enabled; use the HeFFTe install from §9.1.

**Code coverage:** If the HIP build fails at link with undefined `__gcov_*` symbols, disable code coverage (e.g. `-DOpenPFC_ENABLE_CODE_COVERAGE=OFF`); coverage is not always compatible with the HIP/Clang toolchain.

**toml++ and ROCm headers:** If you see a preprocessor error in toml++ about `__has_attribute` requiring an identifier, it is due to ROCm headers defining the `__noinline__` macro. As a workaround, ensure translation units that use both OpenPFC (with HIP) and toml++ include the toml-based headers before any OpenPFC or HIP includes, or try a different ROCm version.

## Compiler notes

- **GCC 11.2.0** is the primary tested toolchain.
- Older GCC (e.g. 8.x) may need extra link flags for `std::filesystem`; OpenPFC’s CMake links `libstdc++fs` automatically for GCC versions older than 9 when using GNU.
