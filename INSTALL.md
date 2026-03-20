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

For ROCm / HIP builds, install HeFFTe with **`-DHeffte_ENABLE_ROCM=ON`** (see the [HeFFTe installation guide](https://icl-utk-edu.github.io/heffte/md_doxygen_installation.html)). Configure OpenPFC with **`-DOpenPFC_ENABLE_HIP=ON`** (and a suitable ROCm environment). CMake will warn if HeFFTe lacks ROCm support when HIP is enabled.

## Compiler notes

- **GCC 11.2.0** is the primary tested toolchain.
- Older GCC (e.g. 8.x) may need extra link flags for `std::filesystem`; OpenPFC’s CMake links `libstdc++fs` automatically for GCC versions older than 9 when using GNU.
