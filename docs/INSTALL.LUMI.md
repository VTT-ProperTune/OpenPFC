<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Building OpenPFC on LUMI-G (HeFFTe + HIP / ROCm)

This guide complements the generic instructions in [INSTALL.md](../INSTALL.md). It targets **LUMI-G** (AMD MI250X, **gfx90a**), the **Cray programming environment** (`cc` / `CC` wrappers, **cray-mpich**), and the **LUMI/25.09** software stack with **ROCm 6.4.x** from the `partition/G` toolchain.

**Storage layout used here**

- **Install prefixes** (small, long-lived): `/projappl/project_462001245/` — e.g. HeFFTe and OpenPFC installs.
- **Sources and build trees** (large, temporary): `/scratch/project_462001245/$USER/` — tarballs, `build-*`, CMake `FetchContent` downloads.

Adjust paths if your project ID or layout differs.

## 1. Module environment

Load the GPU partition toolchain, GNU compilers, FFTW, and fix runtime library search for the Cray PE (see also the [LUMI programming environment](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/)):

```bash
module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
```

With `partition/G`, Lmod should select **`craype-x86-trento`**, **`craype-accel-amd-gfx90a`**, and **`rocm`** automatically. Verify:

```bash
module list
which hipcc rocm-smi
export CC=cc CXX=CC
cc --version
hipcc --version   # may warn on login nodes without GPUs; that is expected
```

**`rocm-smi`** only works on GPU nodes or in a GPU job.

## 2. MPI include path for HIP translation units

HeFFTe and OpenPFC compile HIP sources that include **MPI** headers. The HIP compiler invoked by CMake does not always receive the same implicit include paths as `CC`, which can produce:

```text
fatal error: 'mpi.h' file not found
```

Point CMake at the **Cray MPICH** headers for the **GNU** binding that matches your loaded stack. On **cpeGNU/25.09** this path was:

```text
/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include
```

**How to rediscover it** after a future PE upgrade:

```bash
CC -E -Wp,-v -xc++ /dev/null 2>&1 | grep mpich
```

Use that directory in **`-DCMAKE_HIP_FLAGS=-I<that>/include`** (or the path printed if it already ends with `include`).

## 3. Build and install HeFFTe (FFTW + ROCm)

Work in scratch; install to **projappl**.

```bash
export SCRATCH=/scratch/project_462001245/$USER
export VER=2.4.1
export MPI_INC=/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include   # adjust if needed (§2)
export CMAKE_PREFIX_PATH="${EBROOTROCM}:${CMAKE_PREFIX_PATH}"

mkdir -p "$SCRATCH/src" "$SCRATCH/build"
cd "$SCRATCH/src"
test -f v${VER}.tar.gz || wget -q https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${VER}.tar.gz
tar xf v${VER}.tar.gz

cmake -S "$SCRATCH/src/heffte-${VER}" -B "$SCRATCH/build/heffte-${VER}-rocm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_INSTALL_PREFIX=/projappl/project_462001245/heffte/${VER}-rocm \
  -DHeffte_ENABLE_FFTW=ON \
  -DHeffte_ENABLE_ROCM=ON \
  -DHeffte_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_FLAGS="-I${MPI_INC}"

cmake --build "$SCRATCH/build/heffte-${VER}-rocm" -j8
cmake --install "$SCRATCH/build/heffte-${VER}-rocm"
```

**`HeffteConfig.cmake`** for this layout:

```text
/projappl/project_462001245/heffte/2.4.1-rocm/lib64/cmake/Heffte
```

You may see CMake dev warnings about **GPU_TARGETS** / **amdgpu-arch** on login nodes; **`CMAKE_HIP_ARCHITECTURES=gfx90a`** still produced **gfx90a** code in verified builds.

**Optional:** add **`-DHeffte_ENABLE_TESTING=OFF`** for a faster build.

**Fallback:** if configuration fails with **GNU**, try **`cpeCray`** or **`cpeAMD`** instead of **`cpeGNU`**, keeping **`partition/G`** and **`rocm`**.

## 4. Build and install OpenPFC (HIP)

Use the same modules as in §1. Point CMake at HeFFTe and ROCm:

```bash
module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath

export CC=cc CXX=CC
export SCRATCH=/scratch/project_462001245/$USER
export MPI_INC=/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include
export CMAKE_PREFIX_PATH="/projappl/project_462001245/heffte/2.4.1-rocm:${EBROOTROCM}:${CMAKE_PREFIX_PATH}"

# OpenPFC source: clone or copy to $SCRATCH or your preferred path
export SRC=/path/to/OpenPFC

cmake -S "$SRC" -B "$SCRATCH/build/openpfc-hip" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_INSTALL_PREFIX=/projappl/project_462001245/openpfc/0.1.4-hip \
  -DOpenPFC_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_FLAGS="-I${MPI_INC}" \
  -DOpenPFC_ENABLE_CODE_COVERAGE=OFF \
  -DOpenPFC_BUILD_DOCUMENTATION=OFF \
  -DOpenPFC_BUILD_TESTS=OFF

cmake --build "$SCRATCH/build/openpfc-hip" -j8
cmake --install "$SCRATCH/build/openpfc-hip"
```

**Check the CMake summary:** HIP enabled, **HeFFTe ROCm backend available** (no warning about rebuilding HeFFTe without ROCm).

**FetchContent:** first configure needs network access (e.g. **nlohmann/json**, **toml++**, **Catch2** if tests are on). Run CMake from a **login node** if compute nodes have no outbound HTTP.

### ROCm library path / RPATH warnings

CMake may warn that **`libamdhip64.so`** under **`/appl/.../rocm/...`** could be **hidden** by **`/opt/rocm-6.3.4`**. Prefer the **module** ROCm at runtime:

```bash
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Loading **`lumi-CrayPath`** after all other modules (as above) follows [LUMI’s recommendation](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/) for **`LD_LIBRARY_PATH`** with the Cray PE.

## 5. Running GPU jobs (Slurm)

Use a GPU partition (e.g. **`small-g`**). For **GPU-aware MPI** (device pointers), set:

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
```

Minimal batch fragment (adjust **account** and **partition** to your project):

```bash
#!/bin/bash
#SBATCH --account=project_462001245
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1

srun /projappl/project_462001245/openpfc/0.1.4-hip/bin/tungsten_hip /path/to/input.json
```

The **`tungsten_hip`** binary is the HIP build of the tungsten application. A short smoke run on **small-g** completed two time steps successfully with the stack above (job output showed MPI init, validation, and stepping).

## 6. Runtime FFT / TOML `backend` field

The TOML/JSON option **`backend = "cuda"`** in examples such as [examples/fft_backend_selection.toml](../examples/fft_backend_selection.toml) applies to **NVIDIA / cuFFT** builds only.

On LUMI-G, GPU FFTs use **rocFFT** via HeFFTe inside **HIP-specific** code paths (e.g. **`tungsten_hip`**, **`create_hip`**). Do not expect a **`"rocfft"`** string in **`backend_from_string`** for generic CPU/CUDA runtime switching; use the HIP-enabled applications and APIs documented in the repository.

## 7. Further reading

- [INSTALL.md](../INSTALL.md) — general HeFFTe and OpenPFC options (CUDA, coverage, documentation).
- [LUMI documentation](https://docs.lumi-supercomputer.eu/) — modules, queues, storage, and PE updates.
