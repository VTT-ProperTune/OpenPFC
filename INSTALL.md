<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Installing OpenPFC

This document is the **supported** source build guide. OpenPFC is **routinely tested** with **GCC 11.2.0** in a module environment. Other compilers may work, but if something breaks, try matching this stack first.

### MPI: OpenMPI from modules — do not improvise with another stack

This guide’s **source of truth** for MPI is **OpenMPI provided through environment modules** (`module load openmpi` after `module load gcc/…`), as in §1 below. That matches how OpenPFC and HeFFTe are meant to be built together on clusters and many dev machines.

- **Do not** point CMake at an arbitrary system MPI just because `mpicc` exists (for example **MPICH** under `/usr/lib64/mpich` on RHEL-style systems). **OpenMPI and MPICH are different implementations**; HeFFTe and OpenPFC must be configured, built, and run against the **same** MPI. Mixing “HeFFTe built with MPICH” and “OpenPFC expecting OpenMPI” (or the reverse) produces confusing link or runtime failures.
- **Do** run `module load openmpi` (or your site’s equivalent name — use `module avail openmpi`), then confirm you are using **that** toolchain before building HeFFTe **and** OpenPFC:
  ```bash
  which mpicc
  mpicc --version   # should identify Open MPI / openmpi, not MPICH
  ```
- **Bare OS without modules:** install **Open MPI development** packages (not MPICH) if you want to follow this document literally, **or** use the reproducible [Nix](nix/README.md) workflow in §8. If your site standardizes on MPICH only, you may still build everything consistently with MPICH — but that is **not** the combination this file describes step-by-step; keep HeFFTe and OpenPFC on the **same** `mpicc`/`mpicxx` throughout.

## 1. Environment modules (recommended on clusters)

Load a recent GCC, **OpenMPI**, and (for GPU) CUDA **before** configuring anything:

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

### compile_commands.json (IDEs, clang-tidy, agents)

After modules and compilers are set (§1), configure with **`CMAKE_EXPORT_COMPILE_COMMANDS=ON`** so CMake writes an accurate **`compile_commands.json`** inside the build directory (e.g. `build-cpu/compile_commands.json`). Tools such as **clangd**, **clang-tidy**, and scripted builds depend on this file matching the **same** `mpicc`/GCC the project uses — so always re-run `cmake` **after** `module load`, never from a “clean” shell that falls back to `/usr/bin/gcc` and a different MPI.

```bash
cmake -S . -B build-cpu -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ...   # plus your usual flags (§5)
# Optional: symlink for tools that expect it at the repo root
ln -sf build-cpu/compile_commands.json compile_commands.json
```

To mirror CI static analysis locally (see `.github/workflows/ci.yml`), use the build directory that contains `compile_commands.json` as **`-p`** for `run-clang-tidy` / `clang-tidy`.

### VS Code / Cursor on tohtori (CMake presets)

**CMake Tools** often launches `cmake` without an interactive **Lmod** shell, so the generic **`dev-debug`** preset can pick the OS default compiler (e.g. GCC 8.x) and fail to find **OpenMPI**. On **tohtori**, select the **`tohtori-debug`** or **`tohtori-release`** configure preset in `CMakePresets.json`: they apply **`cmake/toolchains/tohtori-gcc11-openmpi.cmake`** plus `PATH` / `LD_LIBRARY_PATH` matching **`module show gcc/11.2.0`** and **`module show openmpi/4.1.1`**. If **`$HOME/opt/heffte/2.4.1-cpu`** exists, the toolchain prepends it to **`CMAKE_PREFIX_PATH`**. Those presets set **`OpenPFC_ENABLE_CODE_COVERAGE=OFF`** (many cluster images lack **`lcov`**); install **`lcov`** (e.g. **EPEL** + **`dnf install lcov`** on EL8) and reconfigure with **`-DOpenPFC_ENABLE_CODE_COVERAGE=ON`** if you want **`ninja coverage`**.

Optionally create **`.vscode/settings.json`** with `"cmake.configurePreset": "tohtori-debug"` so the folder opens with the right preset (this repo **`.gitignore`** ignores `.vscode` unless you change that). For different paths after a cluster upgrade, use **`CMakeUserPresets.json`** at the repo root — see **`cmake/README.md`**.

## 2. Other dependencies

- **CMake** 3.15+
- **FFTW** (development packages) — required for HeFFTe’s CPU backend
- **nlohmann/json** — optional system install; otherwise CMake may fetch it during configuration
- **MPI** — **Required** for all supported builds (HeFFTe and the distributed FFT stack depend on it). The **documented** setup is **OpenMPI from modules** (see §1 and the MPI callout above). CMake must find `MPI_CXX` for **that** MPI — not a random MPICH install earlier on `PATH`. There is **no supported serial-only** configuration yet — `-DOpenPFC_ENABLE_MPI=OFF` is rejected at configure time with a pointer to this list. If **`mpi.h`** is missing during configure, see **§5.1** (modules + `CC`/`CXX` + stale cache).
- **toml++** and **Catch2** — if not found on the system, CMake may download them (e.g. when building tests). Ensure network access on first configure, or install/provide packages your site supports.
- **Doxygen** — optional; if present, documentation generation is enabled by default (disable with `-DOpenPFC_BUILD_DOCUMENTATION=OFF` if you do not need it)

## 3. HeFFTe (required)

Build and install [HeFFTe](https://github.com/icl-utk-edu/heffte) yourself, then point OpenPFC at it with **`CMAKE_PREFIX_PATH`** or **`Heffte_DIR`**.

If you omit those, CMake still runs **`find_package(Heffte)`** first, then probes common install prefixes automatically (see **`cmake/OpenPFCHeffteHints.cmake`**: e.g. **`$HOME/opt/heffte/*`**, **`/opt/heffte/...`**, **`/share/apps/heffte/...`**, Spack **`EBROOTHEFFTE`**). The first prefix that contains **`lib64/cmake/Heffte`** or **`lib/cmake/Heffte`** wins. Add your site’s root to that file if needed.

- **Releases:** <https://github.com/icl-utk-edu/heffte/releases> (recommended: **v2.4.1**)
- **Upstream install guide:** <https://icl-utk-edu.github.io/heffte/md_doxygen_installation.html>

### Install layout: `$HOME/opt/heffte/<variant>`

Use **one install prefix per backend**, all under **`$HOME/opt/heffte/`**, so you can keep CPU, CUDA, and ROCm builds side by side and wire them in with modules:

| Variant | Typical `CMAKE_INSTALL_PREFIX` | Notes |
|---------|----------------------------------|--------|
| **CPU (FFTW only)** | `$HOME/opt/heffte/2.4.1-cpu` | No GPU backend in HeFFTe |
| **CUDA** | `$HOME/opt/heffte/2.4.1-cuda` | `-DHeffte_ENABLE_CUDA=ON`, `nvcc` on `PATH` when building |
| **ROCm / HIP** | `$HOME/opt/heffte/2.4.1-rocm` | `-DHeffte_ENABLE_ROCM=ON` (see §9.1) |

OpenPFC CPU builds should use **`CMAKE_PREFIX_PATH`** (or `Heffte_DIR`) pointing at **`2.4.1-cpu`**. GPU builds must use the matching **`-cuda`** or **`-rocm`** install.

### Do not unpack or build HeFFTe inside the OpenPFC repository

Keep the OpenPFC clone free of HeFFTe **source trees**, **long-lived build trees**, and **release tarballs** (nothing like `heffte-2.4.1/` or `v2.4.1.tar.gz` next to OpenPFC’s top-level `CMakeLists.txt`).

- Download tarballs to **`$HOME/src`**, **`/tmp`**, or any directory **outside** the OpenPFC tree.
- Point **`-S`** at the extracted source **outside** the repo (e.g. `$HOME/src/heffte-2.4.1`).
- Point **`-B`** at a **build directory outside** the repo (e.g. `$HOME/opt/heffte/build-cpu-2.4.1` or `/tmp/heffte-build-cpu-2.4.1`).
- **`cmake --install`** into **`$HOME/opt/heffte/2.4.1-cpu`** (or `-cuda` / `-rocm`).

OpenPFC’s own **`build-cpu/`** / **`build-gpu/`** directories are only for **OpenPFC** (usually gitignored); do not use the **repository root** as a dump for HeFFTe sources or tarballs.

**GitHub Actions** in this repository follow the same policy: workflows download HeFFTe to **`/tmp`**, build in a temporary directory there, install to **`$HOME/opt/heffte/2.4.1-cpu`** on the runner, and delete the extract — the checked-out OpenPFC tree is not used as a HeFFTe working directory.

### Optional: `OpenPFC_FETCH_HEFFTE=ON`

If no HeFFTe is found, CMake can fetch v2.4.1 via **FetchContent** into **`build/_deps/`** inside an OpenPFC **build** directory (FFTW + MPI still required). This is a fallback only: for day-to-day work, **prefer installs under `$HOME/opt/heffte/<variant>`** as above. FetchContent must **not** be used as an excuse to extract HeFFTe **into the OpenPFC source tree**.

### Optional: Lmod modulefiles (`$HOME/privatemodules`)

After installing to e.g. `$HOME/opt/heffte/2.4.1-cpu`, you can add a personal module that prepends **`CMAKE_PREFIX_PATH`** (and anything else your site needs):

```tcl
#%Module1.0
set root $env(HOME)/opt/heffte/2.4.1-cpu
prepend-path CMAKE_PREFIX_PATH $root
```

Place the file where Lmod looks (e.g. `$HOME/privatemodules/heffte/2.4.1-cpu.lua` or `.modulepath` layout your site uses), then `module load heffte/2.4.1-cpu` before configuring OpenPFC. Repeat with different paths for **`2.4.1-cuda`** and **`2.4.1-rocm`**.

### Build and install HeFFTe manually (outside OpenPFC)

Load **`gcc/11.2.0`**, **`openmpi`**, and (for CUDA) **`cuda`** first (§1). Use the **same** `CC`/`CXX` as for OpenPFC.

**Working copy outside the repo** (adjust `SRC` / `BUILD`):

```bash
export VER=2.4.1
export SRC=$HOME/src/heffte-${VER}
export BUILD=$HOME/opt/heffte/build-cuda-${VER}
mkdir -p "$HOME/src" "$HOME/opt/heffte"
wget -q -O "$HOME/src/v${VER}.tar.gz" \
  https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${VER}.tar.gz
tar xf "$HOME/src/v${VER}.tar.gz" -C "$HOME/src"
export CC=$(which gcc)
export CXX=$(which g++)
```

**CUDA + FFTW** → install to **`$HOME/opt/heffte/2.4.1-cuda`**:

```bash
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_INSTALL_PREFIX=$HOME/opt/heffte/2.4.1-cuda \
  -DHeffte_ENABLE_FFTW=ON \
  -DHeffte_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build "$BUILD" -j"$(nproc)"
cmake --install "$BUILD"
```

Replace `80` with your GPU’s compute capability, use `native` with CMake 3.24+, or pass several architectures (e.g. `75;80;86`).

**CPU only** → install to **`$HOME/opt/heffte/2.4.1-cpu`** (use a separate **`BUILD`** path, e.g. `$HOME/opt/heffte/build-cpu-${VER}`):

```bash
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_INSTALL_PREFIX=$HOME/opt/heffte/2.4.1-cpu \
  -DHeffte_ENABLE_FFTW=ON \
  -DHeffte_ENABLE_CUDA=OFF
cmake --build "$BUILD" -j"$(nproc)"
cmake --install "$BUILD"
```

Point OpenPFC at the install (see `lib` vs `lib64` on your system):

```bash
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cpu:$CMAKE_PREFIX_PATH
# or, explicitly:
# cmake ... -DHeffte_DIR=$HOME/opt/heffte/2.4.1-cpu/lib64/cmake/Heffte
```

Use the directory that contains **`HeffteConfig.cmake`**.

## 4. Get OpenPFC sources

```bash
git clone https://github.com/VTT-ProperTune/OpenPFC.git
cd OpenPFC
```

## 5. Configure and build OpenPFC (CPU)

### 5.1 If `cmake` fails immediately — what is actually blocking?

OpenPFC’s CMake is **not** broken on a correctly prepared machine. Configuration almost always fails for one of these **environment** reasons (all fixable without changing the repo):

| Symptom | Cause | Fix (see earlier sections) |
|--------|--------|----------------------------|
| **`HeFFTe was not found`** / fatal error mentioning `Heffte` | No install on **`CMAKE_PREFIX_PATH`**, and **`OpenPFC_FETCH_HEFFTE=OFF`** (default). | Build and install HeFFTe to **`$HOME/opt/heffte/2.4.1-cpu`** (§3), then `export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cpu:$CMAKE_PREFIX_PATH` **or** pass **`-DHeffte_DIR=$HOME/opt/heffte/2.4.1-cpu/lib64/cmake/Heffte`** (use the directory that contains **`HeffteConfig.cmake`** — sometimes `lib/cmake/Heffte`). Alternatively, **`-DOpenPFC_FETCH_HEFFTE=ON`** (still requires FFTW + MPI; artifacts only under **`build/_deps/`**, not the source tree). |
| **`mpi.h: No such file or directory`** during MPI probe, or MPI from the wrong vendor | Shell never ran **`module load openmpi`** (§1), or CMake used **`/usr/bin/cc`** without OpenMPI **include/lib** flags. | Load **GCC + OpenMPI** modules **first**, then **`export CC=$(which gcc)`** and **`export CXX=$(which g++)`**, then configure — or pass **`-DCMAKE_C_COMPILER=... -DCMAKE_CXX_COMPILER=...`** explicitly. Confirm **`mpicc --version`** reports **Open MPI**, not MPICH (see top callout). |
| Wrong compiler / MPI after you “fixed” modules | **Stale `CMakeCache.txt`** in **`build/`** still points at old paths. | Remove the build directory (**`rm -rf build-cpu`**) and re-run **`cmake`** from a shell **after** modules are loaded. |

**Agents and IDEs:** Do **not** point CMake at a hard-coded MPI under `/share/apps/...` unless that matches **this** machine’s documented modules. Follow **§1** and **§3** every time; the supported recipe is **module-loaded OpenMPI + HeFFTe under `$HOME/opt/heffte/<variant>`**.

---

With modules loaded (§1), **`CMAKE_PREFIX_PATH`** including **`$HOME/opt/heffte/2.4.1-cpu`** (or a `module load` that sets it — §3), pass the **same** compilers as for HeFFTe:

```bash
export CC=$(which gcc)
export CXX=$(which g++)
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cpu:$CMAKE_PREFIX_PATH
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -S . -B build-cpu
cmake --build build-cpu -j"$(nproc)"
```

**Minimal configure (optional):** By default, OpenPFC may enable code coverage and (if Doxygen is installed) documentation. For binaries only:

```bash
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cpu:$CMAKE_PREFIX_PATH
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DOpenPFC_ENABLE_CODE_COVERAGE=OFF \
      -DOpenPFC_BUILD_DOCUMENTATION=OFF \
      -S . -B build-cpu
```

## 6. Configure and build OpenPFC (CUDA)

**HeFFTe requirement:** GPU OpenPFC needs HeFFTe built with **`-DHeffte_ENABLE_CUDA=ON`**, installed under **`$HOME/opt/heffte/2.4.1-cuda`** (§3). Set **`CMAKE_PREFIX_PATH`** to include that prefix (or load a module that does).

Load the **CUDA** module so `nvcc` is on `PATH`, then configure with explicit host compilers:

```bash
export CC=$(which gcc)
export CXX=$(which g++)
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cuda:$CMAKE_PREFIX_PATH
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
export CMAKE_PREFIX_PATH=$HOME/opt/heffte/2.4.1-cuda:$CMAKE_PREFIX_PATH
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

OpenPFC GPU (HIP) needs HeFFTe built with **`-DHeffte_ENABLE_ROCM=ON`**, installed under **`$HOME/opt/heffte/2.4.1-rocm`** (§3). Use the same host compilers as for OpenPFC (§1). **Do not** build inside the OpenPFC repo; use **`$HOME/src`** (or similar) for the tarball and source, and a **build directory outside** the repo.

```bash
export VER=2.4.1
export SRC=$HOME/src/heffte-${VER}
export BUILD=$HOME/opt/heffte/build-rocm-${VER}
mkdir -p "$HOME/src" "$HOME/opt/heffte"
wget -q -O "$HOME/src/v${VER}.tar.gz" \
  https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${VER}.tar.gz
tar xf "$HOME/src/v${VER}.tar.gz" -C "$HOME/src"
export CC=$(which gcc)
export CXX=$(which g++)
# Ensure ROCm is on PATH; optionally:
export CMAKE_PREFIX_PATH=/opt/rocm-6.4.0:$CMAKE_PREFIX_PATH
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_INSTALL_PREFIX=$HOME/opt/heffte/2.4.1-rocm \
  -DHeffte_ENABLE_FFTW=ON \
  -DHeffte_ENABLE_ROCM=ON
cmake --build "$BUILD" -j"$(nproc)"
cmake --install "$BUILD"
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
