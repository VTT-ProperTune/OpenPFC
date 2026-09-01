<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# CMake options reference

High-level switches for OpenPFC. Defaults are chosen for a normal **MPI + HeFFTe** developer build. Full context (toolchains, CUDA/HIP, HeFFTe layout) remains in **[`INSTALL.md`](../../INSTALL.md)**.

## What to build

| Option | Default | Meaning |
|--------|---------|---------|
| **`OpenPFC_BUILD_APPS`** | ON | **`apps/`** (tungsten, aluminumNew, allen_cahn, …) |
| **`OpenPFC_BUILD_EXAMPLES`** | ON | **`examples/`** executables |
| **`OpenPFC_BUILD_TESTS`** | ON | **`tests/`** + Catch2 |
| **`OpenPFC_BUILD_BENCHMARKS`** | OFF | Extra benchmarks under **`tests/benchmarks/`** (slow) |
| **`OpenPFC_BUILD_DOCUMENTATION`** | ON | Doxygen **`docs`** target (when Doxygen available) |

Defined in **`cmake/BuildOptions.cmake`** and **`cmake/Dependencies.cmake`**.

## Features and dependencies

| Option | Default | Meaning |
|--------|---------|---------|
| **`OpenPFC_ENABLE_MPI`** | ON | MPI required for supported builds; **OFF is unsupported** — see **`INSTALL.md`**. |
| **`OpenPFC_ENABLE_HEFFTE`** | ON | Distributed FFT via HeFFTe. **OFF** is an FD-only / kernel-only build: no HeFFTe find, no `fft.cpp`, no spectral apps/examples, and the Catch2 tree under `tests/` is skipped (many TUs include `fft_fftw.hpp`). FD apps (`heat3d_fd*`, `kobayashi_fd*`, `wave2d`, `allen_cahn`) still build. `./scripts/build.sh --cpu --no-heffte` is the script path. |
| **`OpenPFC_ENABLE_CUDA`** | OFF | CUDA toolkit/runtime support and CUDA apps that do not require spectral FFTs (for example FD/kernels-only apps). CUDA spectral targets are enabled only when CUDA HeFFTe is also found (`OpenPFC_ENABLE_CUDA_SPECTRAL=ON` in the configure summary). |
| **`OpenPFC_ENABLE_HIP`** | OFF | ROCm/HIP and **`tungsten_hip`**, **`allen_cahn_hip`**, etc. |
| **`OpenPFC_ENABLE_HDF5`** | OFF | HDF5 export for profiling dumps (see **`performance_profiling.md`**) |
| **`OpenPFC_FETCH_HEFFTE`** | (see CMake) | Fetch/build HeFFTe via CMake when not found (see **`INSTALL.md`**) |
| **`OpenPFC_ENABLE_NAN_CHECK`** | OFF | NaN checks beyond Debug (see **`debugging.md`**). When active (Debug or this option), `NAN_CHECK_ENABLED` is PUBLIC on **`OpenPFC::openpfc`**. |
| **`OpenPFC_ENABLE_CODE_COVERAGE`** | ON where supported | Coverage targets; often OFF on clusters ( **`INSTALL.md`**) |

GPU-aware MPI toggles (`OpenPFC_MPI_CUDA_AWARE` / `OpenPFC_MPI_HIP_AWARE`) and CUDA/HIP compiler discovery are described in **`INSTALL.md`** and **`INSTALL.LUMI.md`**. When CUDA or HIP is enabled, `OpenPFC_ENABLE_CUDA` / `OpenPFC_ENABLE_HIP` and the matching MPI-aware flags are **PUBLIC** compile definitions on **`OpenPFC::openpfc`** (and the vendor kernel libraries), so `find_package(OpenPFC)` consumers see the same macros as the in-tree build. `OpenPFC_ENABLE_GPU_AUTOTUNING` is exported the same way when that option is ON.

## Library and profiling

| Setting | Meaning |
|---------|---------|
| **`CMAKE_BUILD_TYPE`** | Single-config default is **RelWithDebInfo** if unset (`cmake/ProjectSetup.cmake`). |
| **`BUILD_SHARED_LIBS`** | OFF = static **`libopenpfc`** (typical); ON = shared. |
| **`OpenPFC_PROFILING_LEVEL`** | `0` / `1` / `2` — compile-time stripping of **`OPENPFC_PROFILE`** macros (`cmake/LibraryConfiguration.cmake`). |

Implementation detail: the `openpfc` library merges CMake **`OBJECT`** targets **`openpfc_kernel_obj`** and **`openpfc_frontend_obj`** (`cmake/LibraryConfiguration.cmake`). Downstream **`find_package(OpenPFC)`** usage is unchanged — link **`OpenPFC::openpfc`** only; the object targets are build internals, not installed.

## Development

| Option | Default | Meaning |
|--------|---------|---------|
| **`OpenPFC_DEVELOPMENT`** | OFF | When ON, enables **`compile_commands.json`** export and dev version suffix. |
| **`OpenPFC_ENABLE_ADDRESS_SANITIZER`** | OFF | Sanitizers (see **`CompilerSettings.cmake`**, **`INSTALL.md`** `dev-asan` preset). |

## Presets

**[`CMakePresets.json`](../../CMakePresets.json)** and **`cmake/toolchains/tohtori-gcc11-openmpi.cmake`** pin cluster-friendly configure flags; see **`INSTALL.md`** (“VS Code / Cursor on tohtori”).

## See also

- **[`quickstart.md`](../quickstart.md)** — minimal configure line  
- **[`testing.md`](../development/testing.md)** — **`OpenPFC_BUILD_TESTS`**, CTest, Catch2 filters  
- **[`troubleshooting.md`](../troubleshooting.md)** — when configure fails  
