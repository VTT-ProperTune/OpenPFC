<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# CMake options reference

High-level switches for OpenPFC. Defaults are chosen for a normal **MPI + HeFFTe** developer build. Full context (toolchains, CUDA/HIP, HeFFTe layout) remains in **[`INSTALL.md`](../INSTALL.md)**.

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
| **`OpenPFC_ENABLE_HEFFTE`** | ON | Distributed FFT via HeFFTe; **OFF is unsupported** (configure fails). |
| **`OpenPFC_ENABLE_CUDA`** | OFF | CUDA runtime and **`tungsten_cuda`**, **`allen_cahn_cuda`**, etc. |
| **`OpenPFC_ENABLE_HIP`** | OFF | ROCm/HIP and **`tungsten_hip`**, **`allen_cahn_hip`**, etc. |
| **`OpenPFC_ENABLE_HDF5`** | OFF | HDF5 export for profiling dumps (see **`performance_profiling.md`**) |
| **`OpenPFC_FETCH_HEFFTE`** | (see CMake) | Fetch/build HeFFTe via CMake when not found (see **`INSTALL.md`**) |
| **`OpenPFC_ENABLE_NAN_CHECK`** | OFF | NaN checks beyond Debug (see **`debugging.md`**) |
| **`OpenPFC_ENABLE_CODE_COVERAGE`** | ON where supported | Coverage targets; often OFF on clusters ( **`INSTALL.md`**) |

GPU-aware MPI toggles and CUDA/HIP compiler discovery are described in **`INSTALL.md`** and **`INSTALL.LUMI.md`**.

## Library and profiling

| Setting | Meaning |
|---------|---------|
| **`BUILD_SHARED_LIBS`** | OFF = static **`libopenpfc`** (typical); ON = shared. |
| **`OpenPFC_PROFILING_LEVEL`** | `0` / `1` / `2` — compile-time stripping of **`OPENPFC_PROFILE`** macros (`cmake/LibraryConfiguration.cmake`). |

## Development

| Option | Default | Meaning |
|--------|---------|---------|
| **`OpenPFC_DEVELOPMENT`** | OFF | When ON, enables **`compile_commands.json`** export and dev version suffix. |
| **`OpenPFC_ENABLE_ADDRESS_SANITIZER`** | OFF | Sanitizers (see **`CompilerSettings.cmake`**, **`INSTALL.md`** `dev-asan` preset). |

## Presets

**[`CMakePresets.json`](../CMakePresets.json)** and **`cmake/toolchains/tohtori-gcc11-openmpi.cmake`** pin cluster-friendly configure flags; see **`INSTALL.md`** (“VS Code / Cursor on tohtori”).

## See also

- **[`quickstart.md`](quickstart.md)** — minimal configure line  
- **[`troubleshooting.md`](troubleshooting.md)** — when configure fails  
