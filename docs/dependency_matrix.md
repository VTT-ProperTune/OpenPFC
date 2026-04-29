<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Dependency and toolchain matrix

Single-page summary of **what this repository is tested against** vs **optional / site-specific** stacks. Authoritative install steps: [`INSTALL.md`](../INSTALL.md). CMake switches: [`build_options.md`](build_options.md).

## Core build

| Component | Reference / CI baseline | Notes |
|-----------|-------------------------|--------|
| **CMake** | 3.15+ | |
| **C++ standard** | C++17 | |
| **GCC** | 11.2.x (modules, tohtori preset) | Other compilers may work; match MPI/HeFFTe. |
| **MPI** | Open MPI **4.1.1** (reference) | Build and run with the **same** implementation ([`INSTALL.md`](../INSTALL.md)). MPICH-only sites: build *everything* with that stack, not mixed with Open MPI. |
| **HeFFTe** | **2.4.1** (CPU / CUDA / ROCm variants) | Separate install per backend; `CMAKE_PREFIX_PATH` ([`INSTALL.md`](../INSTALL.md) §3). Optional FetchContent vendoring via CMake. |
| **FFTW** | Via HeFFTe / system | CPU spectral FFT path. |
| **Catch2** | Required when `OpenPFC_BUILD_TESTS=ON` | [`testing.md`](testing.md). |

## Optional / GPU

| Component | When | Notes |
|-----------|------|--------|
| **CUDA** | `OpenPFC_ENABLE_CUDA` | Match HeFFTe CUDA build and driver. |
| **ROCm / HIP** | `OpenPFC_ENABLE_HIP` | LUMI-G: [`INSTALL.LUMI.md`](INSTALL.LUMI.md). |
| **HDF5** | Profiling / some exports | e.g. `OpenPFC_ENABLE_HDF5` where used ([`build_options.md`](build_options.md)). |
| **GPU-aware MPI** | Device FFT / large GPU jobs | Site-dependent; see [`INSTALL.LUMI.md`](INSTALL.LUMI.md), [`mpi_io_layout_checklist.md`](mpi_io_layout_checklist.md). |

## Documentation / QA tools

| Tool | Role |
|------|------|
| **clang-format** | Pre-commit / CI formatting ([`styleguide.md`](styleguide.md)). |
| **REUSE** | License header compliance (CI). |
| **`scripts/check_doc_links.py`** | Relative markdown links in `docs/`, `README.md`, `INSTALL.md`, etc. |
| **`scripts/check_examples_catalog.py`** | `examples/CMakeLists.txt` vs [`examples_catalog.md`](examples_catalog.md). |
| **`scripts/check_end_to_end_allen_cahn.py`** | Tutorial vs [`apps/allen_cahn/README.md`](../apps/allen_cahn/README.md) example command. |
| **`scripts/check_doc_bash_syntax.py`** | `bash -n` on ` ```bash` / ` ```sh` fenced blocks under `docs/`. |
| **`scripts/build_handbook.sh`** | Optional Pandoc concatenation — [`handbook_build.md`](handbook_build.md). |

## See also

- [`INSTALL.md`](../INSTALL.md) — full procedure  
- [`build_cpu_gpu.md`](build_cpu_gpu.md) — separate build trees  
- [`troubleshooting.md`](troubleshooting.md) — configure/run fixes  
