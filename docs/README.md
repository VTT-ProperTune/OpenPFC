<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# OpenPFC documentation (repository)

This directory holds extra guides and assets. **Installing and building** the project is documented in the repository root: **[`INSTALL.md`](../INSTALL.md)** (supported toolchains, HeFFTe **2.4.1**, MPI, CUDA/HIP).

## New users — start here

1. **[`quickstart.md`](quickstart.md)** — install → run **examples** or an **app** → or link the library from your own CMake project.
2. **[`getting_started/README.md`](getting_started/README.md)** — hub for tutorials and links to the **examples catalog** and **applications** pages.
3. **[`extending_openpfc/README.md`](extending_openpfc/README.md)** — how to add models, modifiers, and config-driven apps without forking the core library.
4. **[`faq.md`](faq.md)** — short answers (MPI, CMake, missing binaries, `find_package`).

## Quick links

| Topic | Document |
|--------|-----------|
| Architecture (kernel / runtime / frontend) | [`architecture.md`](architecture.md) |
| CPU vs GPU build directories | [`build_cpu_gpu.md`](build_cpu_gpu.md) |
| Halo exchange (FD vs FFT-safe layouts) | [`halo_exchange.md`](halo_exchange.md) |
| Profiling (runtime session, export formats) | [`performance_profiling.md`](performance_profiling.md) |
| Profiling JSON/HDF5 schema | [`profiling_export_schema.md`](profiling_export_schema.md) |
| Debugging (Debug builds, NaN checks) | [`debugging.md`](debugging.md) |
| Code style | [`styleguide.md`](styleguide.md) |
| LUMI-G (HIP / ROCm / Cray) | [`INSTALL.LUMI.md`](INSTALL.LUMI.md) |
| LUMI Slurm / tungsten performance jobs | [`lumi_slurm/README.md`](lumi_slurm/README.md) |
| Runnable `examples/` executables | [`examples_catalog.md`](examples_catalog.md) |
| Shipped `apps/` programs | [`applications.md`](applications.md) |
| FAQ (build, run, extending) | [`faq.md`](faq.md) |
| Extending models and the UI pipeline | [`extending_openpfc/README.md`](extending_openpfc/README.md) |

## Tutorials

| Section | Document |
|---------|-----------|
| Getting started hub | [`getting_started/README.md`](getting_started/README.md) |
| World, decomposition, FFT, CMake “hello” | [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md) |
| Functional IC/BC (`field::apply`, …) | [`getting_started/functional_field_ops.md`](getting_started/functional_field_ops.md) |

## API examples (Doxygen)

C++ snippets under [`api/examples/`](api/examples/) are included in the Doxygen build (see [`CMakeLists.txt`](CMakeLists.txt)).

## Other

- **Scalability analysis plan:** [`scalability_analysis_plan.md`](scalability_analysis_plan.md) points to the experiment-package location when that tree is available in your checkout.
- **Image / branding notes** for artwork: [`image-prompts.md`](image-prompts.md).

## Generated HTML (Doxygen)

With **`OpenPFC_BUILD_DOCUMENTATION=ON`**, configure and build the `docs` target; HTML output is produced under the build tree (see root [`README.md`](../README.md) and [`docs/CMakeLists.txt`](CMakeLists.txt)).
