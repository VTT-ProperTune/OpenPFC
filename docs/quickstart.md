<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Quick start

**Brand new?** If you want the shortest linear path (clone → build → one `mpirun`), use [`start_here_15_minutes.md`](start_here_15_minutes.md) first, then come back here for the full menu of tracks.

This page is the fast path from a clone to a running simulation or a linked program. For install details (modules, HeFFTe 2.4.1, CUDA/HIP), use [`INSTALL.md`](../INSTALL.md) first.

## What you are building toward

OpenPFC is a compiled C++ library (`openpfc`) with public headers. Meaningful use involves MPI and a HeFFTe build that matches your toolchain. You can:

1. Run tutorial executables under `examples/` (after configuring with `OpenPFC_BUILD_EXAMPLES=ON`, the default).
2. Run a shipped application under `apps/` with JSON/TOML input (after `OpenPFC_BUILD_APPS=ON`, the default).
3. Link OpenPFC from your own CMake project via `find_package(OpenPFC)` (see also [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md)).

Pick one track below, then follow Next steps.

---

## 1. Configure and build OpenPFC

From the repository root (after loading compilers and MPI as in `INSTALL.md`):

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build -j"$(nproc)"
```

Use separate build directories for CPU vs GPU if you switch CUDA/HIP options; see [`build_cpu_gpu.md`](hpc/build_cpu_gpu.md).

### IDE and cluster CMake presets

If CMake Tools runs without your shell `module load`, configure may pick the wrong GCC or miss MPI. The repo ships [`CMakePresets.json`](../CMakePresets.json):

- `tohtori-debug` / `tohtori-release` — pinned GCC + Open MPI + optional HeFFTe prefix ([`cmake/toolchains/tohtori-gcc11-openmpi.cmake`](../cmake/toolchains/tohtori-gcc11-openmpi.cmake)); see [`INSTALL.md`](../INSTALL.md) (“VS Code / Cursor on tohtori”).
- `dev-debug`, `dev-asan`, etc. — local development; still load modules yourself when not using a preset that sets paths.

Override paths on your machine with `CMakeUserPresets.json` (see `cmake/README.md`).

### CMake switches for tutorials and apps

| Goal | CMake option |
|------|----------------|
| Skip building `examples/` | `-DOpenPFC_BUILD_EXAMPLES=OFF` |
| Skip building `apps/` | `-DOpenPFC_BUILD_APPS=OFF` |

Defaults are ON for both. If you turned examples or apps off, reconfigure with `ON` (or remove the cache entry) and rebuild—otherwise the paths in §2A / §2B will not exist.

---

## 2A. Run examples (library + MPI)

Examples are built into `<build>/examples/` (path may be `build/examples` or `Release/examples` depending on the generator). They are only produced when `OpenPFC_BUILD_EXAMPLES=ON` (the default).

| Order | Executable | What it shows |
|-------|------------|----------------|
| 1 | `02_domain_decomposition` | `World`, `Decomposition`, MPI layout |
| 2 | `03_parallel_fft` | Distributed FFT with HeFFTe |
| 3 | `05_simulator` | `Simulator`, `Time`, `FieldModifier` |
| 4 | `12_cahn_hilliard` | Fuller spectral model + JSON-style wiring patterns |

Typical run (adjust ranks for your machine):

```bash
cd build
mpirun -n 4 ./examples/05_simulator
```

Signs it worked: `mpirun` exits with status 0. Rank 0 typically prints INFO-level log lines from OpenPFC (world size, timestepping); there is no universal “success” string—if the process aborts, you will see an exception or non-zero exit. For more detail on a specific example, read its source under `examples/`. Sample `05_simulator` / `App` log shapes: [`example_run_output.md`](reference/example_run_output.md).

See [`examples_catalog.md`](reference/examples_catalog.md) for the full list of built targets and short descriptions.

---

## 2B. Run an application (config file)

Shipped apps live under `apps/`; see [`applications.md`](user_guide/applications.md) for binaries and sample inputs.

Tungsten (CPU) after a successful build. Requires `OpenPFC_BUILD_APPS=ON` (the default). From your build directory, point at a file under the source tree (paths are relative to `build/`):

```bash
cd build
mpirun -n 4 ./apps/tungsten/tungsten ../apps/tungsten/inputs_json/tungsten_single_seed.json
```

Signs it worked: `mpirun` exits 0; rank 0 logs progress (e.g. effective configuration, world summary, start of time integration). If `saveat` and `fields` are set, new files appear under the paths in your JSON (see [`io_results.md`](user_guide/io_results.md)). Reference `[app]` log lines: [`example_run_output.md`](reference/example_run_output.md).

Smaller or performance-oriented inputs include `tungsten_fixed_bc.json`, `tungsten_moving_bc.json`, and `tungsten_performance.json` in the same directory. TOML equivalents live under `../apps/tungsten/inputs_toml/`. Layout of sections is documented in [`apps/tungsten/inputs_json/README.md`](../apps/tungsten/inputs_json/README.md). GPU builds may provide `tungsten_cuda` or `tungsten_hip` when enabled—use the same config path with the matching binary.

---

## 2C. Use OpenPFC from your own project

Minimal pattern (your app sources must see the same MPI/HeFFTe environment used to build OpenPFC):

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_sim LANGUAGES CXX)
find_package(OpenPFC REQUIRED)
add_executable(my_sim main.cpp)
target_link_libraries(my_sim PRIVATE OpenPFC)
```

Set `CMAKE_PREFIX_PATH` (or `OpenPFC_DIR`) to the install prefix containing `lib/cmake/OpenPFC/OpenPFCConfig.cmake`. A longer walkthrough is in [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md). For `pfc::ui::App<YourModel>` with a JSON file on disk, `find_package(nlohmann_json)`, and a full CMake sketch, use [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md).

---

## Next steps

| Goal | Where to go |
|------|-------------|
| Run once and inspect output files (PNG / binary) | [`tutorials/end_to_end_visualization.md`](tutorials/end_to_end_visualization.md), [`showcase.md`](user_guide/showcase.md) |
| Tutorials hub (all walkthroughs in `docs/tutorials/`) | [`tutorials/README.md`](tutorials/README.md) |
| VTK / ParaView from `examples/` | [`tutorials/vtk_paraview_workflow.md`](tutorials/vtk_paraview_workflow.md) |
| Spectral `examples/` sequence (`04` → `05` → `12`) | [`tutorials/spectral_examples_sequence.md`](tutorials/spectral_examples_sequence.md) |
| HeFFTe `plan_options` / FFT backend | [`tutorials/fft_heffte_plan_options.md`](tutorials/fft_heffte_plan_options.md) |
| Conceptual layering (kernel / runtime / frontend) | [`architecture.md`](concepts/architecture.md) |
| Tour of main types (`World`, `Model`, `Simulator`, `App`, …) | [`class_tour.md`](reference/class_tour.md) |
| Longer tutorial (world → FFT → CMake) | [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md) |
| Minimal out-of-tree `App` + JSON (MPI, config file) | [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md) |
| Functional IC/BC (`field::apply`, …) | [`getting_started/functional_field_ops.md`](getting_started/functional_field_ops.md) |
| Config files (`plan_options`, JSON/TOML) | [`configuration.md`](user_guide/configuration.md) |
| `App` pipeline (JSON → `Simulator`) | [`app_pipeline.md`](user_guide/app_pipeline.md) |
| Validated `model.params` (custom models) | [`parameter_validation.md`](user_guide/parameter_validation.md) |
| CMake options | [`build_options.md`](reference/build_options.md) |
| Terminology | [`glossary.md`](reference/glossary.md) |
| Configure or MPI errors | [`troubleshooting.md`](troubleshooting.md) |
| Extend models, ICs, coordinates | [`extending_openpfc/README.md`](extending_openpfc/README.md) |
| HTML API reference | [Published docs](https://vtt-propertune.github.io/OpenPFC/dev/) (also build `docs` locally with `OpenPFC_BUILD_DOCUMENTATION=ON`) — pair with [`README.md`](README.md) and [`quickstart.md`](quickstart.md) for prose not generated from headers |
| HPC / LUMI | [`INSTALL.LUMI.md`](hpc/INSTALL.LUMI.md), [`lumi_slurm/README.md`](lumi_slurm/README.md) |
| `ctest` / Catch2 | [`testing.md`](development/testing.md) |
| GPU build + `tungsten_cuda` / HIP | [`tutorials/gpu_app_quickstart.md`](tutorials/gpu_app_quickstart.md) |
| Example log transcripts (shape) | [`example_run_output.md`](reference/example_run_output.md) |

---

## Getting started hub

All beginner-oriented pages are linked from [`getting_started/README.md`](getting_started/README.md). The full documentation index is [`README.md`](README.md) (this `docs/` directory).

## Common issues

- `No such file: .../examples/...` — Examples were not built; configure with `-DOpenPFC_BUILD_EXAMPLES=ON` and rebuild.
- `No such file: .../apps/tungsten/tungsten` — Apps were not built; use `-DOpenPFC_BUILD_APPS=ON` and rebuild.
- `find_package(OpenPFC)` fails — Set `CMAKE_PREFIX_PATH` or `-DOpenPFC_DIR=.../lib/cmake/OpenPFC` to the install prefix where `OpenPFCConfig.cmake` was installed (see [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md)).

More Q&A: [`faq.md`](faq.md).
