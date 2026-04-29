<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Quick start

If you are brand new, start with [`start_here_15_minutes.md`](start_here_15_minutes.md). It is the smallest possible success path. This page is the broader quick start: it shows how to build OpenPFC, run an example, run a shipped application, and link the library from your own CMake project.

For install details — modules, HeFFTe 2.4.1, CUDA, HIP and site-specific toolchains — use [`INSTALL.md`](../INSTALL.md). This page assumes the dependency stack is already available.

## What you are building toward

OpenPFC is a compiled C++ library with public headers. You can use it directly by linking `OpenPFC` from your own project, or indirectly through executables built from this repository. The small programs under `examples/` teach the library one idea at a time. The programs under `apps/` are closer to real runs: they accept model-specific inputs, usually from JSON or TOML, and are the right entry point when you want a deployable binary.

## Configure and build

From the repository root, after loading the compiler and MPI environment described in [`INSTALL.md`](../INSTALL.md):

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build -j"$(nproc)"
```

Use separate build directories for CPU vs GPU if you switch CUDA/HIP options; see [`build_cpu_gpu.md`](hpc/build_cpu_gpu.md).

The default configuration builds both examples and applications. If you previously turned either `OpenPFC_BUILD_EXAMPLES` or `OpenPFC_BUILD_APPS` off, the paths below will not exist until you reconfigure with that option enabled.

If CMake Tools runs without your shell `module load`, configure may pick the wrong GCC or miss MPI. The repo ships [`CMakePresets.json`](../CMakePresets.json):

- `tohtori-debug` / `tohtori-release` — pinned GCC + Open MPI + optional HeFFTe prefix ([`cmake/toolchains/tohtori-gcc11-openmpi.cmake`](../cmake/toolchains/tohtori-gcc11-openmpi.cmake)); see [`INSTALL.md`](../INSTALL.md) (“VS Code / Cursor on tohtori”).
- `dev-debug`, `dev-asan`, etc. — local development; still load modules yourself when not using a preset that sets paths.

Override paths on your machine with `CMakeUserPresets.json` (see `cmake/README.md`).

## Run an example

Examples are built into `<build>/examples/` (path may be `build/examples` or `Release/examples` depending on the generator). They are only produced when `OpenPFC_BUILD_EXAMPLES=ON` (the default).

The simulator example is a good first run because it touches the same core pieces that larger applications use:

```bash
cd build
mpirun -n 4 ./examples/05_simulator
```

If `mpirun` exits with status zero, the run worked. Rank zero typically prints INFO-level lines about the world size and time stepping; if the process aborts, you will see an exception or a non-zero exit code. The shape of successful logs is shown in [`reference/example_run_output.md`](reference/example_run_output.md).

After that first example, the usual reading order is `02_domain_decomposition`, then `03_parallel_fft`, then `05_simulator`, then `12_cahn_hilliard`. The full list is in [`reference/examples_catalog.md`](reference/examples_catalog.md).

## Run an application

Shipped applications live under `apps/`; [`user_guide/applications.md`](user_guide/applications.md) describes the available binaries and sample inputs. The most useful first application run is the CPU tungsten binary with a sample JSON file:

```bash
cd build
mpirun -n 4 ./apps/tungsten/tungsten ../apps/tungsten/inputs_json/tungsten_single_seed.json
```

If `mpirun` exits with status zero, the application ran. Rank zero logs the effective configuration, the world summary and the start of time integration. If the input enables output through `saveat` and `fields`, new files appear under the paths named in the JSON; [`user_guide/io_results.md`](user_guide/io_results.md) explains the writer formats.

The same directory contains smaller or performance-oriented inputs such as `tungsten_fixed_bc.json`, `tungsten_moving_bc.json`, and `tungsten_performance.json`. TOML equivalents live under `apps/tungsten/inputs_toml/`. GPU builds may provide `tungsten_cuda` or `tungsten_hip`; use the same config path with the matching binary, and read [`tutorials/gpu_app_quickstart.md`](tutorials/gpu_app_quickstart.md) before treating GPU runs as production measurements.

## Link OpenPFC from your own project

The minimal CMake shape is small, but the environment still matters: your application must use the same MPI and HeFFTe stack that OpenPFC was built against.

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_sim LANGUAGES CXX)
find_package(OpenPFC REQUIRED)
add_executable(my_sim main.cpp)
target_link_libraries(my_sim PRIVATE OpenPFC)
```

Set `CMAKE_PREFIX_PATH` (or `OpenPFC_DIR`) to the install prefix containing `lib/cmake/OpenPFC/OpenPFCConfig.cmake`. A longer walkthrough is in [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md). For `pfc::ui::App<YourModel>` with a JSON file on disk, `find_package(nlohmann_json)`, and a full CMake sketch, use [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md).

## Where to go next

If you want to inspect output files, continue with [`tutorials/end_to_end_visualization.md`](tutorials/end_to_end_visualization.md), then [`tutorials/vtk_paraview_workflow.md`](tutorials/vtk_paraview_workflow.md) for ParaView or [`user_guide/postprocess_binary_fields.md`](user_guide/postprocess_binary_fields.md) for raw binary data. If you want to understand the stack you just ran, read [`concepts/spectral_stack.md`](concepts/spectral_stack.md) and [`concepts/architecture.md`](concepts/architecture.md). If you are building your own model or app, go to [`learning_paths.md`](learning_paths.md) and follow the extension route.

If a path under `build/examples/` or `build/apps/` does not exist, reconfigure with examples or apps enabled and rebuild. If `find_package(OpenPFC)` fails, point `CMAKE_PREFIX_PATH` or `OpenPFC_DIR` at the install prefix containing `OpenPFCConfig.cmake`. More failure modes are collected in [`troubleshooting.md`](troubleshooting.md), and short answers live in [`faq.md`](faq.md).

