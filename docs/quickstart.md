<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Quick start

If you are brand new, start with
[`start_here_15_minutes.md`](start_here_15_minutes.md). It is the smallest
possible success path. This page is the broader quick start: it shows how to
build OpenPFC, run an example, run a shipped application, and consume an
installed OpenPFC package from another CMake project.

For install details — modules, HeFFTe 2.4.1, CUDA, HIP, and site-specific
toolchains — use [`INSTALL.md`](../INSTALL.md). This page assumes the dependency
stack is already available.

## What you are building toward

OpenPFC is a compiled C++ library with public headers. You can use it directly
from an installed package, or indirectly through executables built from this
repository. Programs under `examples/` teach the library one idea at a time.
Programs under `apps/` are closer to production runs: they accept model-specific
JSON or TOML input and are the normal entry point for deployable simulations.

## Configure and build

From the repository root, after loading the compiler and MPI environment
described in [`INSTALL.md`](../INSTALL.md):

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build -j"$(nproc)"
```

Use separate build directories for CPU and GPU configurations; see
[`build_cpu_gpu.md`](hpc/build_cpu_gpu.md). The default configuration builds
examples, applications, and tests. Disable targets explicitly when you want a
smaller build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenPFC_BUILD_TESTS=OFF
```

If CMake Tools runs without your shell `module load`, it may select the wrong
compiler or fail to find MPI. The repository ships `CMakePresets.json`:

- `tohtori-debug` and `tohtori-release` use the Tohtori toolchain file;
- `dev-debug`, `dev-asan`, and related presets support local development.

Override machine-specific paths with `CMakeUserPresets.json`; see
[`cmake/README.md`](../cmake/README.md).

## Run an example

Examples are built under `<build>/examples/` when
`OpenPFC_BUILD_EXAMPLES=ON`, which is the default. Begin with one MPI rank so
the command works on a workstation or restricted CI environment:

```bash
cd build
mpirun -n 1 ./examples/05_simulator
```

Success means the process exits with status zero. Rank zero normally prints
log lines about the domain and time stepping. The shape of successful output
is shown in
[`reference/example_run_output.md`](reference/example_run_output.md).

After the single-rank run succeeds, increase `-n` only when your local MPI or
scheduler allocation provides the requested slots. A useful reading sequence
is `02_domain_decomposition`, `03_parallel_fft`, `05_simulator`, and
`12_cahn_hilliard`; see
[`reference/examples_catalog.md`](reference/examples_catalog.md).

## Run an application

Shipped applications live under `apps/`. The complete list and input-file
pointers are in
[`user_guide/applications.md`](user_guide/applications.md). A first CPU tungsten
run is:

```bash
cd build
mpirun -n 1 ./apps/tungsten/tungsten \
  ../apps/tungsten/inputs_json/tungsten_single_seed.json
```

If the process exits with status zero, the application initialized and ran.
When the input enables output through `saveat` and `fields`, files are written
to the configured paths; see
[`user_guide/io_results.md`](user_guide/io_results.md).

GPU builds may provide `tungsten_cuda` or `tungsten_hip`. Use a matching
CUDA/HIP HeFFTe installation and read
[`tutorials/gpu_app_quickstart.md`](tutorials/gpu_app_quickstart.md) before
interpreting GPU results or performance.

## Link OpenPFC from your own project

Install OpenPFC first, then point `CMAKE_PREFIX_PATH` or `OpenPFC_DIR` at the
prefix containing `lib/cmake/OpenPFC/OpenPFCConfig.cmake`. The installed package
exports the target `OpenPFC::openpfc`.

The minimal downstream project enables both C and C++ because the package
resolves MPI C and C++ components:

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_sim LANGUAGES C CXX)

find_package(OpenPFC REQUIRED)

add_executable(my_sim main.cpp)
target_link_libraries(my_sim PRIVATE OpenPFC::openpfc)
```

Configure the consumer with the same MPI and HeFFTe stack used to build
OpenPFC:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/openpfc/install
cmake --build build
```

A longer walkthrough is in
[`getting_started/01-basics/README.md`](getting_started/01-basics/README.md).
For a config-driven 0.2 session (`make_simulation_session` + `pfc::sim::run`),
use [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md).
Migrating from 0.1 `Model`/`Simulator`/`App`:
[`MIGRATION_0.1_to_0.2.md`](MIGRATION_0.1_to_0.2.md).

## Where to go next

- Inspect generated output with
  [`tutorials/end_to_end_visualization.md`](tutorials/end_to_end_visualization.md).
- Learn the runtime data flow in
  [`concepts/spectral_stack.md`](concepts/spectral_stack.md).
- Understand package layers in
  [`concepts/architecture.md`](concepts/architecture.md).
- Choose a model-development or integration route in
  [`learning_paths.md`](learning_paths.md).
- Diagnose configuration, link, and runtime failures in
  [`troubleshooting.md`](troubleshooting.md).

If a path under `build/examples/` or `build/apps/` is missing, reconfigure with
the corresponding build option enabled. If `find_package(OpenPFC)` fails, check
the installation prefix and the active compiler/MPI stack before changing the
consumer project.