<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Start here

This page is the shortest path from a fresh clone to one successful MPI run. It does not try to explain every option. It only checks that your compiler, MPI, HeFFTe and OpenPFC build can work together.

If anything here fails, do not keep going and hope the next command fixes it. Open [`troubleshooting.md`](troubleshooting.md), or use the full install guide in [`INSTALL.md`](../INSTALL.md) when the problem is clearly about compilers, MPI or HeFFTe.

## Check the environment

You need a C++20 compiler, an MPI implementation that you will both build with and run with, and HeFFTe 2.4.1 installed somewhere CMake can find it through `CMAKE_PREFIX_PATH` or `Heffte_DIR`. On clusters, `mpicc` and `mpirun` are often unavailable until you load modules. A typical Tohtori-style setup looks like this:

```bash
module load gcc/11.2.0
module load openmpi/4.1.1
```

Then confirm that the MPI wrappers are really visible:

```bash
which mpicc
which mpirun
```

If these commands still do not find MPI, stop and fix the environment. OpenPFC and HeFFTe must use the same MPI implementation end to end; mixing wrappers and launchers from different MPI stacks leads to confusing runtime failures later.

## Configure and build

Run the build from the repository root. If HeFFTe is installed outside CMake's default search paths, prepend its install prefix to `CMAKE_PREFIX_PATH` before configuring.

```bash
export CC=$(which gcc)
export CXX=$(which g++)
# If HeFFTe is not on CMake’s default search path:
# export CMAKE_PREFIX_PATH=/path/to/heffte/prefix:$CMAKE_PREFIX_PATH

cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build -j"$(nproc)"
```

The default configuration builds both `examples/` and `apps/`. If CMake reports missing HeFFTe or MPI, treat that as the real problem and fix it before building.

## Run one example

```bash
cd build
mpirun -n 4 ./examples/05_simulator
```

Success means every rank exits with status zero. Rank zero usually prints a few log lines about the world size and time stepping, but there is no single mandatory “SUCCESS” string.

## What you just exercised

`05_simulator` walks the spectral stack at library level: `World`, `Decomposition`, a HeFFTe-backed FFT, `Simulator`, and time integration. Read [`concepts/spectral_stack.md`](concepts/spectral_stack.md) for the data-flow story and [`concepts/architecture.md`](concepts/architecture.md) for the layer boundaries.

## Where to go next

If you want to run a shipped JSON or TOML application, continue with [`recipes/recipe_spectral_app_json.md`](recipes/recipe_spectral_app_json.md). If you want files you can inspect in ParaView or with Python, use [`recipes/recipe_artifacts_vtk_or_binary.md`](recipes/recipe_artifacts_vtk_or_binary.md). If you are heading toward a cluster or GPU build, read [`learning_paths.md`](learning_paths.md) first so you do not jump into site-specific details too early.
