<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Start here

This page is the shortest path from a fresh clone to one successful MPI run.
It does not explain every option. It checks only that your compiler, MPI,
HeFFTe, and OpenPFC build work together.

If anything fails, stop at that step. Open
[`troubleshooting.md`](troubleshooting.md), or use
[`INSTALL.md`](../INSTALL.md) when the problem concerns compilers, MPI, or
HeFFTe.

## Check the environment

You need:

- a C++20 compiler;
- one MPI implementation used consistently for building and running;
- HeFFTe 2.4.1 installed where CMake can find it through
  `CMAKE_PREFIX_PATH` or `Heffte_DIR`.

On clusters, MPI wrappers may be unavailable until modules are loaded. A
Tohtori-style setup is:

```bash
module purge
module load openmpi/5.0.10
```

Confirm that the wrappers and launcher are visible:

```bash
which mpicc
which mpicxx
which mpirun
```

OpenPFC and HeFFTe must use the same compiler and MPI stack. Mixing MPI
implementations or launching with a different MPI installation commonly causes
link or runtime failures.

## Configure and build

Run from the repository root. If HeFFTe is outside CMake's default search
paths, prepend its install prefix before configuring:

```bash
export CC=$(which gcc)
export CXX=$(which g++)
# export CMAKE_PREFIX_PATH=/path/to/heffte/prefix:$CMAKE_PREFIX_PATH

cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build -j"$(nproc)"
```

The default configuration builds tests, examples, and applications. For the
smallest first build, tests may be disabled explicitly:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
  -DOpenPFC_BUILD_TESTS=OFF \
  -S . -B build
cmake --build build -j"$(nproc)"
```

If CMake reports missing MPI or HeFFTe, fix that dependency mismatch before
continuing.

## Run one example

Begin with one MPI rank so the command works on a workstation and does not
assume that four scheduler slots are available:

```bash
cd build
mpirun -n 1 ./examples/05_simulator
```

Success means the process exits with status zero. Rank zero usually prints a
few lines about the domain and time stepping; there is no mandatory `SUCCESS`
string.

After the one-rank run works, increase `-n` only when your local MPI setup or
scheduler allocation provides the requested slots:

```bash
mpirun -n 4 ./examples/05_simulator
```

## What you exercised

`05_simulator` walks the spectral stack at library level: `Domain`,
`Decomposition`, a HeFFTe-backed FFT, `Simulator`, and time integration. Read
[`concepts/spectral_stack.md`](concepts/spectral_stack.md) for the data flow and
[`concepts/architecture.md`](concepts/architecture.md) for layer boundaries.

## Where to go next

- Run a shipped JSON or TOML application with
  [`recipes/recipe_spectral_app_json.md`](recipes/recipe_spectral_app_json.md).
- Produce inspectable artifacts with
  [`recipes/recipe_artifacts_vtk_or_binary.md`](recipes/recipe_artifacts_vtk_or_binary.md).
- Choose a cluster, GPU, model-development, or integration route in
  [`learning_paths.md`](learning_paths.md).