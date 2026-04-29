<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Start here (~15 minutes)

**Goal:** clone → configure → build → run one MPI example → see a clean exit. No optional branches until the end.

If anything below fails, open [`troubleshooting.md`](troubleshooting.md) or [`INSTALL.md`](../INSTALL.md) (full toolchain: GCC, Open MPI, HeFFTe 2.4.1).

## 0. Prerequisites (60 seconds)

You need a C++17 compiler, **MPI you will build and run with** (Open MPI is the reference stack), and **HeFFTe 2.4.1** installed and discoverable via `CMAKE_PREFIX_PATH` or `Heffte_DIR`. See [`INSTALL.md`](../INSTALL.md) §1–§3.

**If `mpicc` / `mpirun` are missing:** on clusters, load modules first (example pattern):

```bash
module load gcc/11.2.0
module load openmpi/4.1.1
```

Then confirm:

```bash
which mpicc
which mpirun
```

If this still fails, stop here and fix the environment — OpenPFC and HeFFTe must use the **same** MPI implementation end-to-end ([`INSTALL.md`](../INSTALL.md) §“MPI: OpenMPI from modules”).

## 1. Configure and build (~5–10 minutes)

From the **repository root**:

```bash
export CC=$(which gcc)
export CXX=$(which g++)
# If HeFFTe is not on CMake’s default search path:
# export CMAKE_PREFIX_PATH=/path/to/heffte/prefix:$CMAKE_PREFIX_PATH

cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build -j"$(nproc)"
```

Defaults build `examples/` and `apps/` (`OpenPFC_BUILD_EXAMPLES` / `OpenPFC_BUILD_APPS` ON). If configure errors mention missing HeFFTe or MPI, see [`troubleshooting.md`](troubleshooting.md).

## 2. Run one example (~1 minute)

```bash
cd build
mpirun -n 4 ./examples/05_simulator
```

**Success:** all ranks exit with status **0**. Rank 0 usually prints log lines (world size, stepping); there is no single mandatory “SUCCESS” string.

## 3. What you just exercised

`05_simulator` walks the **spectral stack** at library level: `World`, `Decomposition`, HeFFTe-backed FFT, `Simulator`, and time integration. Read the story in [`spectral_stack.md`](spectral_stack.md) and layering in [`architecture.md`](architecture.md).

## 4. Next steps (pick one)

| Next | Open |
|------|------|
| Shipped JSON/TOML app (e.g. tungsten) | [`recipes/recipe_spectral_app_json.md`](recipes/recipe_spectral_app_json.md) |
| VTK / ParaView from an example | [`recipes/recipe_artifacts_vtk_or_binary.md`](recipes/recipe_artifacts_vtk_or_binary.md) |
| Full tutorial index | [`tutorials/README.md`](tutorials/README.md) |
| GPU vs CPU build | [`gpu_path_decision.md`](gpu_path_decision.md) |
| Cluster jobs / I/O | [`hpc_operator_guide.md`](hpc_operator_guide.md) |

Sequenced tracks by role: [`learning_paths.md`](learning_paths.md).
