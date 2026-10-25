<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Start here

Use an installed OpenPFC SDK to create a CPU diffusion model, run it and
inspect its results. If you are working from a source checkout, the
[source-build route](#verify-a-source-build) below checks the spectral stack
with a small MPI example.

If anything fails, stop at that step. Open
[`troubleshooting.md`](troubleshooting.md), or use
[`INSTALL.md`](../INSTALL.md) when the problem concerns compilers, MPI, or
HeFFTe.

## Run your first model with an installed SDK

### Load and check the environment

You need an installed SDK with the native `openpfc` CLI, CMake, a C++20
compiler and the compiler/MPI environment used by that SDK. See the
[installation routes](../INSTALL.md#choose-an-installation-route) for site
modules and source installation. Older release modules may not contain the
CLI commands below; check `openpfc --help` against your installed version.

With the environment loaded, run this once outside an MPI launch:

```bash
openpfc doctor
```

`doctor` compiles, links and runs a small host probe against the SDK. If it
fails, follow its recovery instructions and inspect the printed diagnostics
directory before continuing. It does not install dependencies. If discovery
cannot find your SDK, use `openpfc doctor --sdk /path/to/openpfc-sdk` and pass
the same `--sdk` option to `compile` and `run` below.

On a cluster, use an allocation or another environment where compilation and
execution are permitted. For LUMI, follow the [operator guide](hpc/operator_guide.md).

### Create and run a starter

```bash
openpfc init my-model --model=diffusion
cd my-model
openpfc compile model.cpp
openpfc run model.cpp input.json --launcher=none
```

`my-model` must be a new directory. The starter contains `model.cpp`,
`input.json` and a README. The model solves the periodic heat equation on CPU
with finite differences. `--launcher=none` runs one rank directly. The CLI
prints the attempt directory containing logs and `results/`.

### Inspect the result

Open the files under `results/` in that printed attempt directory:

| File | What to check with the default input |
|------|--------------------------------------|
| `summary.json` | `mean` stays near 1 and `final_l2` is smaller than `initial_l2` |
| `diagnostics.csv` | Time advances, the mean stays near 1 and the L2 norm decreases |
| `field_rank_0.csv` | The final field values for the one-rank run |

These checks show diffusion smoothing the initial field while conserving its
mean. The starter is a numerical example, not a calibrated material model.
The attempt's `stdout.log` and `stderr.log` help diagnose an unsuccessful run.

Edit `input.json` and repeat the `run` command. Input-only changes reuse the
compiled model and save a new attempt. Edit `Diffusion::rhs` in `model.cpp`
when you are ready to change the physics. For two ranks, use
`--ranks=2 --launcher=mpirun` instead of `--launcher=none` when your environment
provides the slots. Continue with the [native CLI guide](user_guide/cli.md)
for input constraints, MPI, durable export and packaging.

## Verify a source build

This route assumes a checkout and a consistent compiler, MPI and HeFFTe stack
as described in [INSTALL.md](../INSTALL.md). Use the canonical build script
from the repository root. On Tohtori:

```bash
./scripts/build.sh --machine=tohtori --cpu
module load openmpi/5.0.10
```

The module load keeps the launch shell on the same MPI stack as the build;
environment changes inside the build script do not persist in that shell.

On an ordinary workstation, install the dependencies first and use portable
mode, which preserves your compiler/MPI environment and CMake package paths:

```bash
./scripts/build.sh --machine=portable --cpu
```

These commands build and test under `builds/release`. On LUMI, follow the
[LUMI guide](hpc/INSTALL.LUMI.md) for the scheduled build and run workflow;
keep the build tree on flash outside the clone.

Begin with one MPI rank so the command works on a workstation and does not
assume that four scheduler slots are available:

```bash
mpirun -n 1 ./builds/release/examples/05_simulator
```

Success means the process exits with status zero. Rank zero usually prints a
few lines about the domain and time stepping; there is no mandatory `SUCCESS`
string.

After the one-rank run works, increase `-n` only when your local MPI setup or
scheduler allocation provides the requested slots:

```bash
mpirun -n 4 ./builds/release/examples/05_simulator
```

`05_simulator` walks the spectral stack at library level: `Domain`,
`SpectralCPUStack`, a HeFFTe-backed FFT, and `pfc::sim::run`. Read
[`concepts/spectral_stack.md`](concepts/spectral_stack.md) for the data flow and
[`concepts/architecture.md`](concepts/architecture.md) for layer boundaries.

## Where to go next

- Run a shipped JSON or TOML application with
  [`recipes/recipe_spectral_app_json.md`](recipes/recipe_spectral_app_json.md).
- Produce inspectable artifacts with
  [`recipes/recipe_artifacts_vtk_or_binary.md`](recipes/recipe_artifacts_vtk_or_binary.md).
- Choose a cluster, GPU, model-development, or integration route in
  [`learning_paths.md`](learning_paths.md).
