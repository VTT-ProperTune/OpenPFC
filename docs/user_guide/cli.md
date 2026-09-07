<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Create, compile, and run a case

The `openpfc` command creates a shipped app case, builds the application
through `scripts/build.sh`, and launches it with MPI or Slurm. It requires
Python 3.8 or newer. It selects shipped physics; it does not
compile custom C++ models or build container images.

From a checkout on Tohtori, run:

```bash
./scripts/openpfc --version
./scripts/openpfc init results/mycase
./scripts/openpfc compile results/mycase --profile=local
# Restore the matching compiler/MPI environment in this shell:
module load openmpi/5.0.10
./scripts/openpfc run results/mycase --ranks=1
```

`init` creates `input.json`, a small `openpfc.json` project manifest, and a
README. It refuses an existing project directory. Edit `input.json` to change
the model parameters, domain, and time interval. The default 16³ seed case is
a quick runtime check, not a quantitatively calibrated materials prediction.

`compile` runs the canonical build and test script. The first build needs
the documented compiler, MPI, and HeFFTe stack, plus network access for missing
dependencies. It records the binary and MPI launcher only after a successful
build. `--jobs=N` controls build parallelism; `--no-test` explicitly skips the
suite for subsequent development builds. `--build-dir=/absolute/path` selects
an existing compatible build tree. Never mix CPU and GPU backends in one tree.

## Choose an app

```bash
./scripts/openpfc apps
./scripts/openpfc init results/spinodal --app=cahn_hilliard --preset=spinodal
./scripts/openpfc compile results/spinodal --profile=local
module load openmpi/5.0.10
./scripts/openpfc run results/spinodal --ranks=2
```

The catalog covers seven JSON-session apps. `init` without options still creates
the tungsten smoke case; otherwise the first listed preset is the app default.

| App | Presets | Backends |
|-----|---------|----------|
| `tungsten` | `smoke` | CPU, CUDA, HIP |
| `aluminum` | `smoke` | CPU, CUDA, HIP |
| `cahn_hilliard` | `spinodal`, `mode` | CPU, HIP |
| `thin_film` | `leveling`, `dewetting` | CPU, HIP |
| `surface_diffusion` | `smoothing` | CPU, HIP |
| `kawahara` | `pulse` | CPU, HIP |
| `ehd_film` | `relaxation` | CPU, HIP |

Support means a build target exists, not that it is installed or verified on
the current machine. Presets are installed alongside the CLI, so initialization
does not require a source checkout. Unsupported app/profile combinations fail
before building. The positional-argument teaching apps (`allen_cahn`, `heat3d`,
`wave2d`, `kobayashi`) retain their own interfaces.

The Cahn–Hilliard `spinodal` preset seeds reproducible broadband perturbations
and writes `results/cahn_hilliard/diagnostics.csv`. `mode` retains the original
single-cosine verification case. See the [app guide](../../apps/cahn_hilliard/README.md)
for energy interpretation and timestep limitations. Aluminum's small uniform
case is a runtime check, not a solidification demonstration.

## Profiles

| Profile | Backend | Build route |
|---------|-------------|-------------|
| `local` (default) | CPU | Machine auto-detection, build in this process; `builds/cli-local` |
| `tohtori` | CUDA | Tohtori CUDA stack; `builds/cli-tohtori` |
| `lumi` | HIP | LUMI HIP stack and Slurm build with `--wait`; flash build tree |

Profiles use the machines and toolchains already supported by
[`scripts/build.sh`](../../scripts/build.sh); `local` currently needs a supported
development environment such as Tohtori. It does not add a generic laptop
toolchain. On LUMI, select `lumi`, which builds under
`$LUMI_FLASH_ROOT/openpfc-cli-lumi` (default root
`/flash/project_462001519/juaho/build`). Keep large cases and output on scratch.
The build script retains ownership of modules, GPU settings, and Slurm account
selection. Run `compile` once outside `mpirun` or `srun`.

Both `compile --dry-run` and `run --dry-run` print the command and working
directory without building, running, or updating the project.

## MPI and output

`run` uses the case directory as its working directory, so the default output
lands in `<case>/results/` regardless of where you invoke the CLI. The scaffold
writes VTK snapshots and a checkpoint at step 10. Use fresh output paths for
subsequent runs: result writers can replace existing snapshots, while checkpoint
publication refuses to replace an existing bundle.

Outside Slurm, `run` uses the MPI launcher recorded by `compile`, or `mpirun`
from `PATH`, with one rank by default. Within a Slurm allocation, it uses
`srun` and inherits the allocation's task count unless `--ranks` overrides it.
Load the same compiler/MPI modules used for the build before running; module
changes made inside the build subprocess do not alter your parent shell.
`--launcher=mpirun`, `--launcher=srun`, and `--launcher=none` select an explicit
route; `none` is for a singleton run or an existing MPI launch.

You can also control MPI directly:

```bash
mpirun -n 2 ./scripts/openpfc run results/mycase
# Inside a Slurm allocation:
srun ./scripts/openpfc run /scratch/project_462001519/juaho/mycase
```

The driver detects MPI rank environment variables and invokes the app directly
in each rank. It rejects a second `--ranks` setting or an explicit nested
launcher. It propagates nonzero build and application exit codes.

For restarts, follow the [tungsten restart guide](../../apps/tungsten/README.md#restart-a-moving-front).
Keep `t0`, `dt`, physics, grid, and boundary configuration consistent with the
checkpoint; extend `t1`, set `restart_from`, and select new output locations.

## Build trees and installations

Configuration also creates `<build>/bin/openpfc`. The normal installation rules
install `bin/openpfc` and `share/openpfc/version`. A relocated installation can
print its version and initialize cases without the source checkout. `run` can
use a sibling installed `tungsten` executable without compiling first.

An installed driver needs a source checkout for `compile`: pass
`--source=/path/to/OpenPFC` or set `OPENPFC_SOURCE_DIR`. The source-tree and
build-tree drivers discover their checkout automatically. `run --executable`
selects a particular installed CPU/CUDA/HIP binary and overrides the case's
recorded build path; load its matching MPI environment.

The separate [CPU runtime baker](../../containers/runtime/README.md) can package
a compiled local tungsten case into an Apptainer image with host-MPI bindings. The CLI
itself does not produce images. Development images, GPU profiles for images,
and multi-node network integration remain packaging work.
