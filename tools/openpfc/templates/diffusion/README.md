<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Your diffusion model

`model.cpp` is an editable C++20 program using OpenPFC fields, periodic halo
exchange, finite differences and a forward-Euler stepper. It solves
`du/dt = diffusivity * Laplacian(u)` in a periodic cubic box with unit spacing.
The initial field is `1 + 0.1*cos(2*pi*x/cells)`: diffusion conserves its mean
and reduces the oscillation. This is a small numerical example for learning.

From this directory, after loading the matching compiler and MPI environment:

```bash
openpfc doctor
openpfc compile model.cpp
openpfc run model.cpp input.json --launcher=none
```

Add `--sdk /path/to/openpfc-sdk` to doctor, compile and run if the SDK is not
discovered automatically. Compiler and generator options passed to doctor
must also be passed to compile/run. No OpenPFC checkout or handwritten CMake
is required. Use `openpfc help compile` for all build options.

The CLI prints the run and attempt directories. Inside the attempt, inspect:

- `stdout.log` and `stderr.log` for the model's messages;
- `results/diagnostics.csv` for time, mean and L2 norm at each step;
- `results/field_rank_0.csv` for the final field on rank zero;
- `results/summary.json` for the final global diagnostics;
- `status.json` for the exit status and output checksums.

The default input runs ten steps on a 16-cubed grid. The mean stays near 1
and the L2 norm decreases. Edit `diffusivity`, `dt`, `steps` or `cells` in
`input.json`, then rerun the same command. Input-only changes reuse the build;
every run request preserves a new attempt. Edit `Diffusion::rhs` in `model.cpp`
to change the physics; source changes trigger compilation.

The starter accepts integer `cells` from 4 to 256, integer `steps` from 1 to
1000000, finite positive `diffusivity` and `dt`, and the explicit stability
bound `dt <= 1/(6*diffusivity)`. Change the timestep rule when changing the
equation or discretization. The model owns its input format and validation.

For two local MPI ranks, invoke the CLI once outside an MPI launch:

```bash
openpfc run model.cpp input.json --ranks=2 --launcher=mpirun
```

Each rank writes its own `results/field_rank_N.csv`; global diagnostics are
written once. Keep results beyond cache cleanup with:

```bash
openpfc export /path/to/cache/runs/openpfc-run-SHA256 /durable/new-record
```

To package this CPU build as an image, use `openpfc bundle model.cpp -o model.sif`
with the host MPI prefix and `--host-lib` for each fabric library that shares a
system directory with the C library. Omitting `--base` builds a scratch image
that copies the host loader, glibc and `/bin/sh`. Pass `--base` to use a pinned
image for those files instead. The image runs `model`, not a compiler or Python.
CUDA and ROCm profiles are not qualified for bundling. Details are in the native
CLI guide.
