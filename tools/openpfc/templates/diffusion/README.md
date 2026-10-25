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
openpfc compile model.cpp -o model
./model input.json
```

Add `--sdk /path/to/openpfc-sdk` to doctor and compile if the SDK is not
found automatically. Compiler and generator options passed to doctor must
also be passed to compile. No OpenPFC checkout or handwritten CMake is
required. Use `openpfc help compile` for all build options.

`model` is a regular executable in this directory. Without `-o`, compilation
uses the source filename without its extension in the current directory.
Run it directly, or choose your own MPI or scheduler launcher:

```bash
mpirun -n 2 ./model input.json
# Inside a suitable Slurm allocation with matching MPI:
srun -n 2 ./model input.json
```

The model writes into `results/` under the current working directory:

- `results/diagnostics.csv`: time, mean and L2 norm at each step;
- `results/field_rank_N.csv`: the final field on each MPI rank;
- `results/summary.json`: final global diagnostics, written once.

The default input runs ten steps on a 16-cubed grid. The mean stays near 1
and the L2 norm decreases. Edit `diffusivity`, `dt`, `steps` or `cells` in
`input.json`, then repeat `./model input.json`; no compilation is needed.
Edit `Diffusion::rhs` in `model.cpp` to change the physics, then compile again.
Direct runs use ordinary terminal output and reuse the model's result paths;
choose separate working directories when retaining multiple runs.

The starter accepts integer `cells` from 4 to 256, integer `steps` from 1 to
1000000, finite positive `diffusivity` and `dt`, and the explicit stability
bound `dt <= 1/(6*diffusivity)`. Change the timestep rule when changing the
equation or discretization. The model owns its input format and validation.

## Optional recorded execution

For frozen inputs, separate attempts, captured logs and output checksums, use:

```bash
openpfc run model.cpp input.json --launcher=none
```

This prints its run/attempt directory and keeps its results there. Preserve
that record beyond cache cleanup with:

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
