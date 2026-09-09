<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Allen–Cahn demo (`apps/allen_cahn`)

Minimal 2D Allen–Cahn example on a structured grid: explicit time stepping, finite differences with separated halos, optional PNG snapshots. This app does not use the JSON/TOML `App` path; it parses simple command-line arguments and calls MPI and decomposition APIs directly.

## Problem setup

The `Problem setup` block the report contract (`#112`) asks every application
to carry. Command-line driven, no JSON, and no science preset. Note that **no
ctest runs this binary**: the area criterion below is enforced by the program's
own exit code when a user runs it, while the automated suite exercises the
update function through fixed small configurations.

| Item | Description |
|---|---|
| **Use case** | A single grain of the favoured phase growing into a matrix of the other under a constant driving force |
| **Question** | Does the favoured phase actually take over, and by how much? |
| **Domain** | `nx x ny x 1` slab, `dx = dy = 1` **hard-coded** (not a CLI argument). Defaults `64 x 64` |
| **Grid/time** | `nx ny n_steps [dt] [M] [epsilon] [driving_force] [png_initial] [png_final]`; defaults 5000 steps at `dt = 9e-5`. No output cadence -- at most two PNG snapshots, initial and final |
| **Boundary conditions** | Periodic in x and y, by separated-layout halo exchange. There is no non-periodic option |
| **Initial condition** | A single Gaussian nucleus, `phi = -1 + 2*exp(-r^2/(2*sigma^2))` centred in the slab, `sigma = max(2, 0.055*min(nx,ny))`. Deterministic; no noise anywhere |
| **Key physical parameters** | `M = 8.0`, `epsilon = 0.19`, `driving_force = 10.0`. **Nondimensional and illustrative**; this app states no units and cites no reference model. `epsilon` is as much a numerical parameter as a physical one, since it sets both the interface width and the stiff reaction timestep limit |
| **Observable** | Superlevel-set area `A = |{phi > 0}|` at start and end, reported as `N1/N0` and used as the exit status (failure unless `N1 >= 5*N0`); global `min phi`, `max phi`, `sum phi`; `avg_step_time_s`; optional grayscale PNGs |
| **Model maturity** | numerical verification: **regression** only -- one pinned CPU checksum tied to a named machine and toolchain, plus CPU-vs-CUDA agreement to `1e-9` and CPU-vs-HIP to `1e-4`. There is no analytical or manufactured-solution check anywhere in this app. Physical completeness: **canonical** -- the standard non-conserved double-well model with a constant driving force, no elastic/thermal/solutal coupling. Calibration: **none** |

### Why periodic

The grain grows inside an infinite square array of identical grains rather than
in an unbounded matrix. There is no free surface and no container, so nothing
pins the interface and there is no curvature-driven drift towards a wall --
which is the right idealization for asking whether a favoured phase grows at
all. It also caps how long the experiment means anything: once the growing
phase spans a substantial fraction of the box it starts meeting its own image,
and the area ratio stops describing an isolated grain. The 5x criterion is a
coarse growth check, not a measurement of interface velocity, and the app does
**not** verify that the seed has stayed clear of its own images.

## Binaries

| Target | When built |
|--------|------------|
| `allen_cahn` | CPU (always when apps are enabled) |
| `allen_cahn_cuda` | `OpenPFC_ENABLE_CUDA` |
| `allen_cahn_hip` | `OpenPFC_ENABLE_HIP`. Device-resident `SparseExchange<HIPSpace>` (rank 0 prints `ALLEN_CAHN_HIP_HALO_MODE`). |

## Build

With `OpenPFC_BUILD_APPS=ON`:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

Executable: `build/apps/allen_cahn/allen_cahn`.

## Usage

```text
mpirun -n <P> ./allen_cahn <nx> <ny> <n_steps> [dt] [M] [epsilon] [driving_force] [png_initial] [png_final]
```

Defaults if omitted: `nx=ny=64`, `n_steps=5000`, `dt=0.00009`, `M=8.0`, `epsilon=0.19`, `driving_force=10.0`.
The app counts the global visible seed area as cells with `phi > 0` at the beginning and end of the run, prints `N1/N0`, and exits with failure unless the final area is at least 5× the initial area. The optional positive `driving_force` favors the `phi≈+1` seed over the `phi≈-1` matrix. If you pass one PNG path, it writes the final field; if two, the first is the initial snapshot and the second the final (grayscale, rank 0).
The reported step timing measures the time-stepping loop only, after an MPI barrier and before PNG output or verification; `avg_step_time_s` is based on the slowest rank.

Example:

```bash
cd build
mpirun -n 4 ./apps/allen_cahn/allen_cahn 128 128 5000 0.00009 8.0 0.19 10.0 initial.png final.png
```

## See also

- [`../../docs/halo_exchange.md`](../../docs/concepts/halo_exchange.md) — halo policies (this demo uses separated layout for FD) 
- [`../../examples/15_finite_difference_heat.cpp`](../../examples/15_finite_difference_heat.cpp) — related FD + halo pattern in `examples/` 
