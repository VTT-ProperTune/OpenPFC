<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Allen–Cahn demo (`apps/allen_cahn`)

Minimal 2D Allen–Cahn example on a structured grid: explicit time stepping, finite differences with separated halos, optional PNG snapshots. This app does not use the JSON/TOML `App` path; it parses simple command-line arguments and calls MPI and decomposition APIs directly.

## Problem setup

The `Problem setup` block the report contract (`#112`) asks every application
to carry. Command-line driven, no JSON, and no science preset. Note that **no
ctest runs this binary**, but the criterion it applies is no longer only
reachable through the binary: `allen-cahn-interface-kinetics` runs the same
update at two grid sizes and asserts that both reach the same verdict.

| Item | Description |
|---|---|
| **Use case** | A single grain of the favoured phase growing into a matrix of the other under a constant driving force |
| **Question** | Does the favoured phase actually take over, and by how much? |
| **Domain** | `nx x ny x 1` slab, `dx = dy = 1` **hard-coded** (not a CLI argument). Defaults `64 x 64` |
| **Grid/time** | `[--strict] nx ny n_steps [dt] [M] [epsilon] [driving_force] [png_initial] [png_final]`; defaults 5000 steps at `dt = 9e-5`. No output cadence -- at most two PNG snapshots, initial and final |
| **Boundary conditions** | Periodic in x and y, by separated-layout halo exchange. There is no non-periodic option |
| **Initial condition** | A single Gaussian nucleus, `phi = -1 + 2*exp(-r^2/(2*sigma^2))` centred in the slab, `sigma = max(2, 0.055*min(nx,ny))`. Deterministic; no noise anywhere |
| **Key physical parameters** | `M = 8.0`, `epsilon = 0.19`, `driving_force = 10.0`. **Nondimensional and illustrative**; this app states no units and cites no reference model. `epsilon` is as much a numerical parameter as a physical one, since it sets both the interface width and the stiff reaction timestep limit |
| **Observable** | Interface velocity `dR/dt`, where `R = sqrt(A/pi)` is the equivalent radius of the superlevel-set area `A = |{phi > 0}|`, measured over the **second half** of the run and compared with the sharp-interface prediction `v = (3/2) F eps sqrt(2M)`. `A` is sampled at `t = 0, t/2, 3t/4, t`; the two quarter velocities must agree before `v_late` is judged. Also global `min phi`, `max phi`, `sum phi`; `avg_step_time_s`; optional grayscale PNGs. The verdict prints as `physics_check=PASS|FAIL|SKIPPED` and **does not** set the exit status unless `--strict` is passed |
| **Model maturity** | numerical verification: **analytical** (loosely) and **regression** -- the end-of-run check compares the measured interface velocity with the sharp-interface solvability result `v = (3/2) F eps sqrt(2M)`, to a factor of two, which is as tight as an interface 0.76 cells wide supports; plus one pinned CPU checksum tied to a named machine and toolchain, CPU-vs-CUDA agreement to `1e-9` and CPU-vs-HIP to `1e-4`. There is no manufactured-solution check. Physical completeness: **canonical** -- the standard non-conserved double-well model with a constant driving force, no elastic/thermal/solutal coupling. Calibration: **none** |

### Why periodic

The grain grows inside an infinite square array of identical grains rather than
in an unbounded matrix. There is no free surface and no container, so nothing
pins the interface and there is no curvature-driven drift towards a wall --
which is the right idealization for asking whether a favoured phase grows at
all. It also caps how long the experiment means anything: once the growing
phase spans a substantial fraction of the box it starts meeting its own image,
the front stops advancing and the app does **not** verify that the seed has
stayed clear of its own images.

### Why a velocity and not an area ratio

The app used to require `N1 >= 5*N0`. The seed radius scales with the grid
(`sigma = 0.055*min(nx,ny)`) but the interface speed does not, so the ratio
`((R0 + v t)/R0)^2` shrinks as the box grows: the same physics scored 6.08x at
64^2, 3.50x at 128^2 and 2.55x at 256^2 -- pass, fail, fail. Measured as
`dR/dt` over the last half of the run the three agree to 5% (12.4 / 12.8 /
13.1). The last half, because the Gaussian initial condition is far from the
equilibrium `tanh` and its collapse moves the contour by a distance that
scales with the seed; a whole-run average inherits exactly the grid dependence
the area ratio had.

### Two limits worth knowing before changing the parameters

* **The interface is not resolved.** `eps*sqrt(2M) = 0.76` cells at the shipped
  preset, i.e. under one grid spacing, so the front is lattice-limited rather
  than continuum-limited and the measured speed sits some tens of percent off
  the sharp-interface law (`+11%` at the defaults, `-35%` at
  `driving_force = 5`). That is why the accepted band is a factor of two either
  way rather than a percentage. The app prints the width and warns when it is
  below one cell.
* **The driving force is close to its ceiling.** A front exists only while
  `F*eps^2 < 2/(3 sqrt 3)`; above that there is no metastable `phi < 0` phase
  and the whole domain simply decays. The shipped `F = 10`, `eps = 0.19` gives
  `0.361` against a limit of `0.385` -- 6% of margin. `driving_force = 20` or
  `epsilon = 0.3` crosses it and flips the entire box, which the app now says
  out loud.

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
mpirun -n <P> ./allen_cahn [--strict] <nx> <ny> <n_steps> [dt] [M] [epsilon] [driving_force] [png_initial] [png_final]
```

Defaults if omitted: `nx=ny=64`, `n_steps=5000`, `dt=0.00009`, `M=8.0`, `epsilon=0.19`, `driving_force=10.0`.
The app samples the global visible seed area (cells with `phi > 0`) at `t = 0`, `t/2`, `3t/4` and the end, converts each to an equivalent radius, and reports the interface velocity over the last half of the run against `(3/2) F eps sqrt(2M)`. The verdict prints as `physics_check=`; the exit status reports whether the *run* finished, so a deliberately short run is not a failed process. Pass `--strict` to have a `FAIL` verdict (never a `SKIPPED` one) set the exit status instead. The optional positive `driving_force` favors the `phi≈+1` seed over the `phi≈-1` matrix. If you pass one PNG path, it writes the final field; if two, the first is the initial snapshot and the second the final (grayscale, rank 0).
The reported step timing measures the time-stepping loop only, after an MPI barrier and before PNG output or verification; `avg_step_time_s` is based on the slowest rank.

Example:

```bash
cd build
mpirun -n 4 ./apps/allen_cahn/allen_cahn 128 128 5000 0.00009 8.0 0.19 10.0 initial.png final.png
```

## See also

- [`../../docs/halo_exchange.md`](../../docs/concepts/halo_exchange.md) — halo policies (this demo uses separated layout for FD) 
- [`../../examples/15_finite_difference_heat.cpp`](../../examples/15_finite_difference_heat.cpp) — related FD + halo pattern in `examples/` 
