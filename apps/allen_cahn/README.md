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
| **Domain** | `nx x ny x 1` slab, `dx = dy = 1` **hard-coded** (not a CLI argument). Defaults `256 x 256`. The spacing being fixed matters: the interface can only be resolved by making `eps*sqrt(2M)` large in cells, never by refining the grid |
| **Grid/time** | `[--strict] nx ny n_steps [dt] [M] [epsilon] [driving_force] [png_initial] [png_final]`; defaults 5000 steps at `dt = 0.005`, 16% of the explicit-Euler diffusive limit `dx^2/(4M) = 0.031`. No output cadence -- at most two PNG snapshots, initial and final |
| **Boundary conditions** | Periodic in x and y, by separated-layout halo exchange. There is no non-periodic option |
| **Initial condition** | A single Gaussian nucleus, `phi = -1 + 2*exp(-r^2/(2*sigma^2))` centred in the slab, `sigma = max(2, 0.055*min(nx,ny))`. Deterministic; no noise anywhere |
| **Key physical parameters** | `M = 8.0`, `epsilon = 0.75`, `driving_force = 0.25`. **Nondimensional and illustrative**; this app states no units and cites no reference model. `epsilon` is as much a numerical parameter as a physical one, since it sets both the interface width and the stiff reaction timestep limit. The three are chosen together so that the interface is `eps*sqrt(2M) = 3.0` cells wide and `F*eps^2 = 0.141` sits a factor 2.7 under the bistability ceiling `2/(3 sqrt 3) = 0.385` -- see "Two limits" below. Before 0.2 the preset was `epsilon = 0.19`, `driving_force = 10.0`: 0.76 cells and 6% of margin |
| **Observable** | Interface velocity `dR/dt`, where `R = sqrt(A/pi)` is the equivalent radius of the superlevel-set area `A = |{phi > 0}|`, measured over the **second half** of the run and compared with the **curvature-corrected** sharp-interface prediction `v = (3/2) F eps sqrt(2M) - M/R`, evaluated at the mean radius of that interval, to `+/-25%`. `A` is sampled at `t = 0, t/2, 3t/4, t`; the two quarter velocities must agree before `v_late` is judged. Also the critical radius `M/v`, the interface width in cells, the distance to the bistability ceiling, global `min phi`, `max phi`, `sum phi`, `avg_step_time_s`, and optional grayscale PNGs. The verdict prints as `physics_check=PASS|FAIL|SKIPPED` and **does not** set the exit status unless `--strict` is passed |
| **Model maturity** | numerical verification: **analytical** and **regression** -- the end-of-run check compares the measured interface velocity with the sharp-interface solvability result plus the curvature term, `v = (3/2) F eps sqrt(2M) - M/R`, to `+/-25%`; the measured residual is `1.6%` at the shipped preset and at most `2.3%` on any grid from `128^2` to `512^2` (`docs/report/data/allen_cahn_resolution_margin.csv`). Plus one pinned CPU checksum tied to a named machine and toolchain, CPU-vs-CUDA agreement to `1e-9` and CPU-vs-HIP to `1e-4`. There is no manufactured-solution check. Physical completeness: **canonical** -- the standard non-conserved double-well model with a constant driving force, no elastic/thermal/solutal coupling. Calibration: **none** |

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
64^2, 3.50x at 128^2 and 2.55x at 256^2 -- pass, fail, fail. `dR/dt` does not
have that defect. The last half of the run, because the Gaussian initial
condition is far from the equilibrium `tanh` and its collapse moves the
contour by a distance that scales with the seed; a whole-run average inherits
exactly the grid dependence the area ratio had.

### Why the prediction carries a curvature term

The measured object is a growing *disc*, and Allen-Cahn moves an interface by
mean curvature as well as by the bulk driving force:

    dR/dt = (3/2) F eps sqrt(2M) - M/R

The second term was ignored while the shipped driving force was large enough
to make `M/R` about 6% of the answer. It is not ignorable at a driving force
far enough under the bistability ceiling to be safe: at the current preset
`M/R` is 20% of the flat-front law, and it does *not* shrink with the grid,
because the seed scales with the box and so `R/R*` is nearly grid-independent.
Comparing a disc against a flat-front law would leave a 20% bias that no
amount of refinement removes -- and it looks exactly like a grid dependence.
With the curvature term in, `128^2` through `512^2` agree with the prediction
to `2.3%`.

The same term sets a **critical radius** `R* = M/v = 7.1` cells: a seed
smaller than that shrinks and vanishes however favourable the bulk driving
force is. That is why the default grid is `256^2` and not `64^2` -- the seed
`sigma = 0.055*min(nx,ny)` gives a `4.1`-cell seed at `64^2`, which dissolves.
The app prints `R*` next to `R0` and warns when the seed is subcritical.

### Two limits worth knowing before changing the parameters

These are **coupled**, which is the single most useful thing to know
before touching `epsilon`, `M` or `driving_force`. Measurements:
[`docs/report/data/allen_cahn_resolution_margin.csv`](../../docs/report/data/allen_cahn_resolution_margin.csv).

* **The interface has to span at least a couple of cells.** The equilibrium
  half-width is `eps*sqrt(2M)`, in cells because `dx = 1` is hard-coded, so
  resolving the interface means making `eps*sqrt(2M)` large -- refining the
  grid cannot do it. Below about `1.5` cells the front is pinned by the
  lattice: at `0.76` cells (the pre-0.2 preset) the measured speed is `-32%`
  off the continuum law at a safe driving force, at `1.5` cells `+0.05%`, and
  at `2.0` cells `+1.2%`. The app warns below `2.0` cells.
* **The driving force has a ceiling.** A front exists only while
  `F*eps^2 < 2/(3 sqrt 3) = 0.385`; above that there is no metastable
  `phi < 0` phase and the whole domain simply decays. `driving_force = 20` or
  `epsilon = 1.5` crosses it and flips the entire box, which the app says out
  loud. Nearing the ceiling is bad well before crossing it: at 94% of it the
  matrix well has moved to `phi = -0.69` and retains 1.5% of the symmetric
  barrier depth, and the `F eps^2 -> 0` law the app checks against is 26%
  slow.
* **Why they are coupled.** The pre-0.2 preset sat 6% under the ceiling *and*
  measured `+9%` against the flat law, which looked like agreement. It was
  two large errors of opposite sign: the lattice pinning the front by ~12%
  and the finite tilt speeding the continuum front by ~26%. Backing the
  driving force off -- the obvious way to buy margin -- removes only one of
  them, so the residual walks from `+17%` at the ceiling through `-6%` at
  70% of it to `-65%` at a quarter of it. There is no safe driving force at a
  0.76-cell interface. The resolution had to be fixed first; once it was, a
  factor-2.7 margin cost nothing.

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

Defaults if omitted: `nx=ny=256`, `n_steps=5000`, `dt=0.005`, `M=8.0`, `epsilon=0.75`, `driving_force=0.25`.
The app samples the global visible seed area (cells with `phi > 0`) at `t = 0`, `t/2`, `3t/4` and the end, converts each to an equivalent radius, and reports the interface velocity over the last half of the run against `(3/2) F eps sqrt(2M) - M/R`. The verdict prints as `physics_check=`; the exit status reports whether the *run* finished, so a deliberately short run is not a failed process. Pass `--strict` to have a `FAIL` verdict (never a `SKIPPED` one) set the exit status instead. The optional positive `driving_force` favors the `phi≈+1` seed over the `phi≈-1` matrix. If you pass one PNG path, it writes the final field; if two, the first is the initial snapshot and the second the final (grayscale, rank 0).
The reported step timing measures the time-stepping loop only, after an MPI barrier and before PNG output or verification; `avg_step_time_s` is based on the slowest rank.

Example:

```bash
cd build
mpirun -n 4 ./apps/allen_cahn/allen_cahn 256 256 5000 0.005 8.0 0.75 0.25 initial.png final.png
```

## See also

- [`../../docs/halo_exchange.md`](../../docs/concepts/halo_exchange.md) — halo policies (this demo uses separated layout for FD) 
- [`../../examples/15_finite_difference_heat.cpp`](../../examples/15_finite_difference_heat.cpp) — related FD + halo pattern in `examples/` 
