<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# wave2d — 2D acoustic wave (coupled first-order system)

This application integrates the 2D wave equation
\(u_{tt} = c^2 (u_{xx}+u_{yy})\) on an `nz=1` slab as the coupled first-order system
\(\partial_t u = v\), \(\partial_t v = c^2 (u_{xx}+u_{yy})\) with explicit Euler in time.

- **Periodic in x** (and z slab) via MPI halo exchange.
- **Physical y boundaries**: homogeneous Dirichlet (`u=u_\mathrm{wall}`, `v=0` on the wall)
  or homogeneous Neumann (zero normal derivative of `u`, implemented via mirrored face halos).

## Problem setup

The `Problem setup` block the report contract (`#112`) asks every application
to carry. Command-line driven, no JSON, and no science preset: every
configuration is a demonstration or a test. Note also that the application
**reports no error against any reference solution** -- unlike `heat3d`, its
summary line carries an amplitude, not an accuracy.

| Item | Description |
|---|---|
| **Use case** | A pulse reflecting between the two walls of an acoustic slab; the smallest coupled two-field example in the tree |
| **Question** | How does the pulse reflect, and does the character of the reflection depend on whether the wall is rigid (Neumann) or pressure-releasing (Dirichlet)? |
| **Domain** | `Nx x Ny x 1` slab, `dx = dy = 1`, origin `(0,0,0)`. Defaults `Nx = Ny = 64` |
| **Grid/time** | `Nx Ny n_steps dt [fd_order] y_bc [u_wall]`; defaults 200 steps at `dt = 0.01`, second order. `fd_order` even in `[2,20]` for `wave2d_fd`; the manual and device drivers are fixed at second order. VTK cadence is `--vtk-every` (default 1) |
| **Boundary conditions** | **Periodic in x** by MPI halo exchange; **physical in y**, either Dirichlet (`u = u_wall`, `v = 0` on the wall, imposed by an odd ghost mirror *and* by writing the owned boundary cells) or Neumann (even mirror). The y correction runs after the unconditional periodic exchange and overwrites the ghosts it produced |
| **Initial condition** | Centred Gaussian bump in `u` with `v = 0`: `sigma = 0.12*min(Nx,Ny)`, centre `(0.5*(Nx-1), 0.5*(Ny-1))`. Each test picks its own `sigma`; there is no single canonical parameter set |
| **Key physical parameters** | Wave speed `c = wave2d::kC = 1.0`, a compile-time constant not exposed on the command line. `u_wall` defaults to 0 and only affects Dirichlet runs. Nondimensional throughout |
| **Observable** | `global_rms_u_interior` with the `interior_cells` count it was averaged over; `cfl_c_dt_dx`; `timing_s` / `avg_step_time_s`; optional VTK series of `u`. "Interior" is the **global** interior: a `fd_order/2`-wide shell is trimmed off each global axis that can spare one, so the z axis of this `nz == 1` slab is kept whole and the value does not depend on the rank count. A configuration whose interior is empty prints `global_rms_u_interior=undefined interior_cells=0` and exits non-zero rather than printing a `0` that means "nothing was summed" |
| **Model maturity** | numerical verification: **manufactured** and **regression** -- RK2/RK4 temporal order is measured against a self-generated fine-timestep RK4 reference, not a closed form; the rest is a pinned CPU checksum, CPU-vs-device parity, cross-implementation parity and hand-checked boundary ghosts. There is **no analytical check** anywhere in this app. Physical completeness: **canonical** -- the constant-coefficient scalar wave equation, with no damping, no heterogeneous medium and no source. Calibration: **none** |

### What the two directions mean physically

Periodicity in x makes the slab infinitely long, or equivalently a ring: a
pulse leaving the right edge arrives at the left, so there is no along-slab
attenuation and no far field. The y walls are the physically interesting
idealization: Dirichlet is a pressure-release boundary and the reflected pulse
comes back inverted; Neumann is a rigid one and it does not. **Neither
absorbs.** There is no radiation condition and no absorbing layer anywhere in
this app, so energy put into the slab never leaves it, and a long run is a
reverberant box rather than a propagation experiment.

Note also that the RK convergence test exercises the RK2/RK4 steppers, while
the shipped drivers step with explicit Euler -- that test measures the
steppers, not the binaries.

## Binaries

| Target | Description |
|--------|-------------|
| `wave2d_fd_manual` | Second-order central stencil on `Field`, non-blocking halos, laboratory-style loop. |
| `wave2d_fd` | Same BC model; spatial accuracy `fd_order` 2,4,…,20 via tabulated central stencils. Every advertised order runs: the halo exchange is restricted to the in-plane `±X`/`±Y` faces (`halo::presets::Axes2D()`), because a 1-thick z cannot host a `fd_order/2`-thick send slab and the Laplacian never reads `k±1` anyway. |
| `wave2d_cuda` | Device path (optional): same positional CLI as `wave2d_fd_manual` plus optional `--vtk` / `--vtk-every`; host orchestrates halos + y-face patch, CUDA kernel for Laplacian + Euler. |
| `wave2d_hip` | HIP analogue of `wave2d_cuda` (same CLI and VTK options). Halos use `SparseExchange<HIPSpace>` on device Fields; y-face BC patches stay on device. Rank 0 prints `WAVE2D_HIP_HALO_MODE`. |

## Usage

```bash
# Manual (fixed 2nd-order space): Nx Ny n_steps dt y_bc [u_wall]
mpirun -n 4 ./wave2d_fd_manual 128 128 500 0.01 neumann

# Higher-order FD: Nx Ny n_steps dt fd_order y_bc [u_wall]
mpirun -n 4 ./wave2d_fd 128 128 500 0.01 4 dirichlet 0.0

# CUDA / HIP: same positionals as manual; optional VTK for ParaView comparison
mpirun -n 2 ./wave2d_cuda 128 128 500 0.01 neumann --vtk out/gpu_%04d.vti --vtk-every 25
```

`y_bc` is `dirichlet` or `neumann` (short forms `d` / `n` accepted). `u_wall` defaults to `0` and only affects Dirichlet runs.

### VTK / ParaView (optional)

Supported on **CPU** (`wave2d_fd`, `wave2d_fd_manual`) and **GPU** (`wave2d_cuda`, `wave2d_hip`). Append `--vtk <pattern>` to write `u` as VTK ImageData (`.vti`, parallel `.pvti` + rank pieces). Use a pattern with a time index, e.g. `out/u_%04d.vti`. Frame `0` is the initial state; later frames use the 1-based step index after each batch of completed steps. `--vtk-every k` saves every `k` steps (default `1`). Open the `.pvti` (multi-rank) or `.vti` (single rank) time series in ParaView for animation. For CPU vs GPU comparisons, use different prefixes or directories (e.g. `cpu/u_%04d.vti` vs `gpu/u_%04d.vti`).

## Stability (CFL)

Explicit Euler requires a sufficiently small \(\Delta t\) (roughly \(\Delta t \lesssim C \,\Delta x / c\) with constant \(C\) of order unity for second-order waves). Reduce `dt` if the run blows up.

## Field vocabulary

Wave2D uses `pfc::data::Field` from `<openpfc/kernel/data/grid_field.hpp>` as its canonical field container. This replaces legacy field types and provides a unified interface for grid-based computations with automatic halo management.

### Field creation from decomposition

Fields in wave2d are created from MPI decomposition geometry using `field_from_subdomain`:

```cpp
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

// Create domain and decomposition
auto domain = pfc::domain::create(
    pfc::GridSize({Nx, Ny, 1}),
    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
    pfc::GridSpacing({1.0, 1.0, 1.0}));
auto decomp = pfc::decomposition::create(domain, nproc);

constexpr int halo_width = 1;

// Type specification: pfc::data::Field<double, pfc::HostSpace>
pfc::data::Field<double, pfc::HostSpace> u =
    pfc::data::field_from_subdomain<double>(decomp, rank, halo_width);
pfc::data::Field<double, pfc::HostSpace> v =
    pfc::data::field_from_subdomain<double>(decomp, rank, halo_width);
pfc::data::Field<double, pfc::HostSpace> lap =
    pfc::data::field_from_subdomain<double>(decomp, rank, halo_width);
```

The `field_from_subdomain` function creates a field whose layout matches the subdomain decomposition, ensuring halo compatibility across MPI ranks. For face-halo layouts (unpadded storage with iteration halos), use `field_from_subdomain_unpadded`.

### Field member access and methods

`pfc::data::Field` provides element access, iteration, and coordinate mapping:

```cpp
// Element access (halo cells included)
u(i, j, k) = value;
double val = u(i, j, k);

// Get local dimensions
const auto local_size = u.local_size();  // returns Int3 (nx, ny, nz)

// Physical coordinates of local index (i,j,k)
const auto coords = u.coords(i, j, k);  // returns Real3 (x, y, z)

// Initialization using physical coordinates
u.apply([](double x, double y, double z) {
    return std::exp(-(x*x + y*y) / (2.0 * sigma * sigma));
});

// Iterate over owned cells (no halo cells)
u.for_each_owned([&](int i, int j, int k) {
    const double v0 = v(i, j, k);
    const double l = lap(i, j, k);
    u(i, j, k) += dt * v0;
    v(i, j, k) += dt * wave2d::kC * wave2d::kC * l;
});

// Raw data access (for interop with C-style APIs)
double* data_ptr = u.data();
std::size_t total_size = u.size();
```

### Halo integration with fields

Fields integrate seamlessly with halo exchangers for MPI communication:

```cpp
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>

pfc::comm::HaloExchange<pfc::HostSpace, double> halo_u(u, decomp, rank,
                                                       MPI_COMM_WORLD);
halo_u.exchange();
```

### Field checkpointing

Filesystem restart uses `pfc::sim::CheckpointService` (`checkpoint.every` /
`checkpoint.directory` / `restart_from`). Owned `u` and `v` cells are
published as MPI-IO bricks plus `metadata.json`; halos are recomputed
after load. See [`docs/development/checkpoint_publish.md`](../../docs/development/checkpoint_publish.md).

## Tests

With `OpenPFC_BUILD_TESTS=ON` and Catch2 available, `ctest -R wave2d` runs `test_wave2d` and, when CUDA/HIP are enabled, the CPU–device parity tests.
