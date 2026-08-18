# wave2d — 2D acoustic wave (coupled first-order system)

This application integrates the 2D wave equation
\(u_{tt} = c^2 (u_{xx}+u_{yy})\) on an `nz=1` slab as the coupled first-order system
\(\partial_t u = v\), \(\partial_t v = c^2 (u_{xx}+u_{yy})\) with explicit Euler in time.

- **Periodic in x** (and z slab) via MPI halo exchange.
- **Physical y boundaries**: homogeneous Dirichlet (`u=u_\mathrm{wall}`, `v=0` on the wall)
  or homogeneous Neumann (zero normal derivative of `u`, implemented via mirrored face halos).

## Binaries

| Target | Description |
|--------|-------------|
| `wave2d_fd_manual` | Second-order central stencil on `Field`, non-blocking halos, laboratory-style loop. |
| `wave2d_fd` | Same BC model; spatial accuracy `fd_order` 2,4,…,20 via tabulated central stencils. |
| `wave2d_cuda` | Device path (optional): same positional CLI as `wave2d_fd_manual` plus optional `--vtk` / `--vtk-every`; host orchestrates halos + y-face patch, CUDA kernel for Laplacian + Euler. |
| `wave2d_hip` | HIP analogue of `wave2d_cuda` (same CLI and VTK options). Halos use `SparseExchange<HipSpace>` on device Fields; y-face BC patches stay on device. Rank 0 prints `WAVE2D_HIP_HALO_MODE`. |

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
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>

// Create halo exchanger from field and decomposition
PaddedHaloExchanger<double> halo_u(u, decomp, rank, MPI_COMM_WORLD);

// Perform halo exchange
pfc::communication::exchange(halo_u);
```

### Field checkpointing

Fields support checkpointing via the `state_capture.hpp` utilities:

```cpp
#include <wave2d/state_capture.hpp>

// Capture field state (halos excluded)
auto state = wave2d::capture_uv(u, v, std::nullopt);

// Restore field state (validates before mutating)
auto outcome = wave2d::restore_uv(state, u, v, std::nullopt);
```

See `apps/wave2d/include/wave2d/state_capture.hpp` for full checkpointing API with validation and metadata support.

## Tests

With `OpenPFC_BUILD_TESTS=ON` and Catch2 available, `ctest -R wave2d` runs `test_wave2d` and, when CUDA/HIP are enabled, the CPU–device parity tests.
