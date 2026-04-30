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
| `wave2d_fd_manual` | Second-order central stencil on `PaddedBrick`, non-blocking halos, laboratory-style loop. |
| `wave2d_fd` | Same BC model; spatial accuracy `fd_order` 2,4,…,20 via tabulated central stencils. |
| `wave2d_cuda` | Device path (optional): same positional CLI as `wave2d_fd_manual` plus optional `--vtk` / `--vtk-every`; host orchestrates halos + y-face patch, CUDA kernel for Laplacian + Euler. |
| `wave2d_hip` | HIP analogue of `wave2d_cuda` (same CLI and VTK options). |

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

## Tests

With `OpenPFC_BUILD_TESTS=ON` and Catch2 available, `ctest -R wave2d` runs `test_wave2d` and, when CUDA/HIP are enabled, the CPU–device parity tests.
