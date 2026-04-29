<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Heat3D (`heat3d`)

MPI CPU driver for the **3D heat equation** \(\partial u/\partial t = D \nabla^2 u\) on a uniform periodic brick \([0,N)^3\) with spacing 1, comparing:

- **Finite differences**: explicit Euler with separated face halos and spatial order **2, 4, or 6** (central Laplacian; see `laplacian_*_interior_separated` in `include/openpfc/kernel/field/finite_difference.hpp`).
- **Spectral**: HeFFTe-backed real FFT, semi-implicit step in Fourier space (same pattern as `examples/diffusion_model.hpp`).

Initial condition matches the diffusion examples: \(u(\mathbf{x},0)=\exp(-|\mathbf{x}|^2/(4D))\) with origin at \((0,0,0)\).

## Build

Enabled with `OpenPFC_BUILD_APPS=ON` (default). Requires HeFFTe (spectral path uses the same FFT stack as the library).

## Usage

```text
heat3d fd <N> <n_steps> <dt> <D> <fd_order>
heat3d spectral <N> <n_steps> <dt> <D>
```

- `fd_order`: `2`, `4`, or `6`. Halo width is `fd_order/2` (1, 2, or 3 ghost layers per face).
- **Stability (FD)**: explicit Euler requires a sufficiently small `dt` (CFL); higher spatial order does not remove this limit.
- **Spectral**: each step applies \(\hat u \leftarrow \hat u / (1 + \Delta t\,D|\mathbf{k}|^2)\) in complex Fourier space (implicit Euler for the linear diffusion operator).

## Example

From your build directory:

```bash
mpirun -n 4 ./apps/heat3d/heat3d fd 64 200 0.001 1.0 4
mpirun -n 4 ./apps/heat3d/heat3d spectral 64 200 0.001 1.0
```

Rank 0 prints `timing_s`, `avg_step_time_s` (MPI max across ranks), and an RMS L2 error against the **infinite-domain** Gaussian reference

\[
u(\mathbf{x},t)=(1+t)^{-3/2}\exp\!\left(-\frac{|\mathbf{x}|^2}{4D(1+t)}\right)
\]

The domain is **periodic**; for a localized Gaussian the mismatch at boundaries contributes to the reported error even when the numerics are consistent.

## See also

- [`examples/15_finite_difference_heat.cpp`](../../examples/15_finite_difference_heat.cpp) — FD + halos pattern
- [`examples/diffusion_model.hpp`](../../examples/diffusion_model.hpp) — spectral diffusion IC and operator
