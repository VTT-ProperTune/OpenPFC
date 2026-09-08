<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Gradient elasticity (`apps/gradient_elasticity`)

Isotropic **strain-gradient elasticity** (Aifantis / Helmholtz–Navier). This
is the 0.2 application for GitHub issue `#82`.

Binaries: `gradient_elasticity` (CPU); `gradient_elasticity_hip` when
`OpenPFC_ENABLE_HIP_SPECTRAL` is on. The HIP twin FFTs on the device and
inverts the \(2\times 2\) hats on the host.

This is a **static** spectral solve, not ETD time-stepping. JSON still
needs a `timestepping` block because the shared Time parser requires
`t0 < t1` and `dt`; those values are unused.

## Physics

Classical isotropic elasticity has no intrinsic length and can produce
singular high-\(k\) fields near idealized defects. Gradient elasticity
penalizes strain gradients with an internal length \(\ell\), the distance
over which those gradients are costly. Features much smaller than \(\ell\)
are regularized.

A convenient isotropic prototype applies a Helmholtz operator to the
Navier operator:

\[
(1-\ell^2\nabla^2)\,L_{\mathrm{navier}}\,\mathbf{u}=\mathbf{f}.
\]

That is fourth order in \(\mathbf{u}\). On a periodic grid with constant
moduli the Fourier problem is a \(2\times 2\) algebraic system at each
wavevector \(\mathbf{k}\). Split \(\hat{\mathbf{f}}\) into longitudinal
and transverse parts:

\[
\hat{\mathbf{u}}_L=\frac{\hat{\mathbf{f}}_L}{-\alpha(\lambda+2\mu)k^2},
\qquad
\hat{\mathbf{u}}_T=\frac{\hat{\mathbf{f}}_T}{-\alpha\mu k^2},
\qquad
\alpha=1+\ell^2 k^2.
\]

The \(k=0\) (rigid) mode is projected to \(\hat{\mathbf{u}}=0\). Optional
sixth-order stretch: `order=6` uses \(\alpha=(1+\ell^2 k^2)^2\); `ell4>0`
uses \(\alpha=1+\ell^2 k^2+\ell_4^4 k^4\).

The shipped loading is a **periodic dilatational eigenstrain**
\(\varepsilon^*=\varepsilon_0 g\,I\) in a 2-D slab (`ux`, `uy`). The
equivalent body force is \(\mathbf{f}=2(\lambda+\mu)\varepsilon_0\nabla g\).
For a cosine inclusion \(g=A\cos(\mathbf{k}\cdot\mathbf{x})\) the
displacement is the sine field
\(\mathbf{u}=B\mathbf{k}\sin(\mathbf{k}\cdot\mathbf{x})\) with
\(B=2(\lambda+\mu)\varepsilon_0 A/(\alpha(\lambda+2\mu)k^2)\). As
\(\ell\to 0\), \(\alpha\to 1\) and the classical Navier solution is
recovered. Finite \(\ell\) reduces high-\(k\) content.

The \(2\times 2\) invert stays in the app (`invert` / `solve_displacement`).
`SpectralDiagonalSolver` is scalar-only and is not used here.

## Run

```bash
mkdir -p results/gradient_elasticity
mpirun -n 1 ./apps/gradient_elasticity/gradient_elasticity \
  ../apps/gradient_elasticity/inputs_json/inclusion.json
mpirun -n 1 ./apps/gradient_elasticity/gradient_elasticity \
  ../apps/gradient_elasticity/inputs_json/gaussian.json
```

VTK of `g`, `ux`, and `uy` goes to `results/gradient_elasticity/`.

JSON `model.params`: `mu`, `lambda`, `ell`, `eps0`, optional `order`
(4 or 6), optional `ell4`. `E` and `nu` may replace `mu`/`lambda`
(plane-strain Lamé conversion). Initial conditions: `"type":
"cosine_mode"` (`g0` / `amplitude` / `nx` / `ny` / `nz`) or
`"gaussian_inclusion"` (`amplitude` / `sigma`, optional `g0` / `x0` /
`y0`).

## Tests

`ctest -R gradient-elasticity` checks the Helmholtz factor, the
longitudinal invert, \(k=0\) projection, \(\ell\to 0\) recovery of
classical elasticity, analytical Fourier displacement for a cosine
inclusion, high-\(k\) reduction at finite \(\ell\), and mean
\(\mathbf{u}=0\). HIP builds add `HIP_GradientElasticity` and
`gradient-elasticity-hip-smoke`.

## Layout

| Path | Role |
|------|------|
| `include/gradient_elasticity/gradient_elasticity_physics.hpp` | \(\alpha(k)\), \(2\times 2\) invert, eigenstrain force |
| `include/gradient_elasticity/gradient_elasticity_solve.hpp` | FFT → host invert → IFFT |
| `include/gradient_elasticity/gradient_elasticity_session.hpp` | JSON session, fields `g`/`ux`/`uy` |
| `src/gradient_elasticity.cpp` / `src/hip/` | CPU / HIP `main` |
| `inputs_json/inclusion.json` | Cosine eigenstrain |
| `inputs_json/gaussian.json` | Localized Gaussian inclusion |
