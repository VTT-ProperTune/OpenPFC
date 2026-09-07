<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Surface diffusion (`apps/surface_diffusion`)

Mullins **small-slope surface diffusion**: thermal smoothing of nanoscale
roughness. This is the 0.2 application for GitHub issue `#79`.

Binaries: `surface_diffusion` (CPU); `surface_diffusion_hip` when
`OpenPFC_ENABLE_HIP_SPECTRAL` is on.

## Physics

For a small-slope, isotropic free surface the height \(h\) obeys Mullins
surface diffusion. The chain is:

1. Mean curvature is \(\kappa\approx-\nabla^2 h\).
2. Chemical potential is \(\mu\propto\gamma\kappa\), so \(\mu\propto-\nabla^2 h\).
3. Atoms hop from high to low \(\mu\); the surface flux is
   \(j\propto-\nabla\mu\propto\nabla(\nabla^2 h)\).
4. Mass conservation on the height field is \(\partial_t h=-\nabla\cdot j\),
   which produces the biharmonic

\[
\partial_t h = -B\nabla^4 h.
\]

\(B\) lumps surface diffusivity, surface energy, atomic volume and temperature
(grid units here). Every Fourier mode decays independently:

\[
h_k(t)=h_k(0)\exp(-B|k|^4 t).
\]

Short wavelengths therefore disappear much faster than long ones — the
annealing / roughness-stability story. With OpenPFC
\(k_{\mathrm{lap}}=-|k|^2\),

\[
L(k)=-B\,k_{\mathrm{lap}}^2.
\]

There is no real-space nonlinearity.

## Run

```bash
mkdir -p results/surface_diffusion
mpirun -n 1 ./apps/surface_diffusion/surface_diffusion \
  ../apps/surface_diffusion/inputs_json/smoothing.json
```

The shipped input is a 128² slab with three cosine wavelengths. VTK of `h`
goes to `results/surface_diffusion/`. Short modes flatten first.

JSON `model.params`: `B` (default 1). Initial condition `"type":
"cosine_mode"` with `h0` and either a single `amplitude`/`nx`/`ny`/`nz` or a
`modes` array.

## Mode amplitude versus time

`smoothing.json` writes `h_%04d.vti`. Open the series in ParaView as a time
sequence: the \((n_x,n_y)=(16,8)\) ripple is gone well before \((2,1)\).

To plot amplitude versus time, take the same inner product as
`tests/test_surface_diffusion.cpp` on each snapshot: for a mode
\((n_x,n_y)\) with wavevector \(k=2\pi(n_x/L_x,n_y/L_y)\),

\[
A(t)=\frac{\sum h\,w}{\sum w^2},\qquad
w=\cos(k_x x+k_y y),
\]

and compare to \(A(0)\exp(-B|k|^4 t)\). Catch2 already checks a single mode
against that exponential and a two-mode rate ratio of 16
(\((k_2/k_1)^4\) with \(n_x=2\) vs \(n_x=1\)). Mean height is conserved.

## Tests

`ctest -R surface-diffusion` checks \(L(k)=-B k_{\mathrm{lap}}^2\), exact
single-mode \(\exp(-B k^4 t)\), two-mode decay rates in the ratio
\((k_2/k_1)^4=16\), and mean-height conservation. HIP builds add
`HIP_SurfaceDiffusionETD` and `surface-diffusion-hip-smoke`. LUMI-G smoke:
job 21791182 (`small-g`, 16², mean \(h=0\)).

## Layout

| Path | Role |
|------|------|
| `include/surface_diffusion/surface_diffusion_physics.hpp` | `L(k)=-B k_{\mathrm{lap}}^2` |
| `include/surface_diffusion/surface_diffusion_pointwise.hpp` | Zero remainder |
| `include/surface_diffusion/surface_diffusion_session.hpp` | JSON session, field `h` |
| `src/surface_diffusion.cpp` / `src/hip/` | CPU / HIP `main` |
| `inputs_json/smoothing.json` | Multi-wavelength annealing demo |
