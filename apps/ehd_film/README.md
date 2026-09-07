<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# EHD film (`apps/ehd_film`)

Elastohydrodynamic **thin film under a flexible plate**. This is the 0.2
application for GitHub issue `#81`.

Binaries: `ehd_film` (CPU); `ehd_film_hip` when `OpenPFC_ENABLE_HIP_SPECTRAL`
is on.

## Physics

A viscous gap \(h\) supports a Kirchhoff plate. Bending pressure, lubrication
flow, and mass conservation give a sixth-order evolution:

1. Plate curvature \(\kappa\sim\nabla^2 h\) stores bending energy
   \(\tfrac12 B(\nabla^2 h)^2\).
2. The variational pressure is \(p=B\nabla^4 h\) (plus optional tension
   and disjoining).
3. The pressure gradient drives a viscous flux \(M_0\nabla p\).
4. Mass conservation \(\partial_t h=\nabla\cdot[M_0\nabla p]\) produces
   \(\nabla^6\).

\[
p=B\nabla^4 h-\gamma\nabla^2 h-\Pi(h),\qquad
\partial_t h=\nabla\cdot\bigl[M_0\nabla p\bigr].
\]

With OpenPFC \(k_{\mathrm{lap}}=-|k|^2\),

\[
L(k)=M_0 B\,k_{\mathrm{lap}}^3-M_0\gamma\,k_{\mathrm{lap}}^2
     -M_0\Pi'(h_0)\,k_{\mathrm{lap}},
\qquad
\lambda(k)=-M_0 B k^6-M_0\gamma k^4+M_0\Pi'(h_0)k^2.
\]

Defaults \(A=\gamma=0\) are pure bending relaxation:
\(h_k(t)=h_k(0)\exp(-M_0 B k^6 t)\). An explicit step on spacing
\(\Delta x\) would need \(\Delta t\lesssim\Delta x^6/(M_0 B\pi^6)\);
ETD treats \(k^6\) as a multiply. Optional \(A>0\) is van der Waals
versus bending (a finite unstable band).

## Run

```bash
mkdir -p results/ehd_film
mpirun -n 1 ./apps/ehd_film/ehd_film \
  ../apps/ehd_film/inputs_json/relaxation.json
```

The shipped input is a 128² corrugation under a stiff plate. VTK of `h`
goes to `results/ehd_film/`. Short waves flatten first (\(k^6\)).

JSON `model.params`: `h0`, `B`, `M0`, `gamma`, `A`. Initial condition
`"type": "cosine_mode"` with `h0` / `amplitude` / `nx` / `ny` / `nz`.

## Tests

`ctest -R ehd-film` checks \(L(k)=M_0 B k_{\mathrm{lap}}^3\), exact
\(\exp(-M_0 B k^6 t)\), two-mode rate ratio 64, mean-gap conservation,
and that \(A>0\) grows at low \(k\). HIP builds add `HIP_EhdFilmETD` and
`ehd-film-hip-smoke`.

## Layout

| Path | Role |
|------|------|
| `include/ehd_film/ehd_film_physics.hpp` | \(L(k)\) with \(k_{\mathrm{lap}}^3\) |
| `include/ehd_film/ehd_film_pointwise.hpp` | \(\Pi\) remainder |
| `include/ehd_film/ehd_film_session.hpp` | JSON session, field `h`, 2/3 dealias |
| `src/ehd_film.cpp` / `src/hip/` | CPU / HIP `main` |
| `inputs_json/relaxation.json` | Bending-driven leveling |
