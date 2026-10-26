<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Conservative fluxes on padded device fields

Include `<openpfc/kernel/field/padded_face_flux.hpp>` and the selected GPU
field-storage headers. `pfc::field::fd::padded_face_flux(coefficient, potential)`
binds the existing `Field<double, Space>` buffers. It returns a device-safe
face evaluator for `a_face * grad(p)`; arithmetic averaging is the default,
with `FaceAverage::Harmonic` available for non-negative coefficients.

Prepare both fields' halos before every stage. The factory uploads stale
host mirrors but does not exchange halos or read the fields back. Compose
its result with `face_divergence` and call that evaluator at each owned
cell from the existing resident RHS kernel. The fields must outlive the
evaluator and their storage must remain stable.

Use `model_walls()` on MPI cuts and periodic seams. Set `walls.xm` or
`walls.xp` to `{WallPolicy::Zero, 0}` only on ranks touching those physical
walls. Even reflection of scalar ghosts alone does not enforce zero flux
for every constitutive law. For an arbitrary anisotropic or anti-trapping
current, supply the model's face functor directly to `face_divergence`.

The update convention is `du/dt = +div(a grad(p))`. A physical current in
`du/dt = -div(J)` must instead be passed with the opposite sign. Evaluations
of a shared face must agree across ranks for the global balance to telescope.

Package tests check independent finite-volume balances, arithmetic/harmonic
averaging, different padding widths and thin local extents. GrainFront's
replacement and CPU/CUDA parity checks remain downstream acceptance work;
these package checks do not establish that adoption.
