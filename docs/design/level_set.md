<!-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Periodic Hamilton–Jacobi level set

One cell-centred field on a periodic Cartesian grid. Cell `i` sits at
`(i + 1/2) hx`. The stencil reads one neighbour, so the halo width is 1.
An x-slab takes those two planes from the caller. The periodic driver
fills them from the owned edges. Coefficients are code units.

Advection is `phi_t = -v · grad phi`, with a first-order upwind difference.
Normal motion is `phi_t = -V_n |grad phi|`, with the Osher–Sethian Godunov
choice: a non-negative speed takes the positive backward part and the
negative forward part. A negative speed takes the opposite pair. A
dimension of length 1 contributes no derivative.

Redistancing is the Sussman form. The sign is frozen from the caller's
field, smoothed by `s = phi0 / sqrt(phi0^2 + h^2)` with `h` the smallest
spacing. The rate is `-s |grad phi| + s`. Time stepping is SSP-RK2. The scalar helper `level_set_stable_dt` estimates `cfl * min(h) / max|speed|`
for a single characteristic direction (and is used for redistancing).
That is **not** a universal multi-dimensional advection limit. For an
upwind advection field use `level_set_advection_stable_dt` instead:
`dt <= cfl / max_cells(|vx|/hx + |vy|/hy + |vz|/hz)`, dropping singleton
axes. This is the monotone SSP-RK2 directional Courant condition.
The caller is responsible for enforcing a stable `dt`; the stepper rejects
nonfinite updates but does not automatically limit an oversized positive step.

A failed check throws before the field is replaced. Volume is the measure
of cells where `phi < 0`. The centroid is that set's cell-centre average.
The +x crossing interpolates the first sign change and does not wrap.

`LevelSetCuda` repeats the periodic step on one device. The two rate
evaluations and the two linear combinations stay on the device, and the
field comes back once. HIP has the kernel entry and was not run.

This is one interface. It does not evolve a grain network, does not couple
to fracture, and does not claim that the basic level set conserves volume.
