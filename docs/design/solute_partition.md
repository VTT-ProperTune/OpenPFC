<!-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Stationary solute partition on one bounded axis

This is one conserved solute line with a prescribed partition factor. It is
the first alloy-side extension that can be checked on its own. It is not an
aluminum calibration, and the aluminum application is not read.

The alloy roadmap also asks for focused child issues for the extensions that
are selected. This file does not open those issues. Solved heat, melt flow,
and a moving solidification front are not selected here.

## Equation

One non-periodic `DiffusionAxis` holds one composition sample `c` and one
partition factor `K > 0` at each node. The factor does not move.
The composition and partition views must not overlap, including partially
shifted spans of the same allocation; such a write would silently change the
prescribed factor. Overlap is rejected before the ledger is opened. The chemical
potential and the balance, in code units, are

μ = c / K,

∂t c = ∂x(M ∂x μ),

with constant mobility `M > 0`. There is no SI conversion and no alloy
constant. Both ends use a zero outward constitutive flux, so solute does not
cross the boundary. `FluxLedger` records that flux. Positive outward flux
would remove inventory. The integrator is RK4. The same step advances every
node. There is no subcycling.

`DiffusionAxis::apply` receives μ and the mobility. The header does not
invent a second diffusion operator.

## Equilibrium

A constant potential is a discrete equilibrium, because the public operator
annihilates a constant and the ends carry no flux. The weighted mass

m = Σ w_i c_i

is therefore conserved, and the equilibrium samples are

c_i = K_i m / Σ w_j K_j.

A factor below one on a prescribed solid side holds less solute than the
liquid side at the same potential. The usual rejection of solute into the
liquid is that statement. The solid set is chosen by the caller. This line
does not find a front.

## What this does not claim

The factor is prescribed. There is no crystalline density, no evolving order
parameter, and no moving solidification front. There is no melt flow and no
latent heat. A sharp embedded interface is a different contract. Host only.
A periodic axis has no face condition; passing the closed ends is rejected
by `DiffusionAxis`, the ledger trial is dropped, and the composition array
is left unchanged.
