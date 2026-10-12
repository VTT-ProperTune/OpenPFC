<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Residual-based inexact Newton

`openpfc/solvers/newton.hpp` composes matrix-free GMRES with a nonlinear
model. Supply `residual(x,r)`, `linearize(x)`, `jacobian(v,Jv)` and
`precondition(r,z,iteration)`. The same vector-operations contract used by
Krylov keeps the algorithm independent of the discretization.

The model also supplies `begin_trial(x)`, `accept_trial(x)` and
`reject_trial()`. These hooks manage reversible iterate caches. Permanent
material history stays with the caller, which commits an increment only
after independent physical admission. A Jacobian action must read unchanged
committed history. Rejected trials preserve the accepted vector; residual
exceptions reject the open trial before propagating.

The forcing criterion is `||J*s+R|| <= eta*||R||`, verified using the true
linear residual. Adaptive forcing implements Eisenstat and Walker's
choice 2 with its safeguard, gamma 0.9 and power 2 by default. Set
`adaptive_forcing=false` and `initial_forcing` to the fixed linear tolerance
for a tight-Newton comparison. `linear.absolute_tolerance` is deliberately
ignored inside Newton, so it cannot bypass the forcing criterion.

Armijo backtracking requires residual decrease. Nonlinear convergence uses
an explicitly evaluated residual with the same relative/absolute target
for fixed and adaptive forcing. A loose linear solve cannot establish
nonlinear convergence. Results report forcing, rejected trials, step
lengths, linear admission, residual calls and linearization costs.

`finite_strain_fft/inexact.hpp` supplies `solve_nonlinear_increment`.
It starts at the accepted deformation plus the prescribed macroscopic
step and solves projected equilibrium. This uses a residual target rather
than the correction-based stopping rule of `solve_increment`; callers
should record that distinction when comparing drivers.

The FFT adapter collectively admits local stress and tangent evaluations
before projection. A constitutive exception propagates on every rank;
peers never continue into an FFT collective while one rank unwinds. Pass
the same communicator used by the projector. Generic distributed models
must likewise synchronize local failures before their collective actions.

The forcing algorithm is standard, based on Eisenstat and Walker,
[SIAM J. Sci. Comput. 17 (1996), 16–32](https://doi.org/10.1137/0917003).
