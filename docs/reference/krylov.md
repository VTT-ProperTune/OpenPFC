<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Matrix-free Krylov solvers

`openpfc/solvers/krylov.hpp` provides restarted GMRES, flexible right GMRES
and CG. The algorithms accept a vector-operations object, an action
`A(x,y)` and an inverse-preconditioner `M(r,z,iteration)`. They do not
require a matrix or a discretization type. `VectorOperations<double>`
provides host vectors and optional MPI reductions; another implementation
can supply the same `make`, `copy`, `scale`, `axpy`, `dot` and `norm` methods.

```cpp
pfc::solvers::VectorOperations<> ops(MPI_COMM_WORLD);
pfc::solvers::KrylovControls controls;
controls.restart = 30;
controls.relative_tolerance = 1e-9;
auto result = pfc::solvers::gmres(ops, action, rhs, solution, controls);
```

The initial guess is supplied in `solution`. Convergence requires the true
unpreconditioned residual to satisfy
`||b-Ax|| <= max(absolute_tolerance, relative_tolerance*||b||)`.
An Arnoldi estimate alone cannot admit a solution. The result distinguishes
convergence, an iteration limit, breakdown, stagnation and nonfinite data.
Records contain residual estimates and true residuals at restart/admission
checks. Operator, preconditioner and reduction timings are available.

Right preconditioning stores each physical direction separately. With
`flexible=true`, the inverse may vary between iterations. Left
preconditioning requires a fixed linear inverse and `flexible=false`.
`Preconditioning::None` bypasses the inverse. Fixed right preconditioning
also uses stored directions; its inverse must obey the fixed contract.
All MPI ranks must execute the same collective sequence. The adapter owns
compatibility, boundary conditions and nullspace constraints.

CG remains available for an operator and fixed inverse that the caller
has established to be symmetric positive definite on the solved subspace.
A random symmetry probe, positive sampled curvature or successful CG run
cannot establish that property. CG checks curvature and verifies its true
residual, but those checks are safeguards rather than a proof.

The finite-strain FFT Newton driver defaults to GMRES. Set
`NewtonControls::krylov_method = KrylovMethod::CG` and
`operator_is_spd = true` only with a mathematical justification on the
compatible zero-mean subspace. Its nonlinear correction stopping rule is
unchanged; the caller should independently verify projected equilibrium.
