<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Structured host geometric multigrid

`openpfc/solvers/multigrid.hpp` supplies a structured-grid V-cycle with
trilinear physical-coordinate transfers, weighted Jacobi smoothing and a
factored dense coarse solve. A factory supplies the sparse auxiliary
operator; the hierarchy does not contain element types or constitutive
history. Construction is separate from repeated inverse applications.

```cpp
using namespace pfc::solvers;
StructuredGrid grid{{33,33,33},3,Boundary::Periodic};
Multigrid inverse(grid, [](const StructuredGrid &g) {
  return elastic_auxiliary(g, 1.0, 0.5);
});
// inverse(residual, correction, krylov_iteration);
```

Periodic sizes coarsen by `(n+1)/2`, down to three nodes per axis. Odd
periodic grids are not nested: transfers interpolate at fixed physical
coordinates. Dirichlet grids include the two boundary nodes and use zero
boundary corrections. Restriction is exactly the transpose of
prolongation. Operators are volume weighted, so this is a weak-form
transfer convention rather than unscaled pointwise residual averaging.

`CoarseOperator::Rediscretized` calls the factory on each coarse grid.
`Galerkin` explicitly assembles `P^T A P`. The latter preserves the fine
energy but costs sparse products and can grow its stencil. Both paths
have independent transfer and random-load verification. Coarse solves
add a componentwise mean gauge before factorization for periodic grids;
constant or translation modes are removed throughout the hierarchy.
The supplied operator must have these nullspaces and a positive diagonal;
positive diagonal checks alone do not establish positive definiteness.

`structured_operators.hpp` supplies second-order Poisson and coupled
isotropic elasticity in the unit cube. Elasticity uses compact diagonal
second derivatives and centered mixed derivatives. Positive shear modulus
and nonnegative lambda plus shear modulus give an SPD auxiliary operator
on the periodic mean-zero subspace. This elastic operator is an
approximation when preconditioning a heterogeneous finite-strain Jacobian.
The true Jacobian remains separate and should use nonsymmetric Krylov.

`poisson_action.hpp` provides an owned-vector periodic Poisson action using
OpenPFC padded fields, MPI halo exchange and the existing finite-difference
kernel. `multigrid_mpi.hpp` provides an explicitly replicated hierarchy
adapter with collective ownership checks. The fine action is distributed;
the adapter gathers residuals and replicates hierarchy storage and work.
This supports correctness checks on MPI and is **not** a scalable
multilevel implementation. A distributed coarse hierarchy remains future
work; account for replicated memory and communication when using this
adapter.
