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
That adapter remains a correctness check, not a scalable hierarchy.

`coarse_correction.hpp` supplies a bounded shared coarse correction.
Owned fine residuals restrict with the rows of P^T, one small collective
sums the coarse residual, and the existing coarse solve prolongs the
correction. The coarse unknown may be replicated while it stays under the
explicit size bound. Periodic constants use one communicator-wide mean, so
unequal ownership does not remove a different gauge on each rank. This is
a two-level correction, not a distributed hierarchy with a coarse-rank
policy.

# Matrix-free nodal Q1 multigrid

`openpfc/solvers/nodal_multigrid.hpp` supplies `NodalMultigrid<Space>`, a
distributed V-cycle preconditioner for the periodic trilinear (Q1) finite
element operator

$$
A = \sum_e a_e K_e + \sigma M ,
$$

with one coefficient $a_e > 0$ per element (indexed by its lowest node),
the exact element stiffness $K_e$ and the exact consistent mass $M$. On a
uniform grid with spacing $(h_x, h_y, h_z)$ and $V = h_x h_y h_z$ the element
matrices are Kronecker products of 1-D ones,

$$
K_e = \sum_d \frac{V}{h_d^2}\, K_1 \otimes M_1 \otimes M_1,\qquad
M_e = V\, M_1 \otimes M_1 \otimes M_1,\qquad
K_1 = \begin{pmatrix}1&-1\\-1&1\end{pmatrix},\;
M_1 = \begin{pmatrix}1/3&1/6\\1/6&1/3\end{pmatrix},
$$

so a row of $A$ is a 27-point stencil whose weights fold the 8 adjacent
element coefficients. The fine operator is never assembled; `apply_operator`
is the Krylov action and costs one read of $u$ and of the coefficients.

```cpp
using MG = pfc::solvers::NodalMultigrid<pfc::CUDASpace>;
MG::Controls c;
c.sigma = 1.0;
MG mg(decomp, rank, MPI_COMM_WORLD, c);
mg.update(element_coefficient);            // owned elements, any storage halo
// Krylov action and preconditioner on halo-1 Fields of the owned node box:
//   mg.apply_operator(u, y);   mg.apply(r, z);   // or mg(r, z, k)
```

**Transfers.** Coarse node $I$ sits on fine node $2I$. Prolongation $P$ is
trilinear nodal interpolation (an odd fine index averages its two coarse
neighbours) and restriction is $R = P^\top$. Both read one halo layer and
write owned values only; nothing is accumulated across ranks.

**Coarse operators.** From level `galerkin_from_level` on, a level stores the
Galerkin product $P^\top A P$ as 27 weights per node. Earlier coarse levels are
rediscretized with each coarse element coefficient the mean of its 8
children. The Q1 spaces are nested, so for a constant coefficient both are the
same operator: $P^\top K_h P = K_{2h}$ and $P^\top M_h P = M_{2h}$. With a
varying or discontinuous coefficient only the Galerkin product keeps the
variational property; it costs 27 values per coarse node, so the first
Galerkin level stores about 3.4 fine fields.

**Smoothers.** Chebyshev iteration on $D^{-1}A$ over
$[\lambda/\text{range}, \lambda]$, with $\lambda$ a safety factor times a
power-iteration estimate per level, or symmetric 8-colour Gauss-Seidel (nodes
of equal global index parity never share a row). The power iteration starts
from a hash of the global node index, so the hierarchy and every cycle are
the same on any decomposition up to the order of floating-point reductions.
Pre- and post-smoothing are adjoint, so the V-cycle is symmetric and can
precondition CG.

**Distribution and agglomeration.** Distributed levels keep the fine process
grid while every local extent and offset halves evenly. When a coarse level
has at most `agglomerate_max` nodes, or cannot be halved on the process grid,
each rank restricts into its own region and one allgather gives every rank the
whole level; the remaining levels are computed redundantly. Only that bounded
coarse part is replicated, and `replicated_bytes()` reports it. The coarsest
level is solved by a dense Cholesky factorization when it has at most
`dense_max` nodes (`coarse_exact()`), otherwise by `coarse_sweeps` smoothings.

**Nullspace.** With $\sigma = 0$ the constant is in the kernel of every level.
The dense coarse matrix gets a rank-one constant term, its right-hand side is
projected to zero mean, and `apply` returns a zero-mean correction. The cycle
is then symmetric on mean-free vectors, which is the space PCG works in for a
consistent right-hand side.

`stats()` reports the setup cost (coarse operators, eigenvalue estimates,
factorization) and, with `Controls::profile`, synchronized per-phase times of
smoothing, residuals, transfers, halo exchange, agglomeration and the coarse
solve. `level_info(l)` gives each level's extents, storage and bound.
