<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Nonperiodic nodal Q1 multigrid: CPU qualification

`pfc::solvers::NodalDirichletMultigrid` is the **bounded host-only**
nonperiodic qualification of the existing nodal Q1 geometric multigrid
research. It is a reusable numerical primitive owned by OpenPFC rather than
by a downstream experiment. The production periodic solver
`NodalMultigrid<HostSpace/CUDASpace/HIPSpace>` is not changed.

## Mathematical and ownership contract

Solve the free-node Q1 Galerkin operator

```text
A = a K_Q1 + sigma M_Q1,  a > 0 constant, sigma >= 0.
```

on a Cartesian brick with homogeneous Dirichlet conditions on all faces.
The **Domain contains free nodes**, not Q1 elements or fixed boundary nodes.
For a physical mesh with `2^k` elements per axis, the correct input Domain
has `2^k-1` free nodes per axis and the original physical element spacing.

The caller supplies a normal OpenPFC `decomposition::Decomposition`; it may
be a z-slab, x-slab, or an x/y brick process grid. The class never partitions
a grid itself and never owns MPI device transport. It allocates one
`data::Field<double, HostSpace>` per required rank-owned workspace and binds
OpenPFC's 26-direction full halo exchange to a reusable padded field.

Nested grids satisfy `n_f = 2 n_c + 1` in free-node coordinates. Transfers
are geometric Q1 trilinear prolongation `P` and the unscaled adjoint
restriction `P^T`. Every rank gets its local box from OpenPFC on *each*
level. Before allocating communication buffers, the constructor verifies
for every rank and axis that the two adjacent partitions need at most two
halo cells. Unsupported partitions fail explicitly; they do not silently
fallback to a custom slab splitter or replicate a fine level.

The fine and intermediate operators are matrix-free 27-point Q1 stencils.
Coefficients and consistent mass weights reuse the factorized constant Q1
stiffness from `nodal_multigrid_kernels.hpp`. Pre/post Jacobi smoothing is
symmetric with conservative damping. The coarsest block alone is replicated,
and its maximum size is bounded by `Controls::max_coarse_unknowns` before
dense Cholesky allocation. OpenPFC's `FieldOperations` + `solvers::cg`
provide rank-owned Krylov vectors, global reductions and true-residual
admission.

## Validation and limits

The separate Catch2/MPI checks compare the matrix-free operator against
independent element-by-element two-point Gauss Q1 integration on both
noncubic physical domains and 2:1 elongated nodal shapes; verify V-cycle
symmetry and positivity; and recover an exact discrete solution via PCG.
They use genuine MPI 1/2/4 process layouts, including multiple decomposed
axes when 4 ranks are available. The default hosted CI currently caps MPI
world size at two ranks; a local or dedicated four-rank configuration must
pass before claiming four-rank acceptance.

This first implementation is **CPU only**, with constant coefficients,
homogeneous Dirichlet conditions, all axes nonperiodic, and a bounded
replicated coarse solve. No HIP/CUDA, mixed boundary type, arbitrary
nonuniform geometry, coefficient jumps, nonlinear mechanics, or large-scale
performance claim has been qualified. Do not extrapolate the host results
to LUMI. Once accepted, downstream DiscretizationLab should call the
installed OpenPFC solver and retain only independent FEM oracles and
experiments. The two nodal solver variants can then be consolidated without
breaking existing periodic GPU users.
