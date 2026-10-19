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

The full geometric V-cycle remains **CPU only**. The constant-Q1
Dirichlet row action, symmetric Jacobi point update, and P/P^T are now
factored into `solvers/nodal_dirichlet_kernels.hpp` with
`OPENPFC_HD` point definitions. OpenPFC's existing CUDA/HIP nodal
kernel translation units expose `dirichlet_apply_cuda/hip` for a
**pre-exchanged halo-2 device array and a trivially-copyable constant
stencil**. This does not perform any halo exchange or solve; the
caller must supply an already consistent device buffer. OpenPFC's
normal GPU Field/HaloExchange runtime will own this in a follow-up
implementation. OpenPFC's same vendor translation units also expose
`dirichlet_jacobi_cuda/hip`, `dirichlet_restrict_cuda/hip` and
`dirichlet_prolong_cuda/hip` as **separate halo-2 point kernel launches**:
the caller provides old/new distinct Jacobi buffers, exchanged fine
residuals and exchanged coarse corrections. The P/P^T operators do
not invent partition rules or perform MPI transport. The full
GPU-resident hierarchy is **not yet implemented**.
Host Catch2 tests independently check Q1 rows,
physical-Dirichlet boundary sentinel masking, shifted rank-local
bricks, and P/P^T adjointness; GPU device-vs-host point-action parity
is a separate device-conditional test. CUDA/HIP compile-only CI
qualifies compilation, not actual accelerator execution.

### Rank-owned CUDA/HIP field and halo action (qualification gate)

`solvers/nodal_dirichlet_device_operator.hpp` supplies
`NodalDirichletDeviceOperator<CUDASpace/HIPSpace>`, the first
**device-resident matrix-free action adapter**. It is not a second
multigrid hierarchy and not a preconditioner. The adapter takes an
existing OpenPFC `Decomposition`, allocates its persistent
`FieldOperations`/`FieldHalo` scratch at **halo width 2**, performs
the owned-only GPU copy of a caller `Field`, exchanges the 26-direction
halo through OpenPFC's device `HaloExchange`, applies the shared Q1
Dirichlet GPU row kernel, and copies only the owned rows back to
the caller's field. External field storage halos of 0, 1 or 2 are
supported; the physical exterior is always treated as zero.

The device pipeline contains **no downloads of fine-grid fields to
host memory**. The later GPU V-cycle will reuse this device/halo
contract plus the Jacobi and P/Pᵀ point launchers. A GPU-conditional
test in `test_nodal_multigrid_device.cpp` compares the entire
device staging/communication/operator sequence against the host
Dirichlet FEM reference on 1/2/4-rank layouts and checks repeated
input and output-alias cases. It also calls OpenPFC's existing
`FieldOperations<MemorySpace>` and CG with **no multigrid preconditioner**
on a known discrete solution, admitting convergence only with the
true unpreconditioned residual. This is a correctness diagnostic,
not evidence of a fast GPU GMG solver.

### Rank-owned two-level GPU transfers

`NodalDirichletDeviceTransfers<CUDASpace/HIPSpace>` connects the existing
GPU point launchers for `R=Pᵀ` and `P` to *two caller-supplied OpenPFC
decompositions*. It does not create a partition, implement a second
multigrid solver, or store a replicated fine grid. The fine residual
and coarse correction are copied by owned-only device kernels into
persistent padded `FieldHalo` workspaces, exchanged by OpenPFC's
26-direction device halo mechanism, then restriction/prolongation
operate on the uniquely owned nodes. External fields may use different
storage halo widths; no fine-grid data is downloaded to the host.

For this Dirichlet free-node convention, adjacent levels satisfy
`n_f=2*n_c+1` and `h_c=2*h_f`. The shared
`nodal_dirichlet_nested_geometry.hpp` gate checks those relations and
all MPI ranks' adjacent-level two-cell transfer reach. The CPU
`NodalDirichletMultigrid` uses the **same** geometry validator.
Independent host MPI acceptance assembles `P` and `Pᵀ` by a test-only
scatter, verifies adjointness across ranks and checks 2×2 brick layouts;
CUDA/HIP device-conditional tests compare full FieldHalo transfer paths
to the host reference. Neither compile-only jobs nor host tests prove
actual accelerator numerical correctness.

**
### Bounded Q1 coarse direct solve

The coarse solve uses exactly one shared, **CPU-resident**
`nmg::dirichlet::DenseCoarseCholesky` implementation for all backends.
The current CPU/MPI V-cycle has been refactored to use it without
changing fine/intermediate rank ownership or the original test
problem. The factor is constructed from the same 27-point Q1 element
weights, only after checking the free-node count and dense-allocation
budget. It separately checks symmetry and positive Cholesky pivots.
The RHS and solution contain exactly the bounded coarse coefficients.

Future CUDA/HIP cycles may reduce a small coarse RHS, solve it with
this factor and upload only its coarse correction. This is **not**
permission to transfer or replicate the global fine/intermediate
fields. Host CI independently verifies diffusion and diffusion/reaction
coarse solves against Gauss-assembled Q1 FEM and rejects invalid
capacities or non-SPD matrices. No GPU recursion or performance claim
is implied.

The complete GPU V-cycle is not implemented by this helper.** The
remaining steps are coupled pre/post-smoothing, residuals, coarse
reduction/solve, and the OpenPFC Krylov preconditioner contract.

The GitHub-hosted CUDA/HIP jobs **compile and link** that test,
but have no GPU hardware to execute it. Actual GPU+MPI correctness
must be reported separately; this stage is not proof of a resident
multilevel solver or HPC performance.

The current solver has constant coefficients,
homogeneous Dirichlet conditions, all axes nonperiodic, and a bounded
replicated coarse solve. No HIP/CUDA, mixed boundary type, arbitrary
nonuniform geometry, coefficient jumps, nonlinear mechanics, or large-scale
performance claim has been qualified. Do not extrapolate the host results
to LUMI. Once accepted, downstream DiscretizationLab should call the
installed OpenPFC solver and retain only independent FEM oracles and
experiments. The two nodal solver variants can then be consolidated without
breaking existing periodic GPU users.
