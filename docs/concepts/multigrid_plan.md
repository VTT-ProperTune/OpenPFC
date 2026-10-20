<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# MultigridPlan: rank-agglomeration planning without MPI execution

\`pfc::solvers::MultigridPlan\` (header
\`openpfc/solvers/multigrid_plan.hpp\`) is the **planning-only** first
step towards rank-agglomerated geometric multigrid under
[OpenPFC issue #420](https://github.com/ahojukka5/OpenPFC/issues/420).

The plan owns **metadata**, not CUDA/HIP buffers, communicators, MPI
requests, or a new domain partitioner. Its \`decomposition\` members and
all boxes use OpenPFC's existing \`decomposition::create\` and
\`split_box\`. It can be computed and numerically checked in ordinary
CPU CI even when an 8-, 64- or 6-rank runtime MPI world is unavailable.

## Supported initial policy

The first version deliberately admits only homogeneous Dirichlet
**free-node** Q1 grids where the two short axes have \`2^k-1\` nodes
and the third has either the same extent or \`2^(k+1)-1\` nodes.
Every geometric coarsening maps \`n_f = 2*n_c+1\`, doubles the physical
spacing and shifts the physical free-node origin by one old spacing.
The terminal short extent is **7**, so the final shape is 7³ or
7×7×15. All other boundary-condition or element families are
explicitly out of scope.

A process-grid dimension can remain unchanged or contract by a factor
of **2 or 3** at one adjacent level. We never assume that the number
of MPI processes must be a power of two; neither does OpenPFC
\`Decomposition\`. The default \`minimum_owned_extent\` is **16** in
each axis on levels containing multiple active ranks. The sole active
rank may descend to the 7-node terminal extent. At each step the plan
searches the at-most 27 legal process-grid candidates, retains the
largest admissible active rank count and breaks ties by an approximate
subdomain surface-area score. When no legal candidate exists, the plan
throws. It does **not** silently use a new MPI process grid with a
prime-factor reduction.

For example, with 127³ free Q1 nodes and a caller-supplied 4×4×4
process grid:

| Level | Free-node grid | MPI process grid | Active ranks |
| --- | --- | --- | ---: |
| 0 | 127³ | 4×4×4 | 64 |
| 1 | 63³ | 2×2×2 | 8 |
| 2 | 31³ | 1×1×1 | 1 |
| 3 | 15³ | 1×1×1 | 1 |
| 4 | 7³ | 1×1×1 | 1 |

With a 6×1×1 process grid on 255³ nodes, the active rank count
evolves 6 → 6 → 3 → 1 → 1 → 1 as the free-node extent reaches
255 → 127 → 63 → 31 → 15 → 7. For a 2:1 long axis the same
policy operates on the actual per-axis extents.

A *rank-grid factor* and the number of *free nodes* are different.
The free-node extent 127 is not divisible by four: OpenPFC's
\`split_box\` assigns uneven disjoint boxes correctly. The plan does
not require equal local box sizes.

## Rank IDs, groups and exact Q1 routes

For every level the plan records the original fine-communicator
\`world_rank\`, its compact \`level_rank\`, the active rank list in
OpenPFC's **x-fastest** order, the OpenPFC-owned \`Box3i\` and field
budget. The active communicator has not been created yet.

A transition contains a set of \`Group\` records. When a process-grid
axis contracts 4→2, leaders inherit world coordinates 0 and 2;
for 6→3 they inherit 0, 2 and 4; and for 3→1 the leader is the
first member, ultimately world rank 0. The groups are intended for
a later \`MPI_Comm_split\`, but the plan calls no MPI function.

**Grouping alone is insufficient to transfer Q1 values.**
A fine owned node at zero-based index \`i\` can contribute to coarse
node \`j\` only when \`2*j <= i <= 2*j+2\`. For each fine rank
we intersect its exact coarse support with all adjacent coarse
owners from OpenPFC's decomposition. The resulting \`Route\` stores
the fine and coarse *world* ranks, the owned coarse block,
the corresponding fine contributor bounds and the message-element
count. A \`crosses_group\` flag exposes the cases where one-cell
interpolation overlaps a neighboring rank group.

- \`R=Pᵀ\`: each fine owner computes **partial contributions from its
  own fine nodes** for every route and sends them to the unique coarse
  owner, which adds incoming contributions.
- \`P\`: each coarse owner sends the block's coefficients back along
  the reverse routes; the fine owner interpolates its own fine nodes.
- A coarse node may receive contributions from *multiple fine
  owners*. It still has exactly one coarse owner, and each fine/coarse
  owner pair has at most one route.

The plan does **not** form any MPI datatype, device index gather, MPI
communication, or actual transfer buffer. In particular it does not
pretend that a simple group-wise reduction automatically covers
cross-group interpolation support.

## Checked capacity, memory and limitations

The plan calculates checked **64-bit** global and rank-local node
counts, halo-padded node counts, per-route element counts and bytes.
Every planned point-to-point route is limited to the current MPI
\`int\` element-count bound; a too-large route throws rather than
silently overflowing. Each rank's
\`estimated_device_bytes_by_world_rank()\` totals a **configurable
number of halo-padded FP64 fields** over the levels on which it is
active (default 12 fields, width-2 halo). These are planning
estimates, *not actual device reservations or peak measurements*:
MPI send/receive staging, kernel temporary buffers, coarse factor
and runtime allocator overhead must be accounted for separately by
the future \`MultigridWorkspace\`.

The analytical flagship 4095×4095×8191 free-node case exceeds 32-bit
global indexing; this plan deliberately counts it in 64-bit without
allocating the grid. A checked plan does not establish that current
CUDA/HIP launchers, local device offsets, MPI counts or accelerator
memory can execute such a problem. Actual memory admission and
64-bit runtime indexing remain mandatory work before large runs.

## CPU/MPI owned transfer layer

`openpfc/solvers/multigrid_mpi_transfers.hpp` adds
`pfc::solvers::MultigridMPITransfers`, a separate runtime consumer of the
immutable plan. Every world rank constructs it collectively once; the class
uses `MPI_Comm_split` to create a compact communicator for each level's exact
active-rank list. Rank IDs are the plan's world identities, not assumptions
about contiguous rank groups.

`restrict_pt(transition, fine_owned, coarse_owned)` evaluates **additive**
`P^T` on the uniquely owned fine-node values. Planned owner-to-owner messages
accumulate at the unique coarse owner. `prolong(transition, coarse_owned,
fine_owned)` reverses those routes and interpolates to uniquely owned
fine nodes. Both use the same Q1 support and exact coarse-owner blocks,
including contributions crossing process-group boundaries. No global fine
field is assembled. Inactive coarse ranks return empty outputs and do not
participate in inactive-level collectives.

The initial transport uses **host FP64 owned-only contiguous arrays** and
allocates its route buffers per call. It does not change the production
`NodalDirichletDeviceVcycle`, allocate a GPU workspace, or claim MPI
performance. Its independent Catch2 tests compare Q1 interpolation and
restriction against directly evaluated analytical-array oracles, verify
global `<Pc,f> = <c,P^T f>`, repeated-call determinism, cubic/2:1
rectangular grids, and 1/2/3/4/6-rank plans including 6 -> 3 -> 1.

Enable `OpenPFC_RUN_MPI_SUITES=ON` to register the
`multigrid_plan_transfer_*ranks` CTests. Existing rank caps are respected;
the six-rank case additionally requires at least six configured logical
cores or an explicit high-rank registration override. A passing compile
does not qualify CUDA/HIP execution or a complete multigrid solve.

## Integration boundary and acceptance

This PR **does not change the existing**
\`NodalDirichletDeviceVcycle\`, \`NodalDirichletMultigrid\`, or their
field layouts. Future separate changes will:

1. Use the plan to construct level-specific active MPI communicators
   and cross-group P/Pᵀ routing, first verifying exact independent
   CPU/MPI ownership, conservation and adjointness.
2. Reserve device and halo/message workspaces once, from an inspected
   memory budget, then execute the existing OpenPFC kernels and Krylov
   algebra without per-cycle setup.
3. Validate real CUDA/HIP multi-rank physical correctness before any
   deferred HPC scaling campaign.

The CPU tests under \`test_multigrid_plan.cpp\` check 64→8→1,
6→3→1, cubic and 2:1 geometry, exact route ownership,
cross-group transfers, minimum widths, invalid prime rank factors
and checked 64-bit global sizes without a GPU or MPI world.

This is infrastructure progress on #420, **not a closure of #420
or #591** and not authorization to run DiscretizationLab's
[deferred HPC campaign #169](https://github.com/ahojukka5/DiscretizationLab/issues/169).
