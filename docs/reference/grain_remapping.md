<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Transactional grain remapping

`pfc::grain::remapping::remap` composes identity propagation, complete
contact construction, a deterministic host coloring decision, and strict
field transfer. The inputs are const. Success returns owning values,
labels, and a current `Snapshot` to publish together; every expected
failure returns empty owning state and leaves the original inputs intact.

Include `<openpfc/runtime/cpu/grain_remapping.hpp>` for the small-grid CPU
reference path. CUDA/HIP consumers include
`<openpfc/runtime/gpu/grain_remapping.hpp>` in a translation unit compiled
by the device compiler, and pass `DataBuffer<Backend, double>` values and
`DataBuffer<Backend, Id>` seeds. Fields and labels stay device-resident;
compact counts, graph edges, status, and registry metadata reach the host.
The implementation uses the same host decision for both paths.

## Input and publication contract

The 2D layout is `slot*(nx*ny)+x+nx*y`, with no halos or decomposition.
The [3D overloads](grain_3d.md) use `slot*(nx*ny*nz)+x+nx*(y+ny*z)`. Values
must be finite and nonnegative with exact zero background; occupancy is
exactly `q>0`. Every surviving nonzero seed on occupied support must belong
to an active sorted registry grain in its recorded source slot. Old seeds
on zero background are cleared. New positive cells may be unlabeled when
connected to an admitted seed; an unseeded positive component fails. No
threshold drops a diffuse tail, and no identity is allocated implicitly.

Each active grain must have support. If a grain disappears, the caller
explicitly retires its registry record, retaining its UID tombstone and
releasing its slot, or the operation returns `MissingSupport`. An inactive
record with positive support cannot publish as a live grain.

The result's snapshot contains the complete freshly constructed topology,
even when no coloring changes. A postcondition validates its exact active
membership and slots, and the decision validates proper coloring before
transfer. `Options::epoch` is a caller-owned logical observation number;
the operation neither infers numerical time nor enforces monotonicity
against a prior checkpoint. Publish values, labels, registry, and epoch as
one logical state. A reader must not observe individually replaced members.

`check_now=false` returns `Deferred` with no staged state. It does not
certify skipped detection or supply an old graph as current topology.
Fatal allocation/runtime exceptions propagate; const originals remain
unwritten by this operation. This contract is not device-fault recovery.

## Cadence and approaching grains

Reassign slots before an evolution step can mix distinct same-slot grain
amplitudes. Label propagation after mixing cannot reconstruct those
amplitudes. Choose a conservative contact buffer for the caller's stencil,
time step, and interval between checks. The default radius-three buffer is
appropriate for the example's one-cell-per-step nearest-neighbor support
growth; it is not a universal guarantee for arbitrary PDEs or time steps.
Four connectivity uses Manhattan contacts; Eight uses Chebyshev contacts.
The propagation stencil has radius one, independent of the contact radius.

This first integration conservatively returns `UnsafeCadence` whenever
different UID labels already touch within the propagation stencil in one
slot. It also refuses fully labeled touching supports: that geometry
alone cannot certify the history of their amplitudes. This is a narrower
domain than the standalone transfer API, which can move fully owned
adjacent data. The result does not assert that mixing has already occurred;
the caller can reject or roll back a proposed evolution step. Distinct
grains in different slots may have overlapping supports.

## Decisions, budgets, and explicit failures

Incremental repair is the default. `GlobalSaturation` and
`GlobalLargestFirst` use practical greedy policies, weighted slot-name
alignment, and by default preserve proper disconnected components.
Weights come from actual integer positive sample counts, converted to
double only for alignment. `componentwise=false` selects whole-graph
global recoloring. All policies skip the decision when the old assignment
is proper; full topology construction and owning publication still run.

`limits.attempts` bounds one coloring invocation, shared across conflicting
components. `limits.depth` bounds incremental recursion. A greedy or
bounded local failure is `SearchLimit`, not proof of uncolorability.
`CompleteOracle` runs bounded complete search and may return `Infeasible`
only after completing that proof. It is an oracle mode, not a scalable
production baseline. `max_sweeps` separately bounds identity propagation;
`IterationLimit` differs from an `Unseeded` support component.

`contact_capacity` is the output edge-list capacity, not a per-grain degree
cap. `CapacityOverflow` reports the required edge count and never publishes
a truncated graph. Registry/layout errors, unknown positive seeds,
missing support, and transfer rejection are explicit. Mixed malformed
positive data use the same preflight precedence on host and device:
`UnknownIdentity` outranks `InvalidInput`. `transfer_status` carries a
registry/support/transaction diagnostic where that layer was involved;
the primary remapping status always determines success.

## Diagnostics

`Statistics` reports synchronized host wall-clock stages for detection
(including support preflight and propagation), complete adjacency,
decision/alignment, transfer, and explicit synchronization. `total_seconds`
also includes destruction of the private staging scope. Internal
synchronization is included in its enclosing stage, so the synchronization
counter is not an additional term to sum into an end-to-end result. These
diagnostics describe an invocation; they are not a performance claim.

`moved_samples`, `moved_value_bytes`, and `moved_label_bytes` count selected
positive samples whose grain actually changes slot. They do not count
the whole-array reads, validation, staging, labels, synchronization, or
compiler-level memory transactions. An empty move plan still uses the
strict transfer path, copying owning arrays for publication:
`staged_storage_bytes` and `published_storage_bytes` report those dense
value/label arrays. Zero selected payload therefore does not mean zero
traffic. `wrapper_storage_bytes` reports directly held auxiliary dense
buffers, excluding STL metadata/queues and nested tracking/transfer
allocations; it is a lower bound, not an allocator high-water measurement.
Graph device-to-host bytes count the compact edge copy alone.

The CPU contact implementation deliberately uses the independent quadratic
coordinate-pair oracle for small grids. Device adjacency stores unique contacts
in an O(edge capacity) hash table and compact-sorts them on the host. This API
does not claim scalable host detection, optimal recoloring, or GPU-resident
coloring.

## Small consumer and validation

[The example](../../examples/grain_remapping.cpp) uses library primitives
without configuration or an additional application. It stages a short
two-dimensional explicit Allen–Cahn-style reaction/diffusion step, remaps
before evolution using a conservative buffer, then checks/stages the new
observation before publishing the proposed step. Both an interior approach
and a periodic-seam approach reassign slots before support coalescence.
The example is a teaching/correctness consumer, not a validated grain-growth
model or timing driver.

The integration suite advances separate one-field-per-grain references
with the same update and checks every positive UID sample and integrated
amplitude at each accepted step, including tiny nonzero tails. Static
controls cover complete current graphs, deterministic no-ops, retirement,
wide UIDs, unequal support weights, preserved proper components, K5
infeasibility, search limits, unsafe cadence, and overflow. Device tests
compare each realization and failure diagnostic against the host path and
include a real producer on a nonblocking caller stream. They require an
actual device; no host-only skip counts as device parity.

A downstream CPU consumer can use the installed public target:

```cmake
find_package(OpenPFC CONFIG REQUIRED)
add_executable(my_grains grain_remapping.cpp)
target_link_libraries(my_grains PRIVATE OpenPFC::openpfc)
```

CUDA/HIP consumers additionally link the exported
`OpenPFC::openpfc_gpu_kernels` / `OpenPFC::openpfc_hip_kernels` target and
compile their translation unit with that backend's C++20 device compiler.
Build/test through `scripts/build.sh` with the backend enabled; the
`GrainRemappingExample` and backend `CUDA_GrainRemapping` /
`HIP_GrainRemapping` entries execute the public entry points.

See [identity/topology](../concepts/grain_topology.md),
[propagation/contacts](../concepts/grain_tracking.md),
[coloring](../development/grain_coloring.md), and
[strict transfer](grain_transfer.md) for constituent contracts.

Optional [invocation diagnostics](grain_diagnostics.md) expose scoped allocation,
copy and declared source-operation accounting without changing publication.
