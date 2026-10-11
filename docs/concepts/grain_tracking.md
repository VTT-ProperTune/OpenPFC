<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Resident grain ownership and contacts

`<openpfc/runtime/gpu/grain_tracking.hpp>` supplies the synchronous
`pfc::grain::tracking` producer behind the
[grain identity contract](grain_topology.md). Select the supported execution
backend with its existing `CUDATag` or `HIPTag`. Device spans denote actual device
storage; the caller completes earlier writes before entering the API. The call
returns after its default-stream kernels and copies complete. Concurrent calls
must use independent inputs and outputs, or the caller serializes them.

The storage layout is slot-major: `slot*(nx*ny) + x + nx*y`. Axis lengths must fit
signed 32-bit coordinates; storage and allocation byte counts are checked for
overflow. Grid periodicity and Four/Eight connectivity are explicit. Each
persistent identity belongs to exactly one order-parameter slot, checked by the
caller through its observation/assignment contract. Neither routine assigns new
identities or decides whether a split/merge creates a new grain.

## Ownership propagation

The application supplies a byte occupancy mask and prior persistent identities.
Occupancy is zero outside admitted grain support, nonzero inside it. The library
does not choose a physical order-parameter threshold. Surviving prior labels act
as fixed seeds; inactive cells clear their label, including disappearing grains.
Every newly occupied connected region needs a surviving seed, or the operation
reports `Unseeded`. Fresh components must be seeded by the caller before use.
Disconnected grains sharing a slot remain distinguishable through their seeds.
If distinct seeded grains in that slot touch, their existing labels remain
separate and the contact producer reports the resulting conflict.

Propagation uses ping-pong sweeps, reading only the previous sweep. The first
arrival at an unlabeled active cell wins; equal-distance ties choose the smallest
persistent identity. This is deterministic multi-source breadth-first ownership,
not an in-place painting race. Once assigned in this observation, a new label is
fixed. The caller supplies a sweep budget; a growing front that exhausts it
returns `IterationLimit`. The result includes actual sweep count. At most one
label-free convergence sweep is needed beyond the maximum propagation distance;
`nx*ny` is a sufficient per-plane budget when all admitted components have seeds.

The complete label grids remain on the device. Each sweep reads back a small
convergence flag. Successful propagation publishes the staged result by device
copy; expected failures leave the destination unchanged. Input/output aliasing
is supported because work is staged before publication. This guarantee concerns
reported status failures; a fatal runtime/copy failure is an exception and must
be handled as an execution failure by the caller.

## Contact extraction

`adjacency` examines same-cell overlaps and spatial contacts across *all* slot
planes. It takes an explicit nonnegative `contact_radius` (default one). Four
connectivity uses Manhattan distance; Eight uses Chebyshev distance. Periodic
wrapped distances are admitted at that radius. For example, radius three with
Eight connectivity is an explicit conservative policy for buffered contacts;
it is independent of the support mask and ownership propagation stencil.
A radius must be smaller than the largest signed 32-bit integer.

The host supplies a canonical identity registry. Kernels validate encountered
identities against that registry, mark active identities, and atomically insert
unique contacts into an open-addressed table of O(edge capacity) slots. Host
compact sorts the occupied keys into a canonical unique contact list. No fixed
per-grain neighbor cap silently discards an edge.

The contact destination is a bounded device span. If it is too small,
`CapacityOverflow` reports the exact required `edge_count`, leaving the
previous destination untouched. A missing identity reports `UnknownIdentity`
without publication. On success only the first `edge_count` entries are valid;
the untouched remainder is not part of the graph. `active` is sorted host
metadata and excludes disappeared identities. Only compact registry/active
metadata and scalar status/count cross the host boundary, while label grids and
contact lists remain on the device.

The contact table is `contact_hash_slots(edge_capacity)` 64-bit keys, at least
eight and four slots per reserved edge, rounded up to a power of two. That is
O(edge capacity), not O(registry size squared). A 10,000-identity registry with
a 16-edge output therefore does not allocate a 10,000 × 10,000 adjacency
matrix. Allocation failures throw rather than truncate. It is a local 2D
producer with no MPI, 3D, asynchronous
update cadence, or PDE timestep ownership. Applications call it after changes to
support/identity observations at their chosen cadence; call frequency changes
detector cost and must be stated by any consumer.

The standalone grain tracking test compares device results against independent
priority-queue shortest-path and coordinate-pair contact oracles, including
periodic thin grids, overlapping planes, conservative radius-three contacts,
shrinking support, shared-slot grains and unchanged failure publications.

The reusable internal host reference in
`<openpfc/runtime/cpu/detail/grain_tracking.hpp>` exposes
`tracking::reference::propagate(grid, slots, occupancy, seeds)` and
`tracking::reference::contact_graph(grid, slots, labels, radius)`. Its spans are
host storage. Propagation returns `{complete, labels}` with zero for unseeded
active cells; callers reject an incomplete result before publication. It uses a
priority queue rather than device sweeps. Contact extraction uses independent
coordinate-pair distances and is quadratic in the total slot-grid storage;
this implementation is a correctness reference for small grids, not a scalable
performance baseline. It has no device dependency or persistent state ownership.
