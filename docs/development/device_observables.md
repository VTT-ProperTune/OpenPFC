<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Fused device observables

Include `<openpfc/runtime/gpu/observable_reduce_gpu.hpp>` in a CUDA/HIP
translation unit. `pfc::sim::gpu::reduce_owned_multi(field, map, ops, comm)`
visits owned cells once, evaluates the supplied device map and combines
1–16 expressions using per-component sum, minimum or maximum. Padding is
excluded. The map receives `(value, i, j, k)` in local owned coordinates and
returns `ReductionValues<N>`. `ReductionOps<N>` holds the matching operations.
Captured device pointers can supply additional synchronized fields.

The call uploads stale primary storage if necessary, reads back at most
256 partial vectors, and performs one success agreement plus at most three
scalar value collectives. All ranks must pass identical operation lists and
participate in the same order. An empty rank contributes the neutral values
0, positive infinity and negative infinity. Floating-point sums follow GPU
and MPI reduction order, so bitwise equality with serial sums is not promised.

NaNs must be handled by the map: count invalid values in a sum component and
return each other operation's neutral value for that sample. No geometry,
field-sized readback, scientific acceptance rule or invalid-data policy is
hidden in the reduction. A local device failure is collectively reported
before any rank enters the value collectives.

`pfc::mpi::all_ranks_succeeded(local_ok, comm)` and
`require_all_ranks(local_ok, comm, message)` in
`<openpfc/kernel/mpi/collective_check.hpp>` are also available independently
of GPU support. Every member participates even when its local condition is
false. The latter throws on every rank after collective rejection; it does
not catch earlier exceptions or roll back model state.

Package checks cover independent scalar references, poisoned padding, NaN
counting, empty ranks and collective rejection. GrainFront integration and
its statistics/remapping parity tests remain downstream acceptance work.
