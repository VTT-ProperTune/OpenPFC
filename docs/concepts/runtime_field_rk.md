<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Runtime coupled field stepping

`RuntimeFieldRKStepper` in
`openpfc/kernel/simulation/steppers/runtime_field_rk.hpp` advances a
runtime-sized group of `Field<double, HostSpace>` objects with an existing
explicit Butcher tableau. It reuses the vector RK engine. Each stage publishes
owned values into the borrowed fields and restores one accepted padded
snapshot afterwards. The tuple-based field factory keeps its own stage
transaction. Existing factories remain available. No physical equation or
boundary law is supplied by this adapter.

`create_runtime(fields, dt, tableau, rhs, comm)` borrows a span of field
pointers. The callback `rhs(actual_stage_time, rates)` observes all bound
fields containing the same stage. Each rate span contains owned cells in
x-fastest order, `i + nx * (j + ny * k)`. Field count is runtime-sized, with
no special case for eight or sixteen components. All fields on a rank must
have identical geometry, halo width and storage layout. Across ranks, field
count, global geometry and method must agree; local boxes may differ.

The callback prepares physical boundaries and exchanges stage halos before
computing rates. The accepted padded buffers are restored after every
callback, including exceptions. Padding restored from accepted state is not
fresh stage padding: preparation is required at every stage. The callback
must complete **all its own communication and collectives before throwing**.
The adapter cannot recover divergence or an unmatched MPI operation inside
application code.

`attempt(t)` returns a candidate without changing any accepted cell.
`candidate_field(attempt, component)` gives a read-only compact owned-cell
view for physical admission. `commit(attempt)` validates the original
attempt descriptor, full accepted buffers and collective agreement before
writing owned cells. Applications remain responsible for physical
constraints and mass or energy admission. A failed attempt, rejected commit,
new attempt or completed commit invalidates the preceding ticket. Candidate
views are valid only until the next attempt or stepper destruction.

Construction, `attempt`, `commit` and `step` are collective on the explicit
communicator. Candidate access and queries are local. The default communicator
is `MPI_COMM_SELF`. Rank-local input errors, nonfinite rates and callback errors
are reduced before proceeding to another stage or publishing a candidate.
All participating ranks must call operations in the same order. The stepper
is not reentrant. Borrowed fields and communicators must outlive it; field
objects, geometry and storage addresses must stay fixed. A callback that
resizes a field has its accepted contents restored without allocation, but
invalidates the old binding; reconstruct the stepper before retrying.

`workspace_bytes()` reports scalar scratch capacity, including compact RK
arrays and one accepted padded snapshot per field. It excludes caller fields,
callback captures, tableau coefficients and pointer metadata. It is not
process memory usage or a performance claim.

The qualified interface is explicit, homogeneous, CPU double precision.
It does not provide device stepping, adaptivity or spectral evolution.
Independent tests use a coupled fixed-FD2 system with an analytic solution,
stage-time polynomial integration and unilateral MPI rejection. The
installed consumer exercises seventeen fields, preserved padding and retry
on one, two and four ranks.
