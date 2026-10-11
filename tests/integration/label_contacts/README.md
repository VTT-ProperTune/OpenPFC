<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Fresh known-UID label contacts

`label_contacts` scans a slot-major padded image whose nonzero labels already
represent complete persistent UID support. It neither thresholds amplitudes nor
propagates identity. The caller must pack the accepted state and complete a
**Full** halo exchange covering the chosen radius before publishing its generation
tag. A timestep's midpoint halo is insufficient. Tags reject stale metadata;
they cannot prove that a caller actually refreshed the image.

The probe compares distinct labels within the same slot and returns compact
owned support counts and a proximity flag. A flag permits on-demand all-slot
exact adjacency extraction. Cross-slot overlap at the same cell creates an edge
but does not itself trigger a same-slot collision. Six connectivity uses a
Manhattan ball; TwentySix uses a Chebyshev cube. Physical nonperiodic exterior
cells are excluded; periodic and decomposition corners come from Full halos.

The collective trigger votes producer failures before reducing flags and checks
radius/generation/scanned field-count agreement. It is not a global UID-admission or retirement
operation. `prepare_contacts` requires full edges, validates the canonical
strictly increasing registry and partition coverage, gathers only owned counts
and edges, then reuses the existing global resolver. `stage` and `transfer_signed`
retain the existing all-rank transaction and support negative and UID-owned zero
amplitudes. Touching same-slot supports already within radius one fail with
`UnsafeCadence`; the producer must detect early enough for its evolution stencil.
The caller supplies the required safety radius; this API does not infer it from
physics or RK stage count.

Both uint64 labels and exact-integer double labels are supported. Double labels
must lie in [0, 2^53]. Unknown UID, incorrect slot, noncanonical registry,
insufficient/Faces/stale halos and bounded edge overflow fail explicitly. No
conservative-bounds fallback is selected automatically.

`ContactWorkspace<Backend>` owns fixed-registry device buffers; reconstruct it
after any registry/slot change. Workspace-backed probes/extraction allocate no
device buffers; host compact result vectors still allocate and calls synchronize.
The convenience overload constructs scratch on every call. Probe scratch is
O(grains), extraction reserves an O(edge capacity) unique-contact table plus
host compact of occupied keys. The literal neighbor loop costs
O(owned cells * slots * radius cubed), and full extraction adds another slot
factor. These are correctness capabilities, not established performance claims.
The workspace is non-reentrant, privately owns its registry and buffers, and
rejects moved-from scratch before any launch.

The executable fixtures use an independent global coordinate-pair oracle and
real host/device Full halo exchange on 1/2/4/8 ranks. Cases distinguish same-slot
face/diagonal/seam flags from cross-slot same-cell edges, exercise disconnected
support, physical boundary poison, capacity failures, stale/incomplete images,
signed remapping and all-rank rejection. Native tests require actual HIP
hardware; compilation alone does not verify execution. Small field downloads
in native unit tests inspect transactional outputs; producers never download the
padded label volume.

Run the canonical `scripts/build.sh --cpu --no-heffte --mpi-tests` or HIP variant.
CTest registrations are `LabelContacts`, `LabelContacts_{1,2,4,8}ranks`, and
`HIP_LabelContacts_{1,2,4,8}ranks` when HIP and MPI suites are enabled. These tests
establish software correctness, not physical model validity or device scaling.
