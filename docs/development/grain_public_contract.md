<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Public grain planning and signed staging

Include installed `runtime/common/grain_remapping.hpp` for graph-only
`pfc::grain::remapping::decide`. It reuses the existing coloring methods;
it does not inspect fields, track identities, cut support, exchange halos
or publish accepted state. Contacts are canonical sorted/unique vertices
and edges; the sorted grain registry may also contain inactive tombstones
or active grains outside this graph. Weights match vertices and are finite
and nonnegative. Invalid structure/registry/weights return `InvalidInput`;
unknown or inactive graph identities return `UnknownIdentity`.

Include `kernel/grain/transfer.hpp` for public `Assignment`, `TransferPlan`,
`prepare_transfer` and host/device `find_assignment`. The assignment table
contains one active UID's `id`, `source` and `destination` slots. Preparation
validates registry and moves, handles cycles against the original state,
and returns an empty payload on failure. A missing lookup returns the
assignment count. Tables remain sorted; callers must not modify a validated
plan while using it. Upload the trivially copyable assignments for a device
kernel; the host plan's vectors are not device pointers.

`kernel/grain/signed_transfer.hpp` exposes host/device `stage_signed_cell`.
Each invocation reads one cell from slot-major source arrays and writes
that cell in separate staging arrays. UID labels define support; signed
values, including zero, do not define identity. Unlabelled cells equal the
finite declared background. Exact or partial staging overlap with values,
labels or the assignment table, overlap between staging buffers, invalid
cell/buffer arguments, byte-extent and address overflow are rejected before
any staging write. Adjacent disjoint ranges remain valid.
Callers must provide correctly sized, disjoint buffers and a validated
immutable assignment table. A failed cell can leave partial **scratch**;
accepted inputs are const and remain unchanged.

Collect statuses across all cells/ranks and verify that every active
assignment has support before publishing values, labels and `plan.grains`
together. The cell primitive cannot certify whole-grain support or buffer
capacity. `transfer_signed` remains the fully checked host transaction;
use it when no specialized field/device staging is needed. These functions
do not implement application-specific physical laws or conservation
ledgers.

`runtime/cpu/grain_tracking.hpp` is the public include for
`tracking::reference` propagation/contact oracles. This retains the
existing independent small-grid implementation and its limitations; it
is not a new scalable contact search.

The prior internal type spellings remain source-compatible aliases.
Moving assignment ownership into the public namespace changes C++ type
symbol names: rebuild the library and consumers together, and never mix
headers and binaries from different revisions. No algorithm, stored value
or UID numbering changes.

Neutral tests cover signed cycles, exact UID/value preservation, invalid
plans and alias rejection, canonical graph decisions/malformed input, and
reference propagation. A matching installed-package consumer and device
compilation should be qualified before downstream adoption.

The independent installed consumer uses no source-tree include paths:

```bash
cmake -S tests/integration/grain_public -B builds/grain-consumer \
  -DCMAKE_PREFIX_PATH='<matching-installed-prefix>;<heffte>' \
  -DCHECK_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build builds/grain-consumer
builds/grain-consumer/grain_consumer
# Run the device consumer inside the appropriate GPU allocation.
builds/grain-consumer/grain_device_consumer
```
