<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Runtime-sized device stage field sets

`StagePreparationService<>` accepts a runtime vector of field names. Bind
an existing blocking exchanger for each name, then register its field and
`ReflectionWalls<Value>` through `set_reflection`. One service can prepare
double phase/solute fields and integer label fields. Field names are owned
by the service; fields, storage and exchangers remain caller-owned and must
outlive their bindings. Reallocation requires rebuilding the bindings.

Include the selected GPU field/transport and `reflection_ghosts_gpu.hpp`
headers when binding device fields. The stage protocol stays in the kernel
layer; it dispatches typed callbacks without including vendor runtime headers.
The default template parameter remains source compatible with host users.

For existing double padded fields, bind `HaloExchange<Space, double>` directly.
For typed sparse transport, `make_padded_sparse_halos(field, decomposition,
rank, tag_base)` builds six-face index lists in the actual padded storage.
Pass those records to `SparseExchange<Space, Value>` and bind that exchanger.
The GPU implementation supports `double` and `int`. It gathers and scatters
on the device; non-aware MPI stages only face buffers through host memory.
GPU-aware MPI follows the existing runtime capability gate. The field itself
is never copied to the host by stage preparation.

Use distinct tag ranges for concurrently active exchangers. The helper adds
canonical direction tags, includes periodic self-neighbors and omits physical
dead ends. Six-face connectivity does not promise edge/corner exchange.
Methods requiring diagonal stencils must select an appropriate full transport
and may still bind it through the same stage protocol.

Call `prepare(requirements, requested_names)` before each evaluation. All
names are checked before mutation. The selected boundary-hook/halo ordering
applies across the complete requested list; registered reflections run after
both phases. On rejection/retry, prepare from the accepted owned state again.
No rollback or post-evaluation boundary enforcement is provided.

For impermeable cell-face walls, select even reflection with cell-face
centering. Only global nonperiodic faces are reflected. Integer labels copy
exactly under even reflection; odd integer reflection requires representable
arithmetic, as on the host. Scalar reflection alone does not enforce zero
constitutive flux: choose the face-current wall policy separately.

Package tests cover runtime counts beyond four, actual integer device labels,
face transport, periodic self-neighbors, physical reflection and repeated
stages after owned values change. GrainFront's stage replacement and its
1/2/4-rank decomposition/orientation parity remain downstream acceptance work.
