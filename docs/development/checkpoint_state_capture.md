<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Checkpoint state capture

OpenPFC exposes a **serialization-agnostic** capture/restore seam under
`include/openpfc/kernel/checkpoint/` so a future checkpoint manager can
orchestrate save/restore without inventing per-app dumps.

## Headers

| Header | Role |
|--------|------|
| [`openpfc/kernel/checkpoint/payloads.hpp`](../../include/openpfc/kernel/checkpoint/payloads.hpp) | `FieldPayload`, `ComponentPayload`, `PersistentState`, `DecompositionMeta`, dtypes |
| [`openpfc/kernel/checkpoint/state_capture.hpp`](../../include/openpfc/kernel/checkpoint/state_capture.hpp) | `capture_field` / `restore_field`, `capture_component` / `restore_component` |

Namespace: `pfc::checkpoint`.

## Field payloads

A `FieldPayload` carries:

- stable `field_id`
- `FieldDtype` (`Float64`, `Complex128`)
- owned-cell `extents` (nx, ny, nz)
- `CoordinateOrder::XFastest` (OpenPFC row-major, x fastest)
- format `version` (`kFieldPayloadFormatVersion`)
- optional `DecompositionMeta` (rank count/rank, global/local extents, local offset)
- contiguous `bytes` of owned cell values only

## Component payloads

A `ComponentPayload` holds irreducible integrator/controller cross-step state
(`component_id`, `version`, `bytes`). Explicit Euler / Heun typically use
`empty_component_payload("euler")` (empty bytes). **Do not** store driver-owned
`pfc::sim::Time` / step counters / config identity in component payloads.

## Validate-before-mutate restore

`restore_field` / `restore_component` check **all** of the following before any
destination write:

1. format version
2. field / component id
3. dtype (fields)
4. extents / shape (fields)
5. coordinate order (fields; must be `XFastest`)
6. optional decomposition equality when the caller supplies expected metadata
7. **exact** `payload.bytes.size() == expected_nbytes` (`BytesSizeMismatch`
   otherwise — truncated or oversized bytes with matching metadata still reject)
8. destination capacity (`BufferTooSmall` if too small)

On any failure the destination buffer is left unchanged. Multi-field adapters
(Wave2D `restore_uv`) validate every field fully before mutating any buffer.

## App adapters

| App | Header | API |
|-----|--------|-----|
| Heat3D (scalar) | `apps/heat3d/include/heat3d/state_capture.hpp` | `capture_u` / `restore_u` (`heat3d.u`); `PaddedBrick` overloads pack **owned** cells only |
| Wave2D (coupled) | `apps/wave2d/include/wave2d/state_capture.hpp` | `capture_uv` / `restore_uv` (`wave2d.u`, `wave2d.v`) |

Physics model headers (`heat_model.hpp`, `wave_model.hpp`) stay free of OpenPFC
includes; adapters live beside them.

## Exclusions

The following are **not** part of captured payloads:

- Stage / scratch buffers (`Workspace` stages, RHS temporaries)
- FFT plans and spectral operator caches
- Halo / ghost rings (recomputable via exchange)
- Driver-owned `Time`, increments, and run-config identity

This API does **not** define a checkpoint file format, atomic publish, or
manager orchestration — those belong to sibling checkpoint-manager leaves.

## Tests

- Kernel: `tests/unit/kernel/checkpoint/test_state_capture.cpp` (`[checkpoint][state_capture]`)
- Heat3D: `apps/heat3d/tests/test_heat3d.cpp` (`[heat3d][state_capture]`)
- Wave2D: `apps/wave2d/tests/test_wave2d.cpp` (`[wave2d][state_capture]`)

## See also

- [Integrator interface contract §6](integrator_interface_contract.md#6-checkpointrestart-semantics-requirements)
- [Class tour](../reference/class_tour.md) — `FieldPayload` / `ComponentPayload` rows
