<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Atomic checkpoint publication

OpenPFC can publish an **accepted** solution state as a versioned filesystem
directory bundle. The entry point is
`pfc::checkpoint::publish_checkpoint_directory` in
[`include/openpfc/kernel/checkpoint/publish.hpp`](../../include/openpfc/kernel/checkpoint/publish.hpp),
with metadata types in
[`checkpoint_metadata.hpp`](../../include/openpfc/kernel/checkpoint/checkpoint_metadata.hpp).

This is the durable **publish** seam. Framework restart is
`pfc::sim::CheckpointService` (`kernel/simulation/checkpoint_service.hpp`):
collective MPI-IO of owned float64 bricks plus rank-0 `metadata.json`, then
the same stage→rename protocol so an interrupted write never becomes a
loadable bundle. It is distinct from scheduled headerless field dumps written
by `ResultsWriter` / frontend `BinaryWriter`
(see [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)).

In-memory `state_capture` restore is a payload-buffer copy, not filesystem
restart. Heat3D, Wave2D, and Tungsten ETD use `CheckpointService` for
filesystem restart.

## API symbols

| Symbol | Header | Role |
|--------|--------|------|
| `CheckpointMetadata`, `DomainParams`, `DecompositionMeta`, `kCheckpointFormatVersion`, `to_json`, `from_json` | `checkpoint_metadata.hpp` | Versioned sidecar JSON (schema version 1) |
| `PublishedFieldBrick`, `PublishOutcome`, `PublishWriteHook`, `publish_checkpoint_directory`, `make_publish_ok`, `make_publish_failed` | `publish.hpp` | Serial atomic directory publish (tests / single-rank bricks) |
| `write_real_brick_mpi` | `brick_io.hpp` | Kernel MPI-IO of one owned float64 brick (no frontend include) |
| `CheckpointService`, `CheckpointConfig`, `checkpoint_config_from_json` | `kernel/simulation/checkpoint_service.hpp` | Collective save/load, JSON `checkpoint.every` / `checkpoint.directory` / `restart_from` |

Callers fill `accepted_time` and `accepted_increment` from driver-owned
`pfc::sim::Time` (`get_current()` / `get_increment()`). Publish does not
construct or advance `Time`.

Field payloads are injectable `PublishedFieldBrick` views (`std::span<const
std::byte>`). Catch2 and drivers can build bricks from owned
`std::vector<double>` without waiting on sibling #166 payload carriers. A
future adapter from those carriers may live outside this header.

## On-disk layout

A checkpoint is a **directory** (not a single opaque file):

```text
<final_dir>/
  metadata.json          # CheckpointMetadata JSON (format_version, …)
  fields/
    <field_id>.bin       # raw float64 bytes, Fortran-order owned cells
```

A bundle is considered loadable only when `final_dir` exists as a directory
and contains readable `metadata.json` with `format_version` equal to
`kCheckpointFormatVersion` (currently 1). `from_json` rejects any other
schema version.

`CheckpointService` writes bundles at `<directory>/step_<increment>/`. JSON:

```json
{
  "restart_from": "results/ckpt/step_10",
  "checkpoint": { "every": 10, "directory": "results/ckpt" }
}
```

`restart_from` restores owned **host `float64`** fields, `Time` accepted
increment/time, result counter, and integrator method identity. Complex
workspace hats (e.g. tungsten `N_hat`) are omitted and recomputed on the
next step. Grid or method mismatch is a hard error that names the field.
Halos are not stored; exchange them after load.

## Atomicity protocol

1. Reject if `final_dir` already exists.
2. Stage under sibling `<final_dir>.publishing/` (same parent path — same
   filesystem required for atomic directory `rename`).
3. Write `metadata.json`, then each `fields/<id>.bin`.
4. `std::filesystem::rename(staging, final_dir)`.
5. On any failure: best-effort `remove_all(staging)`; never leave a half-written
   `final_dir` that could be mistaken for a complete checkpoint.

`publish_checkpoint_directory` is the serial, injectable-brick leaf (Catch2
doubles, crash-injection). Multi-rank production I/O is `CheckpointService`:
every rank writes its owned subarray through `write_real_brick_mpi` (same
Fortran-order contract as `BinaryWriter` / `BinaryReader`), rank 0 writes
`metadata.json`, all ranks barrier, then rank 0 renames staging to `final_dir`.
Kernel code does not include frontend `BinaryWriter`. Optional
`DecompositionMeta` records the writing layout; restore currently requires
the same global grid (method identity and domain origin/spacing/size).

## What is published (and what is not)

**Include:** accepted owned field cells plus irreducible metadata (format
version, accepted time/increment, domain parameters, optional decomposition,
method identity).

**Exclude** (recomputable / transient workspace — do not put these in bricks):

- Stage buffers and per-step RHS scratch
- FFT plans and spectral operator caches
- Exponential coefficient tables that can be rebuilt from `L` and `dt`
- Stepper in-memory rollback buffers (e.g. `EulerStepper` `m_u_checkpoint`)

## Difference from `BinaryWriter` dumps

| | `BinaryWriter` / `ResultsWriter` | `publish_checkpoint_directory` |
|--|----------------------------------|--------------------------------|
| Purpose | Scheduled periodic field dumps, post-processing | Durable accepted-state restart bundle |
| Metadata | None in file (sidecar out of band) | Versioned `metadata.json` in the bundle |
| Atomicity | Each write opens/truncates a path | Stage-then-rename of a directory |
| Kernel layering | Frontend writer | Kernel headers (`brick_io` MPI-IO or ofstream bricks) |

## See also

- [`time_integration_contract.md`](../concepts/time_integration_contract.md) §6
- [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)
- [`class_tour.md`](../reference/class_tour.md)
