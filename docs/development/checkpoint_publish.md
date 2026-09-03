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

This is the **one** durable publish protocol, and it is MPI-collective.
Framework restart is `pfc::sim::CheckpointService`
(`kernel/simulation/checkpoint_service.hpp`): it builds the metadata and calls
`publish_checkpoint_directory` with a callback that writes every owned float64
brick through `brick_io.hpp` (collective MPI-IO); rank 0 writes
`metadata.json`; the stage→rename step runs only after every rank agreed the
writes succeeded, so an interrupted write never becomes a loadable bundle. It is distinct from scheduled headerless field dumps written
by `ResultsWriter` / frontend `BinaryWriter`
(see [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)).

In-memory `state_capture` restore is a payload-buffer copy, not filesystem
restart. Heat3D, Wave2D, and Tungsten ETD use `CheckpointService` for
filesystem restart.

## API symbols

| Symbol | Header | Role |
|--------|--------|------|
| `CheckpointMetadata`, `DomainParams`, `DecompositionMeta`, `kCheckpointFormatVersion`, `to_json`, `from_json` | `checkpoint_metadata.hpp` | Versioned sidecar JSON (schema version 1) |
| `PublishOutcome`, `FieldsWriter`, `publish_checkpoint_directory(final_dir, meta, comm, write_fields)`, `make_publish_ok`, `make_publish_failed` | `publish.hpp` | MPI-collective atomic directory publish (the only publisher) |
| `write_real_brick_mpi` | `brick_io.hpp` | Kernel MPI-IO of one owned float64 brick (no frontend include) |
| `CheckpointService`, `CheckpointConfig`, `checkpoint_config_from_json` | `kernel/simulation/checkpoint_service.hpp` | Collective save/load, JSON `checkpoint.every` / `checkpoint.directory` / `restart_from` |

Callers fill `accepted_time` and `accepted_increment` from driver-owned
`pfc::sim::Time` (`get_current()` / `get_increment()`). Publish does not
construct or advance `Time`.

Field bricks are written by the caller inside the `FieldsWriter` callback,
which receives the staging `fields/` directory and runs on every rank.
Production uses `write_real_brick_mpi` (owned cells of a `pfc::Field`, host or
device via the host mirror); Catch2 doubles write small one-rank bricks the
same way, or throw to inject a mid-publish failure.

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

`restart_from` restores owned **`float64`** fields (host, or device through the
host mirror — `CheckpointService` methods are templated on `MemorySpace`), `Time` accepted
increment/time, result counter, and integrator method identity. Complex
workspace hats (e.g. tungsten `N_hat`) are omitted and recomputed on the
next step. Grid or method mismatch is a hard error that names the field.
Halos are not stored; exchange them after load.

## Atomicity protocol

1. Rank 0 checks that `final_dir` does not exist; the decision is broadcast and
   every rank returns the same failed outcome if it does.
2. Rank 0 clears and creates the sibling staging dir `<final_dir>.publishing/fields`
   (same parent path — same filesystem required for atomic directory `rename`);
   all ranks barrier.
3. Every rank runs the `FieldsWriter` callback (collective `write_real_brick_mpi`
   per field, same Fortran-order contract as `BinaryWriter` / `BinaryReader`);
   rank 0 writes `metadata.json`.
4. `MPI_Allreduce(MIN)` of per-rank success. On failure rank 0 removes staging
   and `final_dir`; every rank returns a failed outcome carrying its message.
5. Rank 0 `std::filesystem::rename(staging, final_dir)`; a final agreement and
   barrier make the bundle visible to all ranks at once.

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
| Kernel layering | Frontend writer | Kernel headers (`brick_io` collective MPI-IO bricks) |

## See also

- [`time_integration_contract.md`](../concepts/time_integration_contract.md) §6
- [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)
- [`class_tour.md`](../reference/class_tour.md)
