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

This is the durable **restart publication** seam. It is distinct from scheduled
headerless field dumps written by `ResultsWriter` / frontend `BinaryWriter`
(see [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)).

Restore / migration validation is a sibling leaf — this document covers
**publish** only.

## API symbols

| Symbol | Header | Role |
|--------|--------|------|
| `CheckpointMetadata`, `DomainParams`, `DecompositionMeta`, `kCheckpointFormatVersion`, `to_json` | `checkpoint_metadata.hpp` | Versioned sidecar JSON |
| `PublishedFieldBrick`, `PublishOutcome`, `PublishWriteHook`, `publish_checkpoint_directory`, `make_publish_ok`, `make_publish_failed` | `publish.hpp` | Atomic directory publish |

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
and contains readable `metadata.json` with a `format_version` key.

## Atomicity protocol

1. Reject if `final_dir` already exists.
2. Stage under sibling `<final_dir>.publishing/` (same parent path — same
   filesystem required for atomic directory `rename`).
3. Write `metadata.json`, then each `fields/<id>.bin`.
4. `std::filesystem::rename(staging, final_dir)`.
5. On any failure: best-effort `remove_all(staging)`; never leave a half-written
   `final_dir` that could be mistaken for a complete checkpoint.

Unit tests run in the serial `openpfc-tests` binary. Multi-rank MPI-IO bricks
inside the bundle are out of scope for this leaf; optional
`DecompositionMeta` records layout for a restore sibling.

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
| Kernel layering | Frontend writer | Kernel headers only (ofstream bricks) |

## See also

- [`time_integration_contract.md`](../concepts/time_integration_contract.md) §6
- [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)
- [`class_tour.md`](../reference/class_tour.md)
