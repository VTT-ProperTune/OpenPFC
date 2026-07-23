<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# ADR 0008: I/O and checkpoint formats for 0.2

## Status

Accepted (2026-07-23). Writers scheduled for M10; checkpoint for M11.

## Context

The audit (§10, §17.5) found: `BinaryWriter` (collective MPI-IO raw bricks +
sidecar spec) is exemplary; `VTKWriter` is solid but not registered in the
default writer catalog (so JSON `"writer": "vtk"` silently no-ops); there is no
HDF5 field output (HDF5 exists only for profiling); and checkpointing is
write-only with no loader and non-collective publication.

## Decision

1. **Hot-path field output stays raw bricks + sidecar** (the `BinaryWriter`
   format), because it is collective, fast, and already the documented on-disk
   spec (`docs/reference/binary_field_io_spec.md`).
2. **HDF5/XDMF is added as an optional `ResultsWriter`** behind the writer
   catalog and the `OpenPFC_ENABLE_HDF5` option (M10). It is the interoperable
   format for post-processing; it does not replace raw bricks on hot paths.
3. **`VTKWriter` is registered** in `default_results_writer_catalog()` (M10) and
   unknown writer types become a hard error, not a silent skip.
4. **The checkpoint bundle uses the same raw brick format + a JSON metadata
   sidecar** (M11): fields written through the collective `BinaryWriter` path,
   metadata via `CheckpointMetadata` (`to_json`/new `from_json`), published
   atomically (stage → rename). A `restart_from: <dir>` config key restores
   fields, accepted time, result counter, and integrator-method identity.

## Consequences

- One field-serialization format (raw bricks) spans results output and
  checkpoints; HDF5/XDMF is an interop add-on, not a second core path.
- The `ResultsWriter` contract narrows so non-file sinks are expressible
  (filename templating moves to a `FileResultsWriter` intermediate, M10).
- Restart becomes a first-class, collective, crash-consistent operation (M11),
  replacing the manual `result_counter` + `from_file` IC ritual.
