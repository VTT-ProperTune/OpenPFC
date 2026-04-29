<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Operator playbooks (symptom → fix)

Short **if-this-then-that** pages for production-style failures. Deep background stays in [`troubleshooting.md`](troubleshooting.md) and [`mpi_io_layout_checklist.md`](mpi_io_layout_checklist.md).

## Hang or deadlock at first result write

| Check | Action |
|-------|--------|
| All ranks call the writer path | `BinaryWriter` / collective MPI-IO requires **every** rank in the communicator to participate in `write()` with a consistent domain. One rank skipping → hang. |
| Same `mpirun` / MPI as build | Mismatched `libmpi` → undefined behavior; see [`INSTALL.md`](../INSTALL.md) §MPI. |
| Writable path from **job cwd** | Relative paths in JSON are relative to the process cwd, not the input file location. |

## Wrong or “empty” binary files

| Check | Action |
|-------|--------|
| No header in file | Raw layout only — you must track `Lx,Ly,Lz`, dtype, ordering ([`binary_field_io_spec.md`](binary_field_io_spec.md)). |
| Mixed endianness | Files are **native** endian; do not mix writers/readers across incompatible architectures without conversion. |

## HeFFTe / FFT errors at plan or first transform

| Check | Action |
|-------|--------|
| HeFFTe built with **same** MPI as OpenPFC | Rebuild HeFFTe or fix `CMAKE_PREFIX_PATH`. |
| GPU backend | CUDA/HIP toolkit on PATH; GPU HeFFTe prefix matches [`gpu_path_decision.md`](gpu_path_decision.md). |

## NaNs or diverging step

| Check | Action |
|-------|--------|
| `dt` too large for explicit pieces | Reduce `dt`; see [`science_numerics_limits.md`](science_numerics_limits.md). |
| Debug builds | NaN checks may abort with a useful message — see [`debugging.md`](debugging.md). |

## See also

- [`hpc_operator_guide.md`](hpc_operator_guide.md) — index of HPC docs  
