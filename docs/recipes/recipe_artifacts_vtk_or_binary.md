<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Recipe: VTK or binary artifacts

**Goal:** go from **running code** to **files on disk** you can open in ParaView or analyze offline.

## Path A — VTK (ParaView / VisIt)

Best for interactive 3D visualization without writing parsers.

1. Follow [`tutorials/vtk_paraview_workflow.md`](../tutorials/vtk_paraview_workflow.md) (runs `11_write_results` / `12_cahn_hilliard`-style flows).  
2. Writers overview: [`io_results.md`](../user_guide/io_results.md) (`VTKWriter` is typically wired in code for examples).

## Path B — Raw binary (MPI-IO)

Best for restarts and custom post-processing; **no file header** — you need sidecar metadata.

1. Normative layout: [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md).  
2. Offline sketches (Python, caveats): [`postprocess_binary_fields.md`](../user_guide/postprocess_binary_fields.md).  
3. End-to-end narrative with a small app: [`tutorials/end_to_end_visualization.md`](../tutorials/end_to_end_visualization.md).

## Path C — PNG (2D quick look)

Small grayscale snapshots (e.g. Allen–Cahn app): see [`io_results.md`](../user_guide/io_results.md) and [`applications.md`](../user_guide/applications.md).

## Checklist before debugging “empty files”

- Same MPI for build and run; writers are often **collective** — all ranks must participate ([`mpi_io_layout_checklist.md`](../hpc/mpi_io_layout_checklist.md)).  
- Paths in JSON point to a writable directory from the job’s cwd.  
