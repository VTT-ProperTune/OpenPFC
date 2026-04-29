<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Post-processing binary field files (external tools)

OpenPFC **`BinaryWriter`** files are **raw MPI-IO payloads**: no header, **Fortran (column-major)** order, **`float64`** per sample for typical real fields ([`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)). This page sketches how to reason about **offline** analysis; for restart inside OpenPFC, use **`BinaryReader`** with the same decomposition.

## What you must know out-of-band

To interpret bytes correctly you need **metadata** that is not in the file:

| Metadata | Example |
|----------|---------|
| Global grid | `Lx`, `Ly`, `Lz` |
| Local brick | Which index range this file’s rank owned (or concatenate ranks in order) |
| Dtype | `float64` real (default writer path) |
| Layout | Fortran index order: `i` varies fastest in the **first** dimension in memory for `MPI_ORDER_FORTRAN` 3D subarrays |

For a full run, record the **same JSON/TOML** (or export a small YAML sidecar next to outputs).

## Python / NumPy (illustrative)

This is **not** a substitute for matching OpenPFC’s decomposition when reading a **single-rank** dump; production workflows often aggregate on rank 0 or use VTK export instead.

```python
import numpy as np

# Example ONLY: one rank wrote a contiguous brick of nx*ny*nz doubles
# Replace with your actual local sizes from the simulation metadata.
nx, ny, nz = 32, 32, 32
data = np.fromfile("frame.bin", dtype=np.float64)
assert data.size == nx * ny * nz
# Fortran-order reshape: first index (i) contiguous in memory
field = data.reshape((nx, ny, nz), order="F")
```

**Endianness:** files use **native** byte order of the machine that wrote them. If you move files between architectures, convert explicitly.

## Safer paths for visualization

- **VTK** from code: [`tutorials/vtk_paraview_workflow.md`](../tutorials/vtk_paraview_workflow.md).  
- **ParaView** / custom readers: prefer exporting **VTK** or documented **HDF5** profiling paths over ad-hoc raw binary unless you control the full metadata story.

## See also

- [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md) — normative format  
- [`io_results.md`](io_results.md) — writers overview  
