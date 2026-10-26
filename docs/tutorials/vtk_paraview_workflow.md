<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: VTK output and ParaView

A field needs geometry and physical time before a viewer can interpret it.
OpenPFC's `SnapshotSeries` writes VTK and raw arrays together with a manifest
and XDMF series. The [result-series lesson](https://vtt-propertune.github.io/OpenPFC/learn/output/vtk-series/)
explains the complete runnable example, its source and its independent checks.
This page gives the path from that output to a viewer.

## Prerequisites

Build OpenPFC with examples enabled through `scripts/build.sh`, as described in
[Quick start](../quickstart.md). Load the MPI environment matching that build.
Run MPI commands inside your workstation resources or scheduler allocation;
use the [LUMI guide](../hpc/INSTALL.LUMI.md) on that cluster.

Install [ParaView](https://www.paraview.org/) or another VTK-capable viewer on
the machine where you will inspect the results.

## Write a small result series

From the checkout root, the retained target name `11_write_results` now runs
the `examples/output/vtk-series/model.cpp` lesson:

```bash
mpirun -n 1 ./builds/release/examples/11_write_results
```

It prints `Wrote 3 frames` and writes under `results/`:

- `density_0000.vti`, `density_0001.vti` and `density_0002.vti` in a serial run;
- raw binary frames for the same owned field;
- `lesson_manifest.json`, recording geometry, steps 0/4/8 and times 0/0.25/0.5;
- XDMF metadata for the raw binary time series.

The field is prescribed as `1 + x + 2*y + 3*z + t`; it is an output-format
example, not a simulated PDE solution. Padding cells carry a sentinel value
that must not appear in the saved field. The lesson tests decode the payloads
and verify the physical coordinates, values and times.

## Open the VTK frames

In ParaView choose **File → Open**, select `density_0000.vti`, then **Apply**.
Select the scalar array `density`. To inspect the sequence, open all three
`density_*.vti` frames as a file series and use the playback controls.
Consult `lesson_manifest.json` for the physical times: a filename ordinal is
not a simulation step or time.

With multiple ranks, VTK output uses `.vti` pieces and a `.pvti` master with
non-overlapping point extents. Some readers do not assemble these into a single
image. Inspect an individual piece or use the generated XDMF series for the
global raw field; do not interpret a reader's assembly limitation as missing
simulation data.

## Use VTK in a configured application

The default results-writer catalog includes `binary` and `vtk`, with `hdf5`
when that build enables HDF5. In a tungsten-style input set a field's
`"writer": "vtk"` and use a `.vti` filename template. All ranks must participate
in each write. An unknown writer key is an error.

For programmatic output use `SnapshotSeries` or attach `VTKWriter` to the
session's save hook. See [Results I/O](../user_guide/io_results.md) and the
[configuration reference](../reference/spectral_app_config_reference.md).
