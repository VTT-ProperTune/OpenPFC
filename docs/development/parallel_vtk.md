<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Parallel VTK ImageData verification

`VTKWriter` accepts non-overlapping owned point samples. For parallel output,
it exchanges the next point along each upper process boundary, including edge
and corner samples. Each piece then spans all cells whose lower corner it owns.
Adjacent pieces share points and have disjoint cell interiors. The assembled
image retains the serial point count, cell count, origin, spacing and values.
Only boundary samples cross ranks; the writer does not gather the global field.

Open the `.pvti` master directly in ParaView or `vtkXMLPImageDataReader`.
Keep the `.vti` pieces beside it; their paths are relative to the master.
Serial `.vti` output keeps the same point-data representation. An invalid
partition or an unrepresentable MPI exchange is rejected collectively before
piece payloads are written. Existing field-size and file-open/write error
agreements still apply.

The optional installed consumer uses a real OpenPFC domain and field with
non-cubic dimensions, anisotropic spacing and a voxel-centre origin. Its pytest
checks load the master directly, verify every physical sample and compare a
seam-crossing isosurface with serial output. The matrix covers one, two and four
ranks, all three split axes and two-axis four-rank decompositions. It also
exercises relative nested output paths and the writer's owned-sample status.

Use a Python environment containing `pytest` and `vtk`, then run:

```sh
cmake -S tests/packaging/vtk_consumer -B /path/to/consumer-build \
  -DCMAKE_PREFIX_PATH=/path/to/installed/openpfc
cmake --build /path/to/consumer-build --parallel
OPENPFC_VTK_CONSUMER=/path/to/consumer-build/vtk_consumer \
  python -m pytest tests/packaging/vtk_consumer/test_reader.py -q
```

Set `OPENPFC_MPIEXEC` and `OPENPFC_MPIEXEC_NUMPROC_FLAG` for a site launcher.
A missing consumer or VTK dependency is reported as skipped and does not
qualify parallel-reader compatibility. No VTK runtime dependency is added to
OpenPFC itself.
