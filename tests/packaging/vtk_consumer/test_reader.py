# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read the installed writer's master file directly and verify seam topology.

Set OPENPFC_VTK_CONSUMER to the independently built consumer binary and,
optionally, OPENPFC_MPIEXEC / OPENPFC_MPIEXEC_NUMPROC_FLAG for the site launcher.
Missing VTK or an unbuilt consumer is an explicit skip, never qualification.
"""

import os
from pathlib import Path
import subprocess

import pytest

pytest.importorskip("vtkmodules.vtkIOXML")
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLPImageDataReader
from vtkmodules.vtkCommonDataModel import vtkDataObject

SIZE = (11, 9, 7)
ORIGIN = (0.3, 0.125, 0.55)
SPACING = (0.6, 0.25, 1.1)


@pytest.fixture(scope="session")
def writer():
    path = os.environ.get("OPENPFC_VTK_CONSUMER")
    if not path:
        pytest.skip("NOT_QUALIFIED: build the installed vtk_consumer first")
    binary = Path(path).resolve(strict=True)
    assert binary.is_file()
    return binary


def write_field(binary, cwd, output, ranks, split):
    launcher = os.environ.get("OPENPFC_MPIEXEC", "mpiexec")
    count_flag = os.environ.get("OPENPFC_MPIEXEC_NUMPROC_FLAG", "-n")
    subprocess.run(
        [launcher, count_flag, str(ranks), str(binary), output, str(split)],
        cwd=cwd, check=True, capture_output=True, text=True, timeout=90,
    )
    extension = "vti" if ranks == 1 else "pvti"
    return cwd / output / f"field_0001.{extension}"


def read_field(path):
    reader = vtkXMLPImageDataReader() if path.suffix == ".pvti" else vtkXMLImageDataReader()
    errors = []
    reader.AddObserver("ErrorEvent", lambda _object, event: errors.append(event))
    reader.SetFileName(str(path))
    reader.Update()
    assert not errors
    assert reader.GetErrorCode() == 0
    data = reader.GetOutput()
    assert data.GetDimensions() == SIZE
    assert data.GetNumberOfPoints() == 11 * 9 * 7
    assert data.GetNumberOfCells() == 10 * 8 * 6
    assert data.GetOrigin() == ORIGIN
    assert data.GetSpacing() == SPACING
    values = data.GetPointData().GetArray("plane")
    assert values is not None
    assert values.GetNumberOfTuples() == data.GetNumberOfPoints()
    for k in range(SIZE[2]):
        for j in range(SIZE[1]):
            for i in range(SIZE[0]):
                slot = i + SIZE[0] * (j + SIZE[1] * k)
                x = tuple(ORIGIN[d] + index * SPACING[d] for d, index in enumerate((i, j, k)))
                assert data.GetPoint(slot) == pytest.approx(x, rel=0, abs=1e-14)
                expected = x[0] + 0.2 * x[1] - 0.1 * x[2]
                assert values.GetTuple1(slot) == pytest.approx(expected, rel=0, abs=1e-13)
    return data


def surface(data):
    contour = vtkContourFilter()
    contour.SetInputData(data)
    contour.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, "plane")
    contour.SetValue(0, 2.7)
    contour.Update()
    mesh = contour.GetOutput()
    assert mesh.GetNumberOfCells() > 0
    points = [mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())]
    # Compare geometric triangles, independently of reader point numbering.
    triangles = []
    for index in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(index)
        assert cell.GetNumberOfPoints() == 3
        triangle = tuple(sorted(points[cell.GetPointId(i)] for i in range(3)))
        triangles.append(triangle)
    return sorted(points), sorted(triangles)


@pytest.mark.parametrize("ranks,split", [(1, 0)] + [(2, d) for d in range(3)] + [(4, d) for d in range(6)])
def test_parallel_master_matches_serial_geometry(writer, tmp_path, ranks, split):
    serial = read_field(write_field(writer, tmp_path, "serial reference", 1, 0))
    parallel = read_field(write_field(writer, tmp_path, "nested outputs/parallel pieces", ranks, split))
    a = serial.GetPointData().GetArray("plane")
    b = parallel.GetPointData().GetArray("plane")
    assert [a.GetTuple1(i) for i in range(a.GetNumberOfTuples())] == [
        b.GetTuple1(i) for i in range(b.GetNumberOfTuples())
    ]
    # An identical full surface rules out seam holes, caps and duplicated cells.
    assert surface(serial) == surface(parallel)
