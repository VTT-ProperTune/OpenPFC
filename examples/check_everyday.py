#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check actual initial fields, configured evolution and output payloads."""
from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from check_foundations import run, table

SOURCE = Path(__file__).resolve().parent


def fields(folder, pattern="field_rank_*.csv"):
    return [
        row
        for path in sorted((folder / "results").glob(pattern))
        for row in table(path)
    ]


def unique_grid(rows, n, ny, nz, origin, spacing):
    seen = set()
    for row in rows:
        coordinates = [float(row[key]) for key in ("x", "y", "z")]
        index = tuple(
            round((x - o) / h) for x, o, h in zip(coordinates, origin, spacing)
        )
        assert index not in seen, index
        seen.add(index)
    assert seen == {(i, j, k) for k in range(nz) for j in range(ny) for i in range(n)}


def check_patterns(binary, root, launch):
    folder = root / "patterns"
    run(
        binary,
        folder,
        launch,
        {"n": 16, "nz": 2, "width": 0.8, "core": 0.5, "circulation": 5},
    )
    data = fields(folder, "fields_rank_*.csv")
    for name in ("gaussian", "vortex_speed", "checkerboard"):
        subset = [row for row in data if row["pattern"] == name]
        unique_grid(subset, 16, 16, 2, (-4, -4, 0), (0.5, 0.5, 0.5))
        for row in subset:
            x, y, z = (float(row[key]) for key in ("x", "y", "z"))
            r = math.hypot(x, y)
            if name == "gaussian":
                exact = math.exp(-(x * x + y * y + z * z) / (2 * 0.8**2))
            elif name == "vortex_speed":
                exact = (
                    0
                    if r == 0
                    else 5 * (1 - math.exp(-r * r / 0.5**2)) / (2 * math.pi * r)
                )
            else:
                exact = -1 if (math.floor(x) + math.floor(y) + math.floor(z)) % 2 else 1
            assert abs(float(row["u"]) - exact) < 2e-14, row
    run(binary, root / "bad-pattern", launch, {"width": 0}, success=False)


def check_catalog(binary, root, launch):
    for specification in (
        {"type": "affine", "offset": 1, "slope": [0.2, -0.1, 0.05]},
        {"type": "constant", "n0": 0.4},
    ):
        folder = root / ("catalog-" + specification["type"])
        run(
            binary,
            folder,
            launch,
            {"n": 16, "nz": 2, "initial_condition": specification},
        )
        data = fields(folder, "fields_rank_*.csv")
        unique_grid(data, 16, 16, 2, (-4, -4, 0), (0.5, 0.5, 0.5))
        for row in data:
            x, y, z = (float(row[key]) for key in ("x", "y", "z"))
            exact = (
                1 + 0.2 * x - 0.1 * y + 0.05 * z
                if specification["type"] == "affine"
                else 0.4
            )
            assert abs(float(row["u"]) - exact) < 2e-14
    result = run(
        binary,
        root / "unknown-catalog",
        launch,
        {"initial_condition": {"type": "unknown"}},
        success=False,
    )
    run(
        binary,
        root / "bad-catalog",
        launch,
        {"initial_condition": {"type": "affine"}},
        success=False,
    )


def check_session(binary, root, launch):
    outputs, documents = [], []
    for extension in ("json", "toml"):
        folder = root / ("session-" + extension)
        folder.mkdir()
        command = [
            *launch,
            str(binary),
            str(SOURCE / "configuration/session" / ("input." + extension)),
        ]
        result = subprocess.run(
            command, cwd=folder, capture_output=True, text=True, timeout=90
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        data = fields(folder)
        unique_grid(data, 16, 4, 4, (-2, 0, 0), (0.5, 1, 1))
        k = math.pi / 4
        amplitude = 0.2 / ((1 + 0.07 * k * k) ** 2 * (1 + 0.06 * k * k))
        for row in data:
            exact = 1 + amplitude * math.cos(k * (float(row["x"]) + 2))
            assert abs(float(row["u"]) - exact) < 2e-13
        outputs.append(
            sorted(data, key=lambda row: tuple(float(row[k]) for k in ("x", "y", "z")))
        )
        documents.append(json.loads((folder / "results/settings.json").read_text()))
    assert outputs[0] == outputs[1] and documents[0] == documents[1]
    bad = json.loads((SOURCE / "configuration/session/input.json").read_text())
    bad["method"] = "fd"
    run(binary, root / "bad-session", launch, bad, success=False)


def check_implicit(binary, root, launch):
    folder = root / "implicit"
    run(binary, folder, launch, {"n": 16, "dt": 0.05, "steps": 20, "diffusivity": 0.7})
    data = fields(folder)
    h = 2 * math.pi / 16
    unique_grid(data, 16, 16, 16, (0, 0, 0), (h, h, h))
    amplitude = 0.2 / (1 + 3 * 0.7 * 0.05) ** 20
    for row in data:
        shape = math.prod(math.cos(float(row[key])) for key in ("x", "y", "z"))
        assert abs(float(row["u"]) - (1 + amplitude * shape)) < 2e-13
    diagnostics = table(folder / "results/diagnostics.csv")
    assert len(diagnostics) == 21
    assert max(abs(float(row["mean"]) - 1) for row in diagnostics) < 2e-13
    run(binary, root / "bad-implicit", launch, {"n": 7}, success=False)


def check_driver(binary, root, launch):
    for end, intervals in ((0.2, [0.07, 0.07, 0.06]), (0.03, [0.03])):
        folder = root / f"driver-{end}"
        run(
            binary,
            folder,
            launch,
            {"n": 16, "dt": 0.07, "end": end, "saveat": min(0.1, end)},
        )
        ledgers = sorted((folder / "results").glob("events_rank_*.csv"))
        first = table(ledgers[0])
        assert all(table(path) == first for path in ledgers)
        steps = [row for row in first if row["hook"] == "step"]
        assert len(steps) == len(intervals)
        assert all(
            abs(float(row["interval"]) - expected) < 1e-14
            for row, expected in zip(steps, intervals)
        )
        assert [row["hook"] for row in first] == ["start", "save"] + [
            hook for _ in intervals for hook in ("apply", "step")
        ] + ["save"]
        amplitude = 0.2 / math.prod(1 + interval for interval in intervals)
        data = fields(folder)
        unique_grid(data, 16, 4, 4, (0, 0, 0), (2 * math.pi / 16, 1, 1))
        for row in data:
            assert (
                abs(float(row["u"]) - (1 + amplitude * math.cos(float(row["x"]))))
                < 2e-13
            )
    run(binary, root / "bad-driver", launch, {"dt": 0}, success=False)


def binary_values(path):
    data = path.read_bytes()
    assert len(data) % 8 == 0
    return struct.unpack("=" + "d" * (len(data) // 8), data)


def affine(i, j, k, t=0):
    return 1 + (-1 + 0.25 * i) + 2 * (2 + 0.5 * j) + 3 * (3 + 0.75 * k) + t


def check_binary(binary, root, launch):
    folder = root / "binary"
    run(binary, folder, launch, {"n": 8})
    values = binary_values(folder / "results/field_0005.bin")
    assert values == tuple(
        affine(i, j, k) for k in range(2) for j in range(4) for i in range(8)
    )


def vtk_piece(path):
    data = path.read_bytes()
    marker = re.search(rb"<AppendedData[^>]*>\s*_", data)
    assert marker
    xml = ET.fromstring(data[: marker.start()] + b"</VTKFile>")
    endian = "<" if xml.attrib["byte_order"] == "LittleEndian" else ">"
    assert xml.attrib["header_type"] == "UInt64"
    image = xml.find("ImageData")
    piece = image.find("Piece")
    extent = list(map(int, piece.attrib["Extent"].split()))
    array = piece.find("PointData/DataArray")
    assert array.attrib["Name"] == "density" and array.attrib["type"] == "Float64"
    begin = marker.end() + int(array.attrib["offset"])
    length = struct.unpack_from(endian + "Q", data, begin)[0]
    values = struct.unpack_from(endian + "d" * (length // 8), data, begin + 8)
    assert length == 8 * math.prod(
        extent[2 * a + 1] - extent[2 * a] + 1 for a in range(3)
    )
    return image, extent, values


def check_vtk(binary, root, launch, ranks):
    folder = root / "vtk"
    run(binary, folder, launch, {"n": 8})
    manifest = json.loads((folder / "results/lesson_manifest.json").read_text())
    assert manifest["steps"] == [0, 4, 8] and manifest["times"] == [0, 0.25, 0.5]
    assert manifest["origin"] == [-1, 2, 3]
    assert (manifest["dx"], manifest["dy"], manifest["dz"]) == (0.25, 0.5, 0.75)
    assert manifest["order"] == "fortran" and manifest["dtype"] == "float64"
    assert manifest["fields"] == ["density", "density_raw"]
    for frame in range(3):
        values = binary_values(folder / f"results/lesson_density_raw_{frame:04d}.bin")
        assert values == tuple(
            affine(i, j, k, 0.25 * frame)
            for k in range(2)
            for j in range(4)
            for i in range(8)
        )
        pieces = sorted((folder / "results").glob(f"density_{frame:04d}*.vti"))
        assert len(pieces) == ranks, pieces
        seen = set()
        for path in pieces:
            image, extent, values = vtk_piece(path)
            assert list(map(float, image.attrib["Origin"].split())) == [-1, 2, 3]
            assert list(map(float, image.attrib["Spacing"].split())) == [
                0.25,
                0.5,
                0.75,
            ]
            indices = [
                (i, j, k)
                for k in range(extent[4], extent[5] + 1)
                for j in range(extent[2], extent[3] + 1)
                for i in range(extent[0], extent[1] + 1)
            ]
            for index, value in zip(indices, values):
                assert index not in seen and value == affine(*index, 0.25 * frame)
                seen.add(index)
        assert len(seen) == 64
    xdmf = list((folder / "results").glob("*.xmf")) + list(
        (folder / "results").glob("*.xdmf")
    )
    assert len(xdmf) == 1, xdmf
    document = ET.parse(xdmf[0])
    assert [float(item.attrib["Value"]) for item in document.findall(".//Time")] == [
        0,
        0.25,
        0.5,
    ]
    for item in document.findall(".//DataItem"):
        if item.attrib.get("Format") == "Binary":
            assert (xdmf[0].parent / item.text.strip()).is_file()


def verify_first_domain(bin_dir, root, launch, ranks):
    assert "(-0.625,0.5,0)" in run(bin_dir / "first_domain", root)
    axes = table(root / "results/domain.csv")
    assert [float(row["upper"]) for row in axes] == [1, 2, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument(
        "--case",
        choices=[
            "first_domain",
            "14_custom_field_initializer",
            "10_ui_register_ic",
            "json_read",
            "04_diffusion_model",
            "05_simulator",
            "write_results",
            "11_write_results",
        ],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bin_dir = args.bin_dir.resolve()
    ranks = args.ranks
    launch = [args.launcher, args.rank_flag, str(ranks)] if args.launcher else []
    with tempfile.TemporaryDirectory(
        prefix="example-check-", dir=args.output
    ) as temporary:
        base = Path(temporary).resolve()
        if args.case in (None, "first_domain"):
            root = base / "first_domain"
            root.mkdir()
            verify_first_domain(bin_dir, root, launch, ranks)
        if args.case in (None, "14_custom_field_initializer"):
            root = base / "14_custom_field_initializer"
            root.mkdir()
            check_patterns(bin_dir / "14_custom_field_initializer", root, launch)
        if args.case in (None, "10_ui_register_ic"):
            root = base / "10_ui_register_ic"
            root.mkdir()
            check_catalog(bin_dir / "10_ui_register_ic", root, launch)
        if args.case in (None, "json_read"):
            root = base / "json_read"
            root.mkdir()
            check_session(bin_dir / "json_read", root, launch)
        if args.case in (None, "04_diffusion_model"):
            root = base / "04_diffusion_model"
            root.mkdir()
            check_implicit(bin_dir / "04_diffusion_model", root, launch)
        if args.case in (None, "05_simulator"):
            root = base / "05_simulator"
            root.mkdir()
            check_driver(bin_dir / "05_simulator", root, launch)
        if args.case in (None, "write_results"):
            root = base / "write_results"
            root.mkdir()
            check_binary(bin_dir / "write_results", root, launch)
        if args.case in (None, "11_write_results"):
            root = base / "11_write_results"
            root.mkdir()
            check_vtk(bin_dir / "11_write_results", root, launch, ranks)
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
