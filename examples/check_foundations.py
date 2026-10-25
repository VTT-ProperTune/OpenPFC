#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the geometry, Fourier, interpolation and regularity teaching programs."""

from __future__ import annotations

import argparse
import cmath
import csv
import json
import io
import math
from pathlib import Path
import subprocess
import tempfile


def table(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def run(binary, directory, launch=(), parameters=None, success=True):
    directory.mkdir(parents=True, exist_ok=True)
    command = [*launch, str(binary)]
    if parameters is not None:
        source = directory / "input.json"
        source.write_text(json.dumps(parameters))
        command.append(str(source))
    result = subprocess.run(
        command, cwd=directory, text=True, capture_output=True, timeout=90
    )
    assert (result.returncode == 0) == success, (command, result.stdout, result.stderr)
    return result.stdout


def profile(name, x):
    sine, cosine = math.sin(x), math.cos(x)
    if name == "finite-smoothness":
        return abs(sine) ** 3, 3 * sine * cosine * abs(sine)
    if name == "smooth-bump":
        q = math.sin(x / 2) ** 2
        if q < 1e-14:
            return 0.0, 0.0
        value = math.exp(-1 / q)
        return value, 0.5 * value * sine / q**2
    if name == "analytic-strip":
        value = 1 / (1 + math.sin(x / 2) ** 2)
        return value, -math.sin(x / 2) * math.cos(x / 2) * value**2
    if name == "analytic":
        return math.exp(sine), cosine * math.exp(sine)
    return math.sin(10 * x), 10 * math.cos(10 * x)


def dft_derivative_error(name, n):
    """Direct complex DFT of the full global signal; no FFT or cotangent matrix."""
    nodes = [2 * math.pi * j / n for j in range(n)]
    samples = [profile(name, x)[0] for x in nodes]
    coefficients = [
        sum(value * cmath.exp(-1j * k * x) for x, value in zip(nodes, samples)) / n
        for k in range(n)
    ]
    errors = []
    for x in nodes:
        value = 0j
        for index, coefficient in enumerate(coefficients):
            k = index if index < n // 2 else index - n
            if index == n // 2:
                continue
            value += 1j * k * coefficient * cmath.exp(1j * k * x)
        errors.append(abs(value.real - profile(name, x)[1]))
    return max(errors)


def verify_02_domain_decomposition(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    decomposition = root / "decomposition"
    run(binary("02_domain_decomposition"), decomposition, launch)
    boxes = table(decomposition / "results/decomposition.csv")
    assert len(boxes) == ranks
    owned = set()
    for row in boxes:
        low = [int(row[key]) for key in ("x0", "y0", "z0")]
        size = [int(row[key]) for key in ("nx", "ny", "nz")]
        for z in range(low[2], low[2] + size[2]):
            for y in range(low[1], low[1] + size[1]):
                for x in range(low[0], low[0] + size[0]):
                    assert (x, y, z) not in owned
                    owned.add((x, y, z))
    assert owned == {(x, y, z) for z in range(4) for y in range(8) for x in range(16)}


def verify_08_discrete_fields(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    assert "u(12,6,0)=49 owners=1" in run(
        binary("08_discrete_fields"), root / "coordinates", launch
    )


def verify_03_parallel_fft(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    transform = root / "transform"
    run(binary("03_parallel_fft"), transform, launch)
    rows = [
        row
        for path in (transform / "results").glob("spectrum_rank_*.csv")
        for row in table(path)
    ]
    assert len(rows) == 5 * 4 * 2
    assert len(
        {tuple(int(row[key]) for key in ("i", "j", "k")) for row in rows}
    ) == len(rows)
    for row in rows:
        i, j, k = (int(row[key]) for key in ("i", "j", "k"))
        expected = ({0: 64, 1: 8j, 2: 16}.get(i, 0)) if j == k == 0 else 0
        value = complex(float(row["real"]), float(row["imag"]))
        assert abs(value - expected) < 1e-11, row


def verify_periodic_collocation(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    collocation = root / "collocation"
    run(binary("periodic_collocation"), collocation, launch)
    rows = [
        row
        for path in (collocation / "results").glob("derivative_rank_*.csv")
        for row in table(path)
    ]
    assert len(rows) == 48
    for row in rows:
        assert abs(float(row["matrix"]) - float(row["fourier"])) < 1e-11
        if row["profile"] == "analytic":
            assert abs(float(row["fourier"]) - float(row["exact"])) < 1e-10


def verify_spectral_smoothness(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    smoothness = root / "smoothness"
    sizes = [6, 20, 22, 50]
    run(binary("spectral_smoothness"), smoothness, launch, {"sizes": sizes})
    rows = table(smoothness / "results/smoothness.csv")
    assert len(rows) == len(sizes) * 4
    for row in rows:
        expected = dft_derivative_error(row["profile"], int(row["n"]))
        assert abs(float(row["fourier_error"]) - expected) < 2e-11, row
        assert abs(float(row["matrix_error"]) - expected) < 2e-11, row
    run(
        binary("spectral_smoothness"),
        root / "bad-profile",
        launch,
        {"profiles": ["unknown"]},
        success=False,
    )


def verify_world_helpers_example(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    spacing = list(
        csv.DictReader(
            io.StringIO(run(binary("world_helpers_example"), root / "spacing"))
        )
    )
    assert len(spacing) == 10
    for row in spacing:
        h = 2 * math.pi / (5 if row["grid"] == "periodic" else 4)
        assert abs(float(row["x"]) - int(row["index"]) * h) < 1e-13
        assert abs(float(row["spacing"]) - h) < 1e-14
        assert abs(float(row["upper"]) - 2 * math.pi) < 1e-14


def verify_world_query_helpers_example(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    queries = list(
        csv.DictReader(
            io.StringIO(run(binary("world_query_helpers_example"), root / "queries"))
        )
    )
    assert len(queries) == 2
    for row in queries:
        assert int(row["points"]) == 30 and int(row["dimension"]) == 3
        assert float(row["lx"]) == (2.5 if int(row["periodic"]) else 2)
        assert float(row["ly"]) == 6 and float(row["lz"]) == 6
        assert float(row["volume"]) == 90


def verify_16_strong_types_demo(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    strong = list(
        csv.DictReader(
            io.StringIO(run(binary("16_strong_types_demo"), root / "strong"))
        )
    )
    assert len(strong) == 1 and int(strong[0]["points"]) == 64
    assert [float(strong[0][key]) for key in ("cx", "cy", "cz")] == [0, 0, 2]


def verify_06_multi_index(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    offsets = run(binary("06_multi_index"), root / "boxes").splitlines()
    assert [int(line.split(",")[3]) for line in offsets] == list(range(8))


def verify_07_array(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    assert "owned=96 padded=288 first_owned_offset=55 first_x=0 mismatch=0" in run(
        binary("07_array"), root / "storage"
    )


def verify_cardinal_interpolation(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    directory = root / "interpolation"
    run(binary("cardinal_interpolation"), directory)
    rows = table(directory / "results/interpolation.csv")
    assert len(rows) == 1203
    overshoot = 0
    for row in rows:
        kind, x, value = int(row["profile"]), float(row["x"]), float(row["value"])
        if abs(x - round(x)) < 1e-12:
            expected = (
                float(x == 0)
                if kind == 0
                else float(abs(x) <= 3) if kind == 1 else max(0, 1 - abs(x) / 3)
            )
            assert abs(value - expected) < 1e-13
        if kind == 0:
            expected = 1 if abs(x) < 1e-14 else math.sin(math.pi * x) / (math.pi * x)
            assert abs(value - expected) < 1e-13
        if kind == 1:
            overshoot = max(overshoot, value - 1)
    assert overshoot > 0.1


def verify_layout(bin_dir, root, launch, ranks):
    binary = bin_dir / "09_parallel_fft_high_level"
    run(binary, root / "layout", launch)
    if ranks == 1:
        run(binary, root / "local", parameters={"implementation": "local"})
    else:
        run(
            binary,
            root / "rejected-local",
            launch,
            {"implementation": "local"},
            success=False,
        )


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
            "02_domain_decomposition",
            "06_multi_index",
            "07_array",
            "08_discrete_fields",
            "03_parallel_fft",
            "09_parallel_fft_high_level",
            "cardinal_interpolation",
            "periodic_collocation",
            "spectral_smoothness",
            "world_helpers_example",
            "world_query_helpers_example",
            "16_strong_types_demo",
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
        if args.case in (None, "02_domain_decomposition"):
            root = base / "02_domain_decomposition"
            root.mkdir()
            verify_02_domain_decomposition(bin_dir, root, launch, ranks)
        if args.case in (None, "06_multi_index"):
            root = base / "06_multi_index"
            root.mkdir()
            verify_06_multi_index(bin_dir, root, launch, ranks)
        if args.case in (None, "07_array"):
            root = base / "07_array"
            root.mkdir()
            verify_07_array(bin_dir, root, launch, ranks)
        if args.case in (None, "08_discrete_fields"):
            root = base / "08_discrete_fields"
            root.mkdir()
            verify_08_discrete_fields(bin_dir, root, launch, ranks)
        if args.case in (None, "03_parallel_fft"):
            root = base / "03_parallel_fft"
            root.mkdir()
            verify_03_parallel_fft(bin_dir, root, launch, ranks)
        if args.case in (None, "09_parallel_fft_high_level"):
            root = base / "09_parallel_fft_high_level"
            root.mkdir()
            verify_layout(bin_dir, root, launch, ranks)
        if args.case in (None, "cardinal_interpolation"):
            root = base / "cardinal_interpolation"
            root.mkdir()
            verify_cardinal_interpolation(bin_dir, root, launch, ranks)
        if args.case in (None, "periodic_collocation"):
            root = base / "periodic_collocation"
            root.mkdir()
            verify_periodic_collocation(bin_dir, root, launch, ranks)
        if args.case in (None, "spectral_smoothness"):
            root = base / "spectral_smoothness"
            root.mkdir()
            verify_spectral_smoothness(bin_dir, root, launch, ranks)
        if args.case in (None, "world_helpers_example"):
            root = base / "world_helpers_example"
            root.mkdir()
            verify_world_helpers_example(bin_dir, root, launch, ranks)
        if args.case in (None, "world_query_helpers_example"):
            root = base / "world_query_helpers_example"
            root.mkdir()
            verify_world_query_helpers_example(bin_dir, root, launch, ranks)
        if args.case in (None, "16_strong_types_demo"):
            root = base / "16_strong_types_demo"
            root.mkdir()
            verify_16_strong_types_demo(bin_dir, root, launch, ranks)
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
