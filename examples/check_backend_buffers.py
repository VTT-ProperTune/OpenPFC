#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check complete native FFT fields, sparse Fourier coefficients and ownership."""

import argparse
import itertools
import json
import math
from pathlib import Path

from check_foundations import run, table


def validate(folder, shape, length, ranks):
    volume = math.prod(shape)
    points, modes = {}, {}
    files = sorted((folder / "results").glob("field_rank_*.csv"))
    spectra = sorted((folder / "results").glob("modes_rank_*.csv"))
    assert len(files) == len(spectra) == ranks
    for path in files:
        for row in table(path):
            index = tuple(int(row[key]) for key in ("i", "j", "k"))
            assert index not in points
            points[index] = row
            assert all(0 <= i < n for i, n in zip(index, shape))
            x, y, z = (float(row[key]) for key in ("x", "y", "z"))
            assert all(
                abs(q - i * period / n) < 3e-14
                for q, i, period, n in zip((x, y, z), index, length, shape)
            )
            i, j, k = index
            expected = (
                1
                + 0.3 * math.cos(2 * math.pi * i / shape[0])
                + 0.2
                * math.sin(4 * math.pi * j / shape[1])
                * math.cos(2 * math.pi * k / shape[2])
            )
            assert abs(float(row["initial"]) - expected) < 3e-14
            assert abs(float(row["u"]) - expected) < 2e-12
    assert len(points) == volume
    for path in spectra:
        for row in table(path):
            index = tuple(int(row[key]) for key in ("i", "j", "k"))
            assert index not in modes
            modes[index] = complex(float(row["real"]), float(row["imag"])) / volume
    compressed = (shape[0] // 2 + 1, shape[1], shape[2])
    assert modes.keys() == set(itertools.product(*(range(n) for n in compressed)))
    # Exact Fourier series coefficients follow directly from Euler's identity.
    expected = {(0, 0, 0): 1.0, (1, 0, 0): 0.15}
    for k in (1, shape[2] - 1):
        expected[0, 2, k] = -0.05j
        expected[0, shape[1] - 2, k] = 0.05j
    for index, value in modes.items():
        assert abs(value - expected.get(index, 0)) < 2e-12, (index, value)
    parseval = sum(
        (1 if i == 0 or shape[0] % 2 == 0 and i == shape[0] // 2 else 2) * abs(v) ** 2
        for (i, j, k), v in modes.items()
    )
    real_square = sum(float(row["u"]) ** 2 for row in points.values()) / volume
    assert abs(parseval - 1.055) < 2e-12 and abs(real_square - parseval) < 2e-12
    summary = json.loads((folder / "results/summary.json").read_text())
    assert summary["ranks"] == ranks and summary["shape"] == shape
    assert abs(summary["mean"] - 1) < 2e-12
    assert abs(summary["mean_square"] - real_square) < 2e-12
    actual_max = max(abs(float(r["u"]) - float(r["initial"])) for r in points.values())
    assert abs(summary["max_error"] - actual_max) < 1e-15
    return points, modes


def check(binary, root, backend, launch, ranks):
    for name, shape, length, cycles in (
        ("default", [16, 18, 20], [2 * math.pi, 3 * math.pi, math.pi], 3),
        ("odd", [9, 11, 13], [2.1, 3.7, 1.3], 1),
        ("repeated", [8, 10, 12], [1.2, 4.3, 2.5], 7),
    ):
        folder = root / name
        run(
            binary,
            folder,
            launch,
            parameters={
                "backend": backend,
                "shape": shape,
                "length": length,
                "cycles": cycles,
            },
        )
        validate(folder, shape, length, ranks)
    for name, parameters in (
        ("bad-shape", {"shape": [2, 8, 8]}),
        ("bad-length", {"length": [1, 0, 1]}),
        ("bad-cycles", {"cycles": 0}),
        ("bad-backend", {"backend": "missing"}),
    ):
        run(binary, root / name, launch, parameters=parameters, success=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "hip"), default="cpu")
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()
    launch = (args.launcher, args.rank_flag, str(args.ranks)) if args.launcher else ()
    if args.ranks != 1 and not args.launcher:
        parser.error("multiple ranks require a launcher")
    check(
        args.bin_dir.resolve() / "fft_backend_benchmark",
        args.output.resolve() / f"backend-{args.backend}-{args.ranks}",
        args.backend,
        launch,
        args.ranks,
    )
    print(
        "FFT backend: complete physical field, sparse modes, normalization, Parseval and ownership pass."
    )


if __name__ == "__main__":
    main()
