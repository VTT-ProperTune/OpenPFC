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
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
