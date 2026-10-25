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


def verify_06_multi_index(bin_dir, root, launch, ranks):
    binary = lambda name: bin_dir / name
    offsets = run(binary("06_multi_index"), root / "boxes").splitlines()
    assert [int(line.split(",")[3]) for line in offsets] == list(range(8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument("--case", choices=["02_domain_decomposition", "06_multi_index"])
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
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
