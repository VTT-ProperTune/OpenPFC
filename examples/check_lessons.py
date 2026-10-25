#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the teaching executables and check fields against independent mode formulas."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile



def run(command: list[str], cwd: Path, success: bool = True) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=90)
    if (result.returncode == 0) != success:
        raise AssertionError(f"{command}\n{result.stdout}\n{result.stderr}")


def check_differentiation(binary, root, launch):
    directory = root / "differentiation"
    directory.mkdir()
    run([*launch, str(binary)], directory)
    rows = list(csv.DictReader((directory / "results/convergence.csv").open()))
    assert [int(row["n"]) for row in rows] == [8, 16, 32, 64, 128]
    for key, lower, upper in (("fd2", 3.5, 4.5), ("fd4", 13.0, 19.0)):
        errors = [float(row[key]) for row in rows]
        assert all(math.isfinite(error) and error > 0.0 for error in errors)
        assert lower < errors[-2] / errors[-1] < upper, (key, errors)
    assert float(rows[0]["fourier"]) > 1e-4
    assert max(float(row["fourier"]) for row in rows[2:]) < 2e-11



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument("--case", choices=['periodic_differentiation'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bin_dir = args.bin_dir.resolve()
    ranks = args.ranks
    launch = [args.launcher, args.rank_flag, str(ranks)] if args.launcher else []
    with tempfile.TemporaryDirectory(prefix="example-check-", dir=args.output) as temporary:
        base = Path(temporary).resolve()
        if args.case in (None, "periodic_differentiation"):
            root = base / "periodic_differentiation"
            root.mkdir()
            check_differentiation(bin_dir / "periodic_differentiation", root, launch)
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
