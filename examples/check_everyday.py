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
    parser.add_argument("--case", choices=["first_domain"])
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
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
