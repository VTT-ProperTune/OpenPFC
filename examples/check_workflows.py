#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exercise native workflow models and verify fields, transactions and exported records."""
from __future__ import annotations

import argparse
import cmath
import csv
import io
import json
import math
from pathlib import Path
import tempfile

from check_foundations import run


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def finite(row, keys):
    assert all(math.isfinite(float(row[key])) for key in keys), row


def check_rk(binary, root, launch, ranks):
    errors = []
    for tolerance in (1e-4, 1e-7):
        directory = root / f"adaptive-{tolerance}"
        run(binary, directory, launch, {"tolerance": tolerance})
        ledgers = [
            rows(p) for p in sorted((directory / "results").glob("attempts_rank_*.csv"))
        ]
        assert len(ledgers) == ranks and all(ledger == ledgers[0] for ledger in ledgers)
        final_errors = []
        for controller in ("memoryless", "pi"):
            t, value, rejects, landed = 0.0, 1.0, 0, False
            for row in (r for r in ledgers[0] if r["controller"] == controller):
                finite(row, ("t0", "dt", "metric", "time", "y"))
                assert abs(float(row["t0"]) - t) < 1e-14
                dt, accepted = float(row["dt"]), int(row["accepted"])
                assert dt > 0 and int(row["rhs"]) == 4
                assert not (t < 0.15 - 1e-13 and t + dt > 0.15 + 1e-13)
                if accepted:
                    assert float(row["metric"]) <= 1
                    t += dt
                    value = float(row["y"])
                    landed |= abs(t - 0.15) < 1e-13
                else:
                    rejects += 1
                    assert float(row["metric"]) > 1
                    assert float(row["y"]) == value
                assert abs(float(row["time"]) - t) < 1e-14
                exact = math.exp(-50 * min(t, 0.15) - max(0, t - 0.15))
                assert abs(float(row["exact"]) - exact) < 2e-15
            assert abs(t - 0.5) < 1e-13 and rejects > 0 and landed
            final_errors.append(abs(value - math.exp(-7.85)))
        errors.append(final_errors)
    assert all(fine < coarse / 10 for coarse, fine in zip(*errors)), errors
    run(binary, root / "bad-rk", launch, {"tolerance": -1}, success=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument("--case", choices=["21_adaptive_stepping"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bin_dir = args.bin_dir.resolve()
    ranks = args.ranks
    launch = [args.launcher, args.rank_flag, str(ranks)] if args.launcher else []
    with tempfile.TemporaryDirectory(
        prefix="example-check-", dir=args.output
    ) as temporary:
        base = Path(temporary).resolve()
        if args.case in (None, "21_adaptive_stepping"):
            root = base / "21_adaptive_stepping"
            root.mkdir()
            check_rk(bin_dir / "21_adaptive_stepping", root, launch, ranks)
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
