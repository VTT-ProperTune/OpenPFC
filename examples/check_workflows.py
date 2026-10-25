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


def rank_rows(directory, prefix):
    return [
        row
        for path in sorted((directory / "results").glob(prefix + "_rank_*.csv"))
        for row in rows(path)
    ]


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


def reaction_reference(n=16, end=0.2):
    """Direct complex DFT Laplacian plus small-step RK4, independent of ETD."""
    phase = [[cmath.exp(2j * math.pi * k * j / n) for j in range(n)] for k in range(n)]
    wave = [k if k <= n // 2 else k - n for k in range(n)]

    def rhs(u):
        coefficients = [
            sum(u[j] * phase[k][j].conjugate() for j in range(n)) / n for k in range(n)
        ]
        return [
            sum(-wave[k] ** 2 * coefficients[k] * phase[k][j] for k in range(n)).real
            - u[j] ** 3
            for j in range(n)
        ]

    u = [0.4 * math.cos(2 * math.pi * j / n) for j in range(n)]
    h = end / 1000
    for _ in range(1000):
        k1 = rhs(u)
        k2 = rhs([v + h * a / 2 for v, a in zip(u, k1)])
        k3 = rhs([v + h * a / 2 for v, a in zip(u, k2)])
        k4 = rhs([v + h * a for v, a in zip(u, k3)])
        u = [
            v + h * (a + 2 * b + 2 * c + d) / 6
            for v, a, b, c, d in zip(u, k1, k2, k3, k4)
        ]
    return u


def check_etd(binary, directory, launch, ranks):
    run(binary, directory, launch)
    ledgers = [
        rows(p) for p in sorted((directory / "results").glob("attempts_rank_*.csv"))
    ]
    assert len(ledgers) == ranks
    first = ledgers[0]
    uniform = 0.4
    rejections = 0
    for name in ("heat", "uniform", "reaction"):
        t = 0.0
        for row in (r for r in first if r["case"] == name):
            finite(row, ("t0", "dt", "metric", "time", "rollback_error"))
            assert abs(float(row["t0"]) - t) < 1e-14
            assert int(row["transforms"]) == 9 and int(row["nonlinear"]) == 3
            assert float(row["rollback_error"]) == 0
            if int(row["accepted"]):
                h = float(row["dt"])
                t += h
                assert float(row["metric"]) <= 1
                if name == "uniform":
                    uniform -= 0.5 * h * uniform**3
                    uniform -= 0.5 * h * uniform**3
            else:
                rejections += 1
                assert float(row["metric"]) > 1
            assert abs(float(row["time"]) - t) < 1e-14
        assert abs(t - 0.2) < 1e-13
    assert rejections > 0
    reference = reaction_reference()
    fields = rank_rows(directory, "fields")
    assert len(fields) == 3 * 16 * 4 * 4
    assert len({(r["case"], r["i"], r["j"], r["k"]) for r in fields}) == len(fields)
    for row in fields:
        x, value = float(row["x"]), float(row["u"])
        finite(row, ("x", "u"))
        if row["case"] == "heat":
            assert abs(value - 0.4 * math.exp(-0.2) * math.cos(x)) < 2e-13
        elif row["case"] == "uniform":
            assert abs(value - uniform) < 3e-13
            assert abs(value - 0.4 / math.sqrt(1 + 2 * 0.4**2 * 0.2)) < 1e-4
        else:
            assert abs(value - reference[int(row["i"])]) < 1e-4
    # Decisions must be rank-consistent; reduction roundoff can change only metric digits.
    for ledger in ledgers[1:]:
        assert [(r["case"], r["accepted"], r["time"], r["dt"]) for r in ledger] == [
            (r["case"], r["accepted"], r["time"], r["dt"]) for r in first
        ]
    run(binary, directory.parent / "bad-etd", launch, {"n": 7}, success=False)


def check_clock(binary, directory):
    run(binary, directory)
    data = rows(directory / "results/clock.csv")
    assert data[0]["event"] == "initial" and data[1]["event"] == "rejected"
    assert float(data[1]["time"]) == 0 and int(data[1]["increment"]) == 0
    accepted = [r for r in data if r["event"] == "accepted"]
    expected = [0.13, 0.2, 0.33, 0.4, 0.53, 0.6, 0.73, 0.8, 0.93, 1.0]
    assert len(accepted) == len(expected)
    for i, (row, t) in enumerate(zip(accepted, expected), 1):
        assert abs(float(row["time"]) - t) < 1e-13 and int(row["increment"]) == i
        assert int(row["save"]) == int(i % 2 == 0)
    run(binary, directory.parent / "bad-clock", parameters={"dt": 0}, success=False)


def check_host(binary, directory, launch):
    run(binary, directory, launch)
    expected = [0.1, 0.15, 0.25, 0.3, 0.4]
    base_sum = sum(
        math.sin(0.25 * i) * math.cos(0.5 * j)
        for i in range(8)
        for j in range(8)
        for _ in range(2)
    )
    for path in (directory / "results").glob("coupling_rank_*.csv"):
        ledger = rows(path)
        assert len(ledger) == 5
        previous = 0
        for i, (row, t) in enumerate(zip(ledger, expected)):
            assert abs(float(row["time"]) - t) < 1e-13
            assert abs(float(row["dt"]) - (t - previous)) < 1e-13
            assert (
                abs(
                    float(row["read_sum"])
                    - (0 if i == 0 else base_sum + 128 * 0.1 * previous)
                )
                < 1e-11
            )
            previous = t
    data = rank_rows(directory, "field")
    assert len(data) == 128
    for row in data:
        x, y, z, value = (float(row[key]) for key in ("x", "y", "z", "u"))
        assert abs(value - (math.sin(x) * math.cos(y) + 0.04)) < 1e-14


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
            "21_adaptive_stepping",
            "24_adaptive_etd",
            "time",
            "22_external_coupling",
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
        if args.case in (None, "21_adaptive_stepping"):
            root = base / "21_adaptive_stepping"
            root.mkdir()
            check_rk(bin_dir / "21_adaptive_stepping", root, launch, ranks)
        if args.case in (None, "24_adaptive_etd"):
            root = base / "24_adaptive_etd"
            root.mkdir()
            check_etd(bin_dir / "24_adaptive_etd", root / "etd", launch, ranks)
        if args.case in (None, "time"):
            root = base / "time"
            root.mkdir()
            check_clock(bin_dir / "time", root / "clock")
        if args.case in (None, "22_external_coupling"):
            root = base / "22_external_coupling"
            root.mkdir()
            check_host(bin_dir / "22_external_coupling", root / "host", launch)
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
