#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Collect HEAT3D_CPU_WALL_STEP_MS_MEDIAN lines into a method-cost CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MEDIAN_RE = re.compile(r"HEAT3D_CPU_WALL_STEP_MS_MEDIAN=([0-9.eE+-]+)")
TAG_RE = re.compile(r"(spectral|fd2|fd8|fd12)-(\d+)\.log$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log_dir", type=Path)
    p.add_argument("jobid")
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()

    rows: list[tuple[str, int, str, float]] = []
    for path in sorted(args.log_dir.glob(f"*-{args.jobid}.log")):
        m = TAG_RE.search(path.name)
        if not m:
            continue
        tag, job = m.group(1), m.group(2)
        text = path.read_text(errors="replace")
        med = MEDIAN_RE.search(text)
        if not med:
            raise SystemExit(f"no median in {path}")
        method = "spectral" if tag == "spectral" else "fd"
        order = 0 if tag == "spectral" else int(tag[2:])
        rows.append((method, order, job, float(med.group(1))))

    if len(rows) != 4:
        raise SystemExit(f"expected 4 logs, got {len(rows)}")

    lines = [
        "# Cost per step at equal grid, heat3d on LUMI-C, N=1024, 1 node,",
        "# 8 MPI ranks x 16 OpenMP threads, 25 accepted steps after 5 warm-up,",
        "# dt=0.01. wall_step_ms is the median of barriered steps.",
        "method,fd_order,job,wall_step_ms",
    ]
    for method, order, job, ms in rows:
        lines.append(f"{method},{order},{job},{ms}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
