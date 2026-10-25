#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the halo bandwidth table's byte counts and rates without asserting a speed."""

import argparse
import math
import os
from pathlib import Path
import subprocess

COLUMNS = "N Send_B Time_us Uni_GBps Duplex_GBps RemoteFaces SelfFaces OffNodeFaces"


def parse(text):
    header = {}
    rows = []
    for line in text.splitlines():
        if line.startswith("# ranks "):
            words = line[2:].split()
            header = dict(zip(words[::2], words[1::2]))
        elif line.startswith("# N "):
            assert line[2:] == COLUMNS
        elif line and not line.startswith("#"):
            rows.append(line.split())
    return header, rows


def expected_traffic(n, grid, axes, halo):
    """Bytes one rank sends to other ranks, and its remote and self faces."""
    local = [n // g for g in grid]
    remote_bytes = remote = own = 0
    for axis in axes:
        face = halo * 8
        for other in range(3):
            if other != axis:
                face *= local[other]
        # Periodic in every axis: an undivided axis wraps to the same rank.
        if grid[axis] == 1:
            own += 2
        else:
            remote += 2
            remote_bytes += 2 * face
    return remote_bytes, remote, own


def validate(text, ranks, sizes, grid, axes, halo):
    header, rows = parse(text)
    assert int(header["ranks"]) == ranks
    printed = [int(g) for g in header["grid"].split("x")]
    assert math.prod(printed) == ranks
    # Without an override, the automatic grid is whatever the program reports.
    grid = grid or printed
    assert printed == grid
    assert int(header["halo"]) == halo
    assert [int(row[0]) for row in rows] == sizes
    for row in rows:
        # A table row has no send_min field when every rank sends the same bytes.
        assert len(row) == 8
        n, send, remote, own, offnode = (int(row[i]) for i in (0, 1, 5, 6, 7))
        time_us, uni, duplex = (float(row[i]) for i in (2, 3, 4))
        assert (send, remote, own) == expected_traffic(n, grid, axes, halo)
        assert offnode == 0  # Single-node check run.
        assert math.isfinite(time_us) and time_us > 0
        # Recompute the rate from the rounded time; allow the printed rounding.
        low = send / ((time_us + 0.05) * 1.0e3)
        high = send / (max(time_us - 0.05, 1.0e-3) * 1.0e3)
        assert low - 0.0005 <= uni <= high + 0.0005
        assert abs(duplex - 2.0 * uni) <= 0.0015


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", type=int, default=2)
    args = parser.parse_args()
    assert args.ranks == 2  # The cases below name two-rank process grids.
    sizes = [16, 32]
    # Most cases select a process grid, so the remote and self faces are known.
    # The automatic case derives them from the grid the program prints.
    for name, grid, flags, axes, halo in (
        ("automatic-z", None, ["--axis", "z"], [2], 1),
        ("x-split", [2, 1, 1], [], [0], 1),
        ("y-pair-undivided", [2, 1, 1], ["--axis", "y"], [1], 1),
        ("z-split", [1, 1, 2], ["--axis", "z"], [2], 1),
        ("all-faces-wide", [2, 1, 1], ["--faces", "--halo", "2"], [0, 1, 2], 2),
    ):
        folder = args.output.resolve() / name
        folder.mkdir(parents=True, exist_ok=True)
        command = [
            args.launcher,
            args.rank_flag,
            str(args.ranks),
            str(args.binary.resolve()),
            "--sizes",
            ",".join(map(str, sizes)),
            "--iters",
            "4",
            "--warmup",
            "1",
            *flags,
        ]
        env = dict(os.environ)
        env.pop("OPENPFC_FD_PROC_GRID", None)
        if grid:
            env["OPENPFC_FD_PROC_GRID"] = ",".join(map(str, grid))
        run = subprocess.run(
            command, cwd=folder, env=env, check=True, timeout=45, capture_output=True, text=True
        )
        (folder / "table.txt").write_text(run.stdout)
        validate(run.stdout, args.ranks, sizes, grid, axes, halo)
    print("Halo bandwidth: grids, face byte counts, face classes and rate arithmetic pass.")


if __name__ == "__main__":
    main()
