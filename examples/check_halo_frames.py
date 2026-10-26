#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the real halo profiler's per-rank export without asserting a speed."""

import argparse
import json
import math
from pathlib import Path
import subprocess


def validate(path, ranks, iterations):
    report = json.loads(path.read_text())
    assert report["schema_version"] == 2
    assert report["n_mpi_ranks"] == ranks
    assert report["total_frames"] == ranks * iterations
    names = report["frame_metric_names"]
    assert len(names) == len(set(names))
    assert {"step", "mpi_rank", "wall_step"} <= set(names)
    assert {"communication", "fft", "gradient"} <= set(report["region_paths"])
    entries = report["ranks"]
    assert {r["mpi_rank"] for r in entries} == set(range(ranks))
    assert len(entries) == ranks
    for rank in entries:
        assert rank["n_frames"] == len(rank["frames"]) == iterations
        for step, frame in enumerate(rank["frames"]):
            assert len(frame["scalars"]) == len(names)
            scalars = dict(zip(names, frame["scalars"]))
            assert all(math.isfinite(x) for x in scalars.values())
            assert scalars["mpi_rank"] == rank["mpi_rank"]
            assert scalars["step"] == step and scalars["wall_step"] >= 0
            for region in frame["regions"].values():
                assert math.isfinite(region["inclusive"]) and region["inclusive"] >= 0
                assert math.isfinite(region["exclusive"]) and region["exclusive"] >= 0
            assert frame["regions"]["fft"]["inclusive"] == 0
            assert frame["regions"]["gradient"]["inclusive"] == 0
            for key in ("rss_bytes", "model_heap_bytes", "heap_secondary_bytes"):
                assert (
                    scalars[key] == 0
                )  # Program deliberately does not measure memory.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", type=int, default=2)
    args = parser.parse_args()
    # Retain the previous two-rank 16^3 / four-frame / one-warmup smoke case.
    # Extra warmup and frame cases distinguish those two counts independently.
    for name, warmup, iterations, connectivity in (
        ("original", 1, 4, []),
        ("more-warmup", 3, 4, []),
        ("more-frames-full", 1, 6, ["--full"]),
    ):
        folder = args.output.resolve() / name
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "profile.json"
        output.unlink(missing_ok=True)
        command = [
            args.launcher,
            args.rank_flag,
            str(args.ranks),
            str(args.binary.resolve()),
            "--nx",
            "16",
            "--iters",
            str(iterations),
            "--warmup",
            str(warmup),
            "--output",
            str(output),
            *connectivity,
        ]
        subprocess.run(command, cwd=folder, check=True, timeout=45)
        validate(output, args.ranks, iterations)
        placement = (folder / "halo_placement.txt").read_text().splitlines()
        assert placement[0] == "rank host gpu rx ry rz offnode_faces"
        assert {int(row.split()[0]) for row in placement[1:]} == set(range(args.ranks))
        assert len(placement) == args.ranks + 1
    print(
        "Halo frames: rank identities, warmup exclusion, frame counts and finite durations pass."
    )


if __name__ == "__main__":
    main()
