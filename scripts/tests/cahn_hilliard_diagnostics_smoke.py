#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Real MPI entrypoint checks: diagnostics without VTK, restart, and failures."""

import argparse
import copy
import csv
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--mpiexec", required=True)
    args = parser.parse_args()
    settings = {
        "model": {"name": "cahn_hilliard", "params": {}},
        "domain": {"Lx": 32, "Ly": 16, "Lz": 1, "dx": 1, "dy": 1, "dz": 1, "origin": "corner"},
        "timestepping": {"t0": 0, "t1": 0.04, "dt": 0.01, "saveat": 0.01},
        "initial_conditions": [{"type": "seeded_noise", "target": "c", "c0": 0.32, "amplitude": 0.02, "seed": 42}],
        "diagnostics": {"csv": "results/diagnostics.csv"},
        "checkpoint": {"every": 2, "directory": "results/checkpoints"},
    }
    with tempfile.TemporaryDirectory(prefix="openpfc-ch-diagnostics-") as temporary:
        root = Path(temporary)

        def run(name, config, ranks=2, success=True):
            case = root / name
            case.mkdir(exist_ok=True)
            input_file = case / "input.json"
            input_file.write_text(json.dumps(config))
            result = subprocess.run([args.mpiexec, "-n", str(ranks), args.binary, str(input_file)],
                                    cwd=case, capture_output=True, text=True, timeout=40)
            assert (result.returncode == 0) == success, result.stdout + result.stderr
            return case, result

        def rows(path):
            with path.open() as stream:
                return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]

        full, _ = run("full", settings, ranks=1)
        reference = rows(full / "results/diagnostics.csv")
        assert [r["step"] for r in reference] == [0, 1, 2, 3, 4]
        for first, second in zip(reference, reference[1:]):
            assert second["total_energy"] <= first["total_energy"] + 1e-10
            assert abs(second["mass"] - first["mass"]) < 1e-10
            assert second["invalid_cells"] == 0

        split_settings = copy.deepcopy(settings)
        split_settings["timestepping"]["t1"] = 0.02
        split, _ = run("split", split_settings)
        split_rows = rows(split / "results/diagnostics.csv")
        for a, b in zip(reference, split_rows):
            for key in a:
                assert abs(a[key] - b[key]) < 1e-10, (key, a, b)

        restart = copy.deepcopy(settings)
        restart["restart_from"] = str(split / "results/checkpoints/step_2")
        resumed, _ = run("resumed", restart)
        resumed_rows = rows(resumed / "results/diagnostics.csv")
        assert [r["step"] for r in resumed_rows] == [2, 3, 4]
        for a, b in zip(reference[2:], resumed_rows):
            for key in a:
                assert abs(a[key] - b[key]) < 1e-10, (key, a, b)
        def field(case):
            data = (case / "results/checkpoints/step_4/fields/c.bin").read_bytes()
            return struct.unpack("={}d".format(len(data) // 8), data)
        assert max(abs(a-b) for a, b in zip(field(full), field(resumed))) < 1e-12

        previous = (split / "results/diagnostics.csv").read_bytes()
        _, failed = run("split", split_settings, success=False)
        assert "cannot create fresh CSV" in failed.stderr + failed.stdout
        assert (split / "results/diagnostics.csv").read_bytes() == previous

        bad = copy.deepcopy(settings)
        bad["initial_conditions"] = [{"type": "cosine_mode", "target": "c", "c0": 0.32,
                                      "amplitude": 1, "nx": 1, "ny": 0, "nz": 0}]
        invalid, failed = run("invalid", bad, success=False)
        assert "out-of-range composition" in failed.stderr + failed.stdout
        assert rows(invalid / "results/diagnostics.csv")[0]["invalid_cells"] > 0


if __name__ == "__main__":
    main()
