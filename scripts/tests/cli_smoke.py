#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CTest smoke: invoke the built CLI and run its scaffold with real tungsten."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--app", default="tungsten")
    args = parser.parse_args()
    cli = [sys.executable, args.cli]
    subprocess.run(cli + ["--version"], check=True)
    with tempfile.TemporaryDirectory(prefix="openpfc-cli-smoke-") as directory:
        case = Path(directory) / "case with spaces"
        subprocess.run(cli + ["init", str(case), "--app", args.app], check=True)
        if args.app != "tungsten":
            settings = json.loads((case / "input.json").read_text())
            # Preserve shipped spatial physics/modes but keep CI time bounded.
            settings["timestepping"].update(t1=0.02, dt=0.01, saveat=0.01)
            (case / "input.json").write_text(json.dumps(settings))
        subprocess.run(cli + ["run", str(case), "--launcher=none",
                              "--executable", args.binary], check=True)
        outputs = list((case / "results").rglob("*.vti"))
        assert len(outputs) == 3, outputs
        if args.app != "tungsten":
            if args.app == "cahn_hilliard":
                import csv
                with (case / "results/cahn_hilliard/diagnostics.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == 3
                assert abs(float(rows[0]["mean"]) - 0.32) < 1e-12
                assert float(rows[-1]["total_energy"]) < float(rows[0]["total_energy"])
                assert float(rows[-1]["invalid_cells"]) == 0
            return
        metadata = json.loads((case / "results/checkpoints/step_10/metadata.json").read_text())
        assert metadata["accepted_increment"] == 10
        assert metadata["result_counter"] == 3


if __name__ == "__main__":
    main()
