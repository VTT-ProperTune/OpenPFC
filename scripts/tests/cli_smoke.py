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
    args = parser.parse_args()
    cli = [sys.executable, args.cli]
    subprocess.run(cli + ["--version"], check=True)
    with tempfile.TemporaryDirectory(prefix="openpfc-cli-smoke-") as directory:
        case = Path(directory) / "case with spaces"
        subprocess.run(cli + ["init", str(case)], check=True)
        subprocess.run(cli + ["run", str(case), "--launcher=none",
                              "--executable", args.binary], check=True)
        outputs = list((case / "results").glob("*.vti"))
        assert len(outputs) == 3, outputs
        metadata = json.loads((case / "results/checkpoints/step_10/metadata.json").read_text())
        assert metadata["accepted_increment"] == 10
        assert metadata["result_counter"] == 3


if __name__ == "__main__":
    main()
