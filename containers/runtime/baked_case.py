#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Entrypoint for a baked CPU case; MPI is supplied by the host."""

import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    if args == ["--provenance"]:
        print((ROOT / "provenance.json").read_text(), end="")
        return
    if len(args) == 2 and args[0] == "init":
        destination = Path(args[1]).resolve()
        shutil.copytree(ROOT / "case", destination)
        print("Created baked case: " + str(destination))
        return
    if args == ["--version"]:
        os.execv(str(ROOT / "bin/openpfc"), ["openpfc", "--version"])
    if len(args) == 2 and args[0] == "run":
        # The host mpirun/srun launches this wrapper once per rank. There is no
        # MPI launcher inside the runtime image, including for singleton runs.
        os.execv(str(ROOT / "bin/openpfc"), ["openpfc", "run", args[1],
                 "--launcher=none", "--executable", str(ROOT / "bin/tungsten")])
    raise ValueError("usage: image {--version | --provenance | init CASE | run CASE}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print("openpfc image: " + str(error), file=sys.stderr)
        sys.exit(2)
