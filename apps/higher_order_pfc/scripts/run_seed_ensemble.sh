#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Frozen eight-seed ensemble for Gate B.
# Seeds are declared before any run and are not discarded after inspection.

set -euo pipefail

SEEDS=(42 7 11 99 137 256 1024 2024)
CASES=(single_mode_triangular two_mode_square)
SRC="${SRC:-/pfs/lustref1/flash/project_462001519/juaho/dev/worktrees/openpfc/608-hopfc}"
BIN="${BIN:-/flash/project_462001519/juaho/build/openpfc-lumi-rocm-heffte-trace-heat3d/apps/higher_order_pfc/higher_order_pfc}"
OUT="${OUT:-$(pwd)}"

python3 - "$SRC" "$OUT" "${SEEDS[@]}" << 'PY'
import json, sys
from pathlib import Path
src = Path(sys.argv[1]) / "apps/higher_order_pfc/inputs_json"
out = Path(sys.argv[2])
seeds = [int(s) for s in sys.argv[3:]]
cases = ("single_mode_triangular", "two_mode_square")
out.mkdir(parents=True, exist_ok=True)
(out / "results/higher_order_pfc").mkdir(parents=True, exist_ok=True)
for case in cases:
    base = json.loads((src / f"{case}.json").read_text())
    for seed in seeds:
        cfg = json.loads(json.dumps(base))
        cfg["initial_conditions"][0]["seed"] = seed
        cfg["fields"][0]["data"] = f"results/higher_order_pfc/{case}_s{seed}_%04d.vti"
        cfg["diagnostics"]["csv"] = (
            f"results/higher_order_pfc/{case}_s{seed}_diagnostics.csv"
        )
        path = out / f"{case}_s{seed}.json"
        path.write_text(json.dumps(cfg, indent=4) + "\n")
        print(path)
PY

cd "${OUT}"
for case in "${CASES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== ${case} seed ${seed} ==="
    srun -n 1 "${BIN}" "${OUT}/${case}_s${seed}.json"
  done
done
