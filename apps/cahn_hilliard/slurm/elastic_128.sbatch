#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=ch-el128
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/cahn-hilliard/elastic_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/cahn_hilliard/cahn_hilliard_elastic"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-cpu || module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"

mkdir -p "${OUT}/elastic" "${OUT}/chemical"
python3 - <<PY
import json
from pathlib import Path
src = Path("${SRC}/apps/cahn_hilliard/inputs_json/fe_cr_elastic.json")
cfg = json.loads(src.read_text())
Path("${OUT}/elastic.json").write_text(json.dumps(cfg, indent=2) + "\n")
cfg["elasticity"]["eps0"] = 0.0
cfg["diagnostics"]["csv"] = "results/fe_cr_elastic/diagnostics.csv"
Path("${OUT}/chemical.json").write_text(json.dumps(cfg, indent=2) + "\n")
PY

echo "=== elastic 128^2 ==="
cd "${OUT}/elastic"
srun --ntasks=1 "${BIN}" "${OUT}/elastic.json" | tee "${OUT}/elastic.log"
echo "=== chemical (eps0=0) 128^2 ==="
cd "${OUT}/chemical"
srun --ntasks=1 "${BIN}" "${OUT}/chemical.json" | tee "${OUT}/chemical.log"
echo "=== done ==="
tail -3 "${OUT}/elastic/results/fe_cr_elastic/diagnostics.csv"
tail -3 "${OUT}/chemical/results/fe_cr_elastic/diagnostics.csv"
