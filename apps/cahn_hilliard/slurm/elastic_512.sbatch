#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --job-name=ch-el512
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/cahn-hilliard/el512_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/cahn_hilliard/cahn_hilliard_elastic"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
[[ -d "${HOME}/privatemodules" ]] && module use "${HOME}/privatemodules"
module load heffte-cpu || module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" --target cahn_hilliard_elastic

mkdir -p "${OUT}/results/fe_cr_elastic"
python3 - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${SRC}/apps/cahn_hilliard/inputs_json/fe_cr_elastic.json").read_text())
cfg["domain"]["Lx"] = 512
cfg["domain"]["Ly"] = 512
cfg["timestepping"]["t1"] = 40.0
cfg["timestepping"]["dt"] = 0.05
cfg["timestepping"]["saveat"] = 10.0
Path("${OUT}/case.json").write_text(json.dumps(cfg, indent=2)+"\n")
PY
cd "${OUT}"
srun --ntasks=1 --cpus-per-task=8 "${BIN}" "${OUT}/case.json" | tee "${OUT}/run.log"
tail -5 "${OUT}/results/fe_cr_elastic/diagnostics.csv"
