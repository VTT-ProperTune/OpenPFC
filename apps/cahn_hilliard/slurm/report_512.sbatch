#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:40:00
#SBATCH --job-name=ch-512
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# 512^2 Cahn–Hilliard dumps for the report money figure. Uses the existing
# HIP binary — no compile on the login node.

set -uo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-hip}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/cahn-hilliard/fig512_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/cahn_hilliard/cahn_hilliard_hip"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"

mkdir -p "${OUT}"
echo "=== CH 512 HIP job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} $(hostname) ==="
echo "BIN ${BIN}"
ls -l "${BIN}"
module list

if [[ ! -x "${BIN}" ]]; then
  echo "Missing HIP binary ${BIN}"
  exit 2
fi

# Science coarsening: shipped 512^2, c0=0.5, t in [0, 4000].
mkdir -p "${OUT}/science/results/fe_cr_coarsening"
echo "=== science 512^2 c0=0.5 ==="
cd "${OUT}/science"
/usr/bin/time -f 'elapsed_sec %e' srun --ntasks=1 --gpus-per-node=1 \
  "${BIN}" "${SRC}/apps/cahn_hilliard/inputs_json/fe_cr_coarsening.json" \
  | tee "${OUT}/science.log"

# Verification broadband at 512^2 (same physics as coarsening.json, bigger box).
mkdir -p "${OUT}/broadband/results/cahn_hilliard"
python3 - <<PY
import json
from pathlib import Path
src = Path("${SRC}/apps/cahn_hilliard/inputs_json/coarsening.json")
cfg = json.loads(src.read_text())
cfg["domain"]["Lx"] = 512
cfg["domain"]["Ly"] = 512
Path("${OUT}/broadband/coarsening_512.json").write_text(json.dumps(cfg, indent=2) + "\n")
PY
echo "=== broadband 512^2 c0=0.32 ==="
cd "${OUT}/broadband"
/usr/bin/time -f 'elapsed_sec %e' srun --ntasks=1 --gpus-per-node=1 \
  "${BIN}" "${OUT}/broadband/coarsening_512.json" \
  | tee "${OUT}/broadband.log"

# Single-mode 512^2, same wavelength as fe_cr_spinodal.json (nx,ny x4).
mkdir -p "${OUT}/single/results/cahn_hilliard"
python3 - <<PY
import json
from pathlib import Path
src = Path("${SRC}/apps/cahn_hilliard/inputs_json/fe_cr_spinodal.json")
cfg = json.loads(src.read_text())
cfg["domain"]["Lx"] = 512
cfg["domain"]["Ly"] = 512
cfg["initial_conditions"][0]["nx"] = 32
cfg["initial_conditions"][0]["ny"] = 24
cfg["timestepping"]["t1"] = 10.0
cfg["timestepping"]["dt"] = 0.05
cfg["timestepping"]["saveat"] = 2.0
Path("${OUT}/single/spinodal_512.json").write_text(json.dumps(cfg, indent=2) + "\n")
PY
echo "=== single-mode 512^2 ==="
cd "${OUT}/single"
/usr/bin/time -f 'elapsed_sec %e' srun --ntasks=1 --gpus-per-node=1 \
  "${BIN}" "${OUT}/single/spinodal_512.json" \
  | tee "${OUT}/single.log"

echo "=== done ==="
echo "OUT ${OUT}"
ls -lh "${OUT}/science/results/fe_cr_coarsening" | tail
ls -lh "${OUT}/broadband/results/cahn_hilliard" | tail
ls -lh "${OUT}/single/results/cahn_hilliard" | tail
