#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=tf-512
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# 512^2 thin-film science dumps for the report money figure.
# Compile stays on standard-g (not the login node).

set -uo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-hip}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/thin-film/fig512_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/thin_film/thin_film_nonlinear_hip"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

mkdir -p "${OUT}"
echo "=== thin-film 512 HIP job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} $(hostname) ==="
module list

if [[ ! -f "${BUILD}/build.ninja" && ! -f "${BUILD}/Makefile" ]]; then
  echo "No HIP build tree at ${BUILD}"
  exit 2
fi
cd "${BUILD}"
cmake --build "${BUILD}" -j 8 --target thin_film_nonlinear_hip
ls -l "${BIN}"

python3 - <<PY
import json
from pathlib import Path
src = Path("${SRC}/apps/thin_film/inputs_json")
out = Path("${OUT}")

def with_fields(name, fields_path, csv_path):
    cfg = json.loads((src / name).read_text())
    cfg["fields"] = [{"name": "h", "data": fields_path}]
    cfg["diagnostics"]["csv"] = csv_path
    return cfg

spont = with_fields(
    "thin_film_dewetting.json",
    "results/dewetting/h_%04d.vti",
    "results/dewetting/diagnostics.csv",
)
defect = with_fields(
    "thin_film_defect.json",
    "results/defect/h_%04d.vti",
    "results/defect/diagnostics.csv",
)
(out / "spontaneous.json").write_text(json.dumps(spont, indent=2) + "\n")
(out / "defect.json").write_text(json.dumps(defect, indent=2) + "\n")
print("wrote", out / "spontaneous.json")
print("wrote", out / "defect.json")
PY

mkdir -p "${OUT}/spontaneous/results/dewetting" "${OUT}/defect/results/defect"

echo "=== spontaneous 512^2 ==="
cd "${OUT}/spontaneous"
/usr/bin/time -f 'elapsed_sec %e' srun --ntasks=1 --gpus-per-node=1 \
  "${BIN}" "${OUT}/spontaneous.json" | tee "${OUT}/spontaneous.log"

echo "=== defect 512^2 ==="
cd "${OUT}/defect"
/usr/bin/time -f 'elapsed_sec %e' srun --ntasks=1 --gpus-per-node=1 \
  "${BIN}" "${OUT}/defect.json" | tee "${OUT}/defect.log"

echo "=== done ==="
echo "OUT ${OUT}"
ls -lh "${OUT}/spontaneous/results/dewetting" | tail
ls -lh "${OUT}/defect/results/defect" | tail
tail -5 "${OUT}/spontaneous.log"
tail -5 "${OUT}/defect.log"
