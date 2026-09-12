#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:50:00
#SBATCH --job-name=tf-fd-rupt
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Conservative FD through rupture. 256^2, same physics as the 512^2 science
# spectral case, integrated past the point where ETD diverges.

set -euo pipefail
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/thin-film/fd_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/thin_film/thin_film_fd"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
[[ -d "${HOME}/privatemodules" ]] && module use "${HOME}/privatemodules"
module load heffte-cpu || module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

echo "=== thin_film_fd rupture ${SLURM_JOB_ID} $(hostname) ==="
cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" --target thin_film_fd

mkdir -p "${OUT}"
python3 - <<PY
import json, copy
from pathlib import Path
src = Path("${SRC}/apps/thin_film/inputs_json")
out = Path("${OUT}")

def prep(name, dest, t1, defect):
    cfg = json.loads((src / name).read_text())
    cfg["domain"]["Lx"] = 256
    cfg["domain"]["Ly"] = 256
    cfg["timestepping"]["t1"] = t1
    cfg["timestepping"]["dt"] = 0.001
    cfg["timestepping"]["saveat"] = 10.0
    cfg["initial_conditions"]["defect_amplitude"] = defect
    cfg["fields"] = [{"name": "h", "data": "results/h_%04d.vti"}]
    cfg["diagnostics"]["csv"] = "results/diagnostics.csv"
    (out / dest).write_text(json.dumps(cfg, indent=2) + "\n")

prep("thin_film_fd_dewetting.json", "spontaneous.json", 400.0, 0.0)
prep("thin_film_fd_dewetting.json", "defect.json", 280.0, -0.30)
print("wrote cases")
PY

for case in spontaneous defect; do
  echo "=== ${case} ==="
  mkdir -p "${OUT}/${case}/results"
  cd "${OUT}/${case}"
  srun --ntasks=8 --cpus-per-task=4 \
    "${BIN}" "${OUT}/${case}.json" | tee "${OUT}/${case}.log"
  echo "--- ${case} csv tail ---"
  tail -5 "${OUT}/${case}/results/diagnostics.csv" || true
  ls "${OUT}/${case}/results" | tail
done
echo "=== done ==="
