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
#SBATCH --job-name=ch-elcamp
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Rebuild with composition projection, Catch2, then a 256^2 quartet:
# cubic / isotropic / none / suppressed.

set -euo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/cahn-hilliard/elcamp_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/cahn_hilliard/cahn_hilliard_elastic"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-cpu || module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

echo "=== CH elastic campaign ${SLURM_JOB_ID} $(hostname) ==="
cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" \
  --target cahn_hilliard_elastic test_cahn_hilliard
srun --ntasks=1 --cpus-per-task=4 \
  "${BUILD}/apps/cahn_hilliard/test_cahn_hilliard" "[cahn_hilliard][elastic]"

mkdir -p "${OUT}"
python3 - <<PY
import json, copy
from pathlib import Path
src = json.loads(Path("${SRC}/apps/cahn_hilliard/inputs_json/fe_cr_elastic.json").read_text())
src["domain"]["Lx"] = 256
src["domain"]["Ly"] = 256
src["timestepping"]["t1"] = 100.0
src["timestepping"]["dt"] = 0.05
src["timestepping"]["saveat"] = 20.0
src["diagnostics"]["csv"] = "results/fe_cr_elastic/diagnostics.csv"
src["fields"][0]["data"] = "results/fe_cr_elastic/c_%04d.vti"
out = Path("${OUT}")

def dump(name, mut):
    cfg = copy.deepcopy(src)
    mut(cfg)
    (out / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")

dump("cubic", lambda c: None)
dump("none", lambda c: c["elasticity"].__setitem__("eps0", 0.0))
def iso(c):
    e = c["elasticity"]
    e["eps0"] = 0.04
    e["c11"] = 0.0
    e["c12"] = 0.0
    e["c44"] = 0.0
    e["E"] = 2.0e11
    e["nu"] = 0.3
dump("isotropic", iso)
dump("strong", lambda c: c["elasticity"].__setitem__("eps0", 0.08))
print("wrote cases")
PY

for case in cubic none isotropic strong; do
  echo "=== ${case} ==="
  mkdir -p "${OUT}/${case}/results/fe_cr_elastic"
  cd "${OUT}/${case}"
  srun --ntasks=1 --cpus-per-task=8 \
    "${BIN}" "${OUT}/${case}.json" | tee "${OUT}/${case}.log"
done
echo "=== done ==="
for case in cubic none isotropic strong; do
  echo "--- ${case} ---"
  tail -1 "${OUT}/${case}/results/fe_cr_elastic/diagnostics.csv" || true
  tail -1 "${OUT}/${case}.log" || true
done
