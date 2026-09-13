#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# One LUMI-C job for research issue ahojukka5/research#237: freeze the
# three linear waves against the in-tree dispersion solver, plus the
# Weibel k=3 no-growth check. Not HIP parity (job 21943612). Not a
# k-scan. Do not import chapter-17 tables without this job's IDs.
#
# debug is 30 minutes; three 64^3 linear runs plus k=3 belong on
# standard.
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=vm-oracle
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-heat3d-cpu}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/results/vlasov_gamma_oracle}"
BIN="${BUILD}/apps/vlasov_maxwell/vlasov_run"
HEFFTE_LIB="/flash/project_462001519/juaho/build/heffte-2.3.0-milan/lib"
export LD_LIBRARY_PATH="${HEFFTE_LIB}:${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# This vlasov_run has DT_RPATH to gcc 11.2, which beats LD_LIBRARY_PATH
# and lacks GLIBCXX_3.4.32. Jobs 22012539 and 22012544 died on that.
# Preload the system C++ runtime (has 3.4.32+).
export LD_PRELOAD="/usr/lib64/libstdc++.so.6:/lib64/libgcc_s.so.1"
export CRAYPE_LINK_TYPE=dynamic MPICH_GPU_SUPPORT_ENABLED=0 OMP_NUM_THREADS=16
module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${HEFFTE_LIB}:${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="/usr/lib64/libstdc++.so.6:/lib64/libgcc_s.so.1"

test -x "${BIN}"
mkdir -p "${OUT}"
SUMMARY="${OUT}/summary-${SLURM_JOB_ID}.csv"
TWOPI="6.283185307179586"
# k d_e = 1 and 3 with mode=1: Lx = 2 pi / k
LX_K1="${TWOPI}"
LX_K3="2.0943951023931953"

echo "=== vlasov gamma-vs-oracle job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="
echo "host $(hostname)"
echo "binary ${BIN}"
echo "revision $(git -C "${SRC}" rev-parse HEAD)"
echo "date $(date -Iseconds)"

run_one() {
  local tag="$1"; shift
  echo "=== ${tag} ==="
  srun --ntasks=8 --cpus-per-task=16 --cpu-bind=cores --export=ALL \
    "${BIN}" --nx=64 --nvx=64 --nvy=64 --device=host \
    --run-id="${tag}" --summary="${SUMMARY}" \
    --csv="${OUT}/series-${tag}-${SLURM_JOB_ID}.csv" \
    "$@"
}

# Landau: default Lx gives k lambda_D = 0.5
run_one landau-kld0.5 --case=landau
run_one twostream --case=twostream
run_one weibel-kde1 --case=weibel --Lx="${LX_K1}"
run_one weibel-kde3 --case=weibel --Lx="${LX_K3}"
echo "=== done ==="
echo "summary ${SUMMARY}"
