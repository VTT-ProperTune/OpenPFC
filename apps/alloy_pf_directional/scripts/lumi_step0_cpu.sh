#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LUMI-C step 0: OpenMP vs MPI np=1 on the 1280×160 (or GRID=) 2D DS brick.
#
#   sbatch apps/alloy_pf_directional/scripts/lumi_step0_cpu.sh
#
#SBATCH --job-name=alcu-s0-c
#SBATCH --account=project_462001519
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/alcu_fta/logs/step0-c-%j.out
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/lumi_paths.sh" ]]; then
  # shellcheck source=lumi_paths.sh
  source "${SCRIPT_DIR}/lumi_paths.sh"
else
  source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_directional/scripts/lumi_paths.sh
fi
module purge
module load "${LUMI_STACK}" partition/C cpeGNU cray-fftw lumi-CrayPath
lumi_cpu_runtime_env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_PROC_BIND=close OMP_PLACES=cores

ROOT="${LUMI_SRC}"
OUT="${LUMI_SCALE2D}/step0_cpu"
mkdir -p "${OUT}" "${LUMI_LOGS}"
export BUILD="${LUMI_BUILD}"
export OPENMP_BIN="${LUMI_BIN}"
export MPI_BIN="${LUMI_BIN_MPI}"
export OUT GRID="${GRID:-1280x160}" STEPS="${STEPS:-800}" NTHREADS="${OMP_NUM_THREADS}"
cd "${ROOT}"
bash "${SCRIPT_DIR}/check_nz1_vs_2d.sh"
# Stable path for the HIP compare
cp -f "${OUT}/openmp.log" "${LUMI_SCALE2D}/step0_cpu/openmp.log"
echo "step 0 CPU logs in ${OUT}"
