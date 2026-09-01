#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Tiny Al-Cu FTA smoke on LUMI-C (debug queue, 8 cores, 15 min).
# Submit on LUMI:  sbatch apps/alloy_pf_directional/scripts/lumi_smoke.sh
#SBATCH --job-name=alcu-smoke
#SBATCH --account=project_462001519
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1750
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/alcu_fta/logs/smoke-%j.out
set -euo pipefail

# shellcheck source=lumi_paths.sh
source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_directional/scripts/lumi_paths.sh

module purge
module load "${LUMI_STACK}" partition/C cpeGNU cray-fftw lumi-CrayPath
lumi_cpu_runtime_env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export OPENPFC_ALCU_QUIET=1

OUT="${LUMI_RUNS}/smoke/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cd "${OUT}"

if [[ ! -x "${LUMI_BIN}" ]]; then
  echo "missing ${LUMI_BIN}" >&2
  exit 1
fi

srun --cpu-bind=cores --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  "${LUMI_BIN}" smoke "${SLURM_CPUS_PER_TASK}"
echo "smoke done in ${OUT}"
