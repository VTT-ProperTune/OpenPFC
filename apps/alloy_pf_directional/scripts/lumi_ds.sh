#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# One directional-solidification case on LUMI-C (OpenMP, shared node).
# Do NOT submit without an explicit go-ahead for a production case.
#
#   CASE=w0_5nm_dx0.4 W0=5e-9 DXW=0.4 SAVE_EVERY=20000 \
#     sbatch --job-name=alcu-5nm-dx04 --time=08:00:00 \
#       apps/alloy_pf_directional/scripts/lumi_ds.sh
#
# Defaults below are a 16-core, 4 h cap — still ask before submitting.
#SBATCH --job-name=alcu-ds
#SBATCH --account=project_462001519
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/alcu_fta/logs/ds-%x-%j.out
set -euo pipefail

# shellcheck source=lumi_paths.sh
source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_directional/scripts/lumi_paths.sh

CASE="${CASE:-w0_5nm_dx1}"
W0="${W0:-5e-9}"
DXW="${DXW:-1.0}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
LOG_EVERY="${LOG_EVERY:-}"

module purge
module load "${LUMI_STACK}" partition/C cpeGNU cray-fftw lumi-CrayPath
lumi_cpu_runtime_env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export OPENPFC_ALCU_QUIET=1
export OPENPFC_ALCU_W0="${W0}"
export OPENPFC_ALCU_DXW="${DXW}"
export OPENPFC_ALCU_LX="${OPENPFC_ALCU_LX:-12.0e-6}"
export OPENPFC_ALCU_LY="${OPENPFC_ALCU_LY:-0.80e-6}"
export OPENPFC_ALCU_TEND="${OPENPFC_ALCU_TEND:-120.0e-6}"
export OPENPFC_ALCU_SEED="${OPENPFC_ALCU_SEED:-0.20e-6}"
export OPENPFC_ALCU_NGRANS="${OPENPFC_ALCU_NGRANS:-1}"
export OPENPFC_ALCU_THETA="${OPENPFC_ALCU_THETA:-30}"
export OPENPFC_ALCU_NOISE="${OPENPFC_ALCU_NOISE:-0}"
export OPENPFC_ALCU_WINDOW="${OPENPFC_ALCU_WINDOW:-0}"
export OPENPFC_ALCU_STOP_FAR_C="${OPENPFC_ALCU_STOP_FAR_C:-1}"
export OPENPFC_ALCU_STOP_RIGHT="${OPENPFC_ALCU_STOP_RIGHT:-0}"
export OPENPFC_ALCU_PERIODIC_Y="${OPENPFC_ALCU_PERIODIC_Y:-1}"
export OPENPFC_ALCU_SKIP_VTK="${OPENPFC_ALCU_SKIP_VTK:-1}"
export OPENPFC_ALCU_SKIP_PNG="${OPENPFC_ALCU_SKIP_PNG:-1}"
export OPENPFC_ALCU_G="${OPENPFC_ALCU_G:-3.0e6}"
export OPENPFC_ALCU_VP="${OPENPFC_ALCU_VP:-0.4}"
export OPENPFC_ALCU_DT_OVER_TAU="${OPENPFC_ALCU_DT_OVER_TAU:-0.1}"

if [[ "${OPENPFC_ALCU_NGRANS}" == "2" ]]; then
  OUT="${LUMI_RUNS}/bicrystal/${CASE}"
else
  OUT="${LUMI_RUNS}/ds/${CASE}"
fi
mkdir -p "${OUT}"
cd "${OUT}"

if [[ ! -x "${LUMI_BIN}" ]]; then
  echo "missing ${LUMI_BIN}" >&2
  exit 1
fi

extra=()
if [[ -n "${SAVE_EVERY}" ]]; then
  extra+=(--save-every "${SAVE_EVERY}")
fi
if [[ -n "${LOG_EVERY}" ]]; then
  extra+=(--log-every "${LOG_EVERY}")
fi

srun --cpu-bind=cores --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  "${LUMI_BIN}" ds "${OUT}" "${SLURM_CPUS_PER_TASK}" "${extra[@]}"
echo "ds ${CASE} done in ${OUT}"
