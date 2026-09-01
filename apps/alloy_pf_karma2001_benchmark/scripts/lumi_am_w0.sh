#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Decaying AM cooling (τ=12 μs, Ṫ₀=1e7 K/s) at one W0 (nm), L=7 μm box
# (matches laptop tau12_L7e-6). Override with KARMA_W0, KARMA_OUT, KARMA_BIN,
# OPENPFC_KARMA_L. Submit via lumi_submit_am_w0.sh.
#
#   sbatch --export=ALL,KARMA_W0=10 apps/alloy_pf_karma2001_benchmark/scripts/lumi_am_w0.sh
#SBATCH --job-name=karma-am
#SBATCH --account=project_462001519
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1750
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/karma2001/logs/%x-%j.out
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/lumi_paths.sh" ]]; then
  # shellcheck source=lumi_paths.sh
  source "${SCRIPT_DIR}/lumi_paths.sh"
else
  # shellcheck source=lumi_paths.sh
  source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_karma2001_benchmark/scripts/lumi_paths.sh
fi

module purge
module load "${LUMI_STACK}" partition/C cpeGNU cray-fftw lumi-CrayPath
lumi_cpu_runtime_env

W0_NM="${KARMA_W0:-2.5}"
BIN="${KARMA_BIN:-${LUMI_KARMA_BIN}}"
OUT="${KARMA_OUT:-${LUMI_KARMA_RUNS}/tau12_L7e-6/W${W0_NM}nm_th45}"
mkdir -p "${OUT}" "${LUMI_KARMA_LOGS}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENPFC_KARMA_SKIP_PNG=1 OPENPFC_KARMA_QUIET=1
export OPENPFC_KARMA_TDOT=1e7 OPENPFC_KARMA_TEND=18e-6 OPENPFC_KARMA_TDECAY=12e-6
export OPENPFC_KARMA_L="${OPENPFC_KARMA_L:-7e-6}"
export OPENPFC_KARMA_STOP_FRAC="${OPENPFC_KARMA_STOP_FRAC:-0.80}"
export OPENPFC_KARMA_NCONTOUR="${OPENPFC_KARMA_NCONTOUR:-12}"
unset OPENPFC_KARMA_NOISE OPENPFC_KARMA_VD OPENPFC_KARMA_BETA0 || true

if [[ ! -x "${BIN}" ]]; then
  echo "missing ${BIN} — build with apps/alloy_pf_karma2001_benchmark/scripts/lumi_build_cpu.sh" >&2
  exit 1
fi
echo "AM W0=${W0_NM} nm  L=${OPENPFC_KARMA_L} m  out=${OUT}  threads=${OMP_NUM_THREADS}"
"${BIN}" am "${W0_NM}" 45 "${OUT}" "${OMP_NUM_THREADS}" | tee "${OUT}/run.log"
