#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Extra: isothermal trapping dendrite at one d0/W. KARMA_TRAP=1 (default here)
# uses magnified V_D and β₀; KARMA_TRAP=0 is the PRL present model (A=β0=εk=0).
# The advertised paper suite is lumi_paper.sh, not this script.
#
#   sbatch --export=ALL,KARMA_D0W=1.217,KARMA_TRAP=1,KARMA_DT=0.1,KARMA_BETA0=4 \
#     apps/alloy_pf_karma2001_benchmark/scripts/lumi_glasner_w0.sh
#SBATCH --job-name=karma-glasner
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

D0W="${KARMA_D0W:-1.217}"
TRAP="${KARMA_TRAP:-1}"
PHI1="${KARMA_PHI1:-45}"
BIN="${KARMA_BIN:-${LUMI_KARMA_BIN}}"
TAG=""
if [[ "${TRAP}" == "0" ]]; then
  TAG="_present"
  export OPENPFC_KARMA_VD=0 OPENPFC_KARMA_BETA0=0 OPENPFC_KARMA_EPSK=0
else
  TAG=""
  export OPENPFC_KARMA_VD="${KARMA_VD:-0.15}"
  export OPENPFC_KARMA_BETA0="${KARMA_BETA0:-4}"
  export OPENPFC_KARMA_EPSK="${KARMA_EPSK:-0.12}"
fi
OUT="${KARMA_OUT:-${LUMI_KARMA_RUNS}/glasner/d0W_${D0W}_th${PHI1}${TAG}}"
mkdir -p "${OUT}" "${LUMI_KARMA_LOGS}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENPFC_KARMA_SKIP_PNG=1 OPENPFC_KARMA_QUIET=1
if [[ -n "${KARMA_DT:-}" ]]; then
  export OPENPFC_KARMA_DT="${KARMA_DT}"
fi
unset OPENPFC_KARMA_NOISE || true

if [[ ! -x "${BIN}" ]]; then
  echo "missing ${BIN} — build with apps/alloy_pf_karma2001_benchmark/scripts/lumi_build_cpu.sh" >&2
  exit 1
fi
echo "glasner d0/W=${D0W} trap=${TRAP} dt/tau0=${OPENPFC_KARMA_DT:-0.02} beta0=${OPENPFC_KARMA_BETA0:-default} out=${OUT} threads=${OMP_NUM_THREADS}"
"${BIN}" glasner "${D0W}" "${PHI1}" "${OUT}" "${OMP_NUM_THREADS}" | tee "${OUT}/run.log"
