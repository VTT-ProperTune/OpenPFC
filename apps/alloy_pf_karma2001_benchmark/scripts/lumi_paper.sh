#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LUMI-C job: Karma 2001 present-model 3-case suite (t*=10^4, L/d0=1000).
#   sbatch apps/alloy_pf_karma2001_benchmark/scripts/lumi_paper.sh
#   QUICK=1 sbatch .../lumi_paper.sh    # short t* check on the cluster
#
#SBATCH --job-name=karma-paper
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
  SRC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
else
  # shellcheck source=lumi_paths.sh
  source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_karma2001_benchmark/scripts/lumi_paths.sh
  SRC_ROOT="${LUMI_KARMA_SRC}"
fi

module purge
module load "${LUMI_STACK}" partition/C cpeGNU cray-fftw lumi-CrayPath
lumi_cpu_runtime_env

BIN="${KARMA_BIN:-${LUMI_KARMA_BIN}}"
OUTROOT="${KARMA_OUT:-${LUMI_KARMA_RUNS}/paper}"
mkdir -p "${OUTROOT}" "${LUMI_KARMA_LOGS}"

if [[ ! -x "${BIN}" ]]; then
  echo "missing ${BIN} — build with apps/alloy_pf_karma2001_benchmark/scripts/lumi_build_cpu.sh" >&2
  exit 1
fi

export BIN OUTROOT
export NTHREADS="${SLURM_CPUS_PER_TASK:-32}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
echo "Karma 2001 paper suite  BIN=${BIN}  OUTROOT=${OUTROOT}  threads=${NTHREADS}  QUICK=${QUICK:-0}"
bash "${SRC_ROOT}/apps/alloy_pf_karma2001_benchmark/scripts/run_karma2001_benchmark.sh"
