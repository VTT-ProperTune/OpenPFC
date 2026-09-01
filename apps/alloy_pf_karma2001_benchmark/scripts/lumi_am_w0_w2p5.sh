#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# W0=2.5 nm AM cooling (τ=12 μs) — too large for a laptop (~30× the 5 nm case).
# CPU OpenMP on LUMI-C shared nodes (not exclusive, not a full 128-core node).
#
#   sbatch apps/alloy_pf_karma2001_benchmark/scripts/lumi_am_w0_w2p5.sh
#
# SPDX-SnippetBegin
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-SnippetEnd
#SBATCH --job-name=karma-w2.5
#SBATCH --account=project_462001519
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/project_462001519/%u/karma2001/logs/%x-%j.out

set -euo pipefail
module load LUMI/25.09
# Build with scripts/build.sh (CPU) so this binary exists:
BIN="${KARMA_BIN:-${HOME}/projappl/build/openpfc-cpu/apps/alloy_pf_karma2001_benchmark/alloy_pf_karma2001_benchmark_openmp}"
OUT="${KARMA_OUT:-/scratch/project_462001519/${USER}/karma2001/tau12/W2.5nm_th45}"
mkdir -p "${OUT}" "$(dirname "${BIN}")" /scratch/project_462001519/"${USER}"/karma2001/logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENPFC_KARMA_SKIP_PNG=1 OPENPFC_KARMA_QUIET=1
export OPENPFC_KARMA_TDOT=1e7 OPENPFC_KARMA_TEND=18e-6 OPENPFC_KARMA_TDECAY=12e-6
export OPENPFC_KARMA_L=3.5e-6 OPENPFC_KARMA_STOP_FRAC=0.80 OPENPFC_KARMA_NCONTOUR=12
unset OPENPFC_KARMA_NOISE || true

if [[ ! -x "${BIN}" ]]; then
  echo "missing ${BIN} — build OpenPFC CPU with scripts/build.sh first" >&2
  exit 1
fi
"${BIN}" am 2.5 45 "${OUT}" "${OMP_NUM_THREADS}" | tee "${OUT}/run.log"
