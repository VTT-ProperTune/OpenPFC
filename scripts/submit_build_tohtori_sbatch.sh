#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Submit scripts/sbatch_build_tohtori.slurm with sane defaults.
#
# Usage:
#   ./scripts/submit_build_tohtori_sbatch.sh
#   SBATCH_PARTITION=gen05_epyc SBATCH_TIME=08:00:00 ./scripts/submit_build_tohtori_sbatch.sh
#   BUILD_TOHTORI_EXTRA_ARGS="--build-openmpi --clean-openmpi" ./scripts/submit_build_tohtori_sbatch.sh
#
# Environment:
#   OPENPFC_REPO              (default: parent of scripts/ = repo root)
#   SBATCH_PARTITION          (if set, passed as -p; else no -p — site default partition)
#   SBATCH_TIME               (default: 04:00:00)
#   SBATCH_CPUS_PER_TASK      (default: 192)
#   SBATCH_MEM                (default: 750G)
#   SBATCH_EXCLUSIVE          (default: 1; set 0 to omit --exclusive)
#   SBATCH_ACCOUNT            (optional: passed as --account=...)
#   BUILD_TOHTORI_EXTRA_ARGS  (optional: forwarded to build_tohtori.sh)

if [ -z "${BASH_VERSION-}" ]; then
  exec /usr/bin/env bash "$0" ${1+"$@"}
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLURM_JOB_SCRIPT="${SCRIPT_DIR}/sbatch_build_tohtori.slurm"

OPENPFC_REPO="${OPENPFC_REPO:-${REPO_ROOT}}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-192}"
SBATCH_MEM="${SBATCH_MEM:-750G}"
SBATCH_EXCLUSIVE="${SBATCH_EXCLUSIVE:-1}"

if [[ ! -f "${SLURM_JOB_SCRIPT}" ]]; then
  echo "ERROR: missing ${SLURM_JOB_SCRIPT}" >&2
  exit 1
fi

sbatch_args=(
  --job-name=openpfc-build
  --nodes=1
  --ntasks=1
  --cpus-per-task="${SBATCH_CPUS_PER_TASK}"
  --mem="${SBATCH_MEM}"
  --time="${SBATCH_TIME}"
  --chdir="${OPENPFC_REPO}"
)

if [[ -n "${SBATCH_PARTITION:-}" ]]; then
  sbatch_args+=(--partition="${SBATCH_PARTITION}")
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  sbatch_args+=(--account="${SBATCH_ACCOUNT}")
fi
if [[ "${SBATCH_EXCLUSIVE}" != "0" ]]; then
  sbatch_args+=(--exclusive)
fi

export OPENPFC_REPO
export BUILD_TOHTORI_EXTRA_ARGS

echo "Submitting: sbatch ${sbatch_args[*]} --export=ALL,OPENPFC_REPO,BUILD_TOHTORI_EXTRA_ARGS ${SLURM_JOB_SCRIPT}"
exec sbatch "${sbatch_args[@]}" \
  --export=ALL,OPENPFC_REPO,BUILD_TOHTORI_EXTRA_ARGS \
  "${SLURM_JOB_SCRIPT}"
