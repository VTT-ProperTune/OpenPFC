#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Slurm batch driver for scripts/build_tohtori.sh (long OpenMPI / OpenPFC builds).
#
# Submit from the OpenPFC repo root, e.g.:
#   ./scripts/submit_build_tohtori_sbatch.sh
#   OPENPFC_REPO=/path/to/openpfc sbatch -p gen05_epyc scripts/sbatch_build_tohtori.slurm
#
# Required at run time:
#   OPENPFC_REPO  — absolute path to the OpenPFC clone (set by submit_build_tohtori_sbatch.sh)
#
# Optional:
#   BUILD_TOHTORI_EXTRA_ARGS — extra flags for build_tohtori.sh (e.g. --build-openmpi --clean-openmpi)

#SBATCH --job-name=openpfc-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=750G
#SBATCH --exclusive
#SBATCH --time=04:00:00

set -euo pipefail

: "${OPENPFC_REPO:?OPENPFC_REPO must be set to the OpenPFC repository root}"

cd "${OPENPFC_REPO}"

echo "[sbatch_build_tohtori] host=$(hostname) OPENPFC_REPO=${OPENPFC_REPO}"
echo "[sbatch_build_tohtori] BUILD_TOHTORI_EXTRA_ARGS=${BUILD_TOHTORI_EXTRA_ARGS:-<none>}"

# shellcheck disable=SC2086
exec bash "${OPENPFC_REPO}/scripts/build_tohtori.sh" ${BUILD_TOHTORI_EXTRA_ARGS:-}
