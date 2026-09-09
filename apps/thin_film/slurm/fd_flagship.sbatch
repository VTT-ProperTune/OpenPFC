#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --job-name=tf-fd-flagship
#SBATCH --output=/flash/project_462001519/juaho/tmp-shared/tf_fd_flagship_%j.out

# Flagship two-method comparison (#124): the conservative face-flux FD
# solver integrated through rupture, 512x512 dx=0.5 -- the same physical
# case as apps/thin_film/inputs_json/thin_film_dewetting.json /
# thin_film_defect.json, run far enough past the point where the spectral
# solver diverges to reach hole formation and rim/droplet coarsening.
#
# CPU-only MPI job (no HIP/GPU kernels in the FD path); submitted on
# standard-g per the shared-allocation policy for long/large runs.

set -euo pipefail

BUILD_DIR=/flash/project_462001519/juaho/build/tf-fd-rupture
BIN="${BUILD_DIR}/apps/thin_film/thin_film_fd"
CASES_DIR=/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc/.claude/worktrees/agent-a38076156c6ff0ac4/apps/thin_film/inputs_json

cd "${BUILD_DIR}/apps/thin_film"
mkdir -p results/fd_dewetting results/fd_defect

echo "=== spontaneous (FD) ==="
srun "${BIN}" "${CASES_DIR}/thin_film_fd_dewetting.json"

echo "=== defect-triggered (FD) ==="
srun "${BIN}" "${CASES_DIR}/thin_film_fd_defect.json"

echo "=== done ==="
