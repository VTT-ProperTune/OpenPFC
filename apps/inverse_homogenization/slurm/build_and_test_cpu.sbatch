#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=invhom-cpu
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Compile and run the homogenization Catch2 suite on a LUMI-C node.
# Configure (FetchContent) already ran on the login node. Do not compile
# on the login node.

set -euo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc
export CXX=CC
export PATH="${HOME}/.local/bin:${PATH}"

echo "=== invhom CPU build+test job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="
echo "host $(hostname)  build ${BUILD}"
module list
echo "CC=$(command -v CC)"

cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" \
  --target test_homogenization test_inverse_homogenization \
           openpfc_homogenize openpfc_inverse_homogenize

echo "=== ctest homogenization ==="
cd "${BUILD}"
ctest --output-on-failure --timeout 300 -R homogenization
echo "=== CLI smoke ==="
srun --ntasks=1 --cpus-per-task=1 \
  "${BUILD}/apps/inverse_homogenization/openpfc_homogenize" \
  --shape=homogeneous --nx=8 --ny=8 --nz=8 --volume=1
echo "=== done ==="
