#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# The parity measurement behind the HIP twin of equations (1)-(4).
#
# Two things are measured and neither can be done on a login node, which is
# why PR #147 shipped without a GPU path at all:
#
#   1. CPU against GPU, per rank, at 1 / 2 / 4 / 8 GCDs. Both steppers run in
#      the same process on the same decomposition from the same initial
#      condition, so the reported L-inf is the port's drift and nothing else.
#   2. 1 rank against N ranks on the GPU. The 1-rank run writes the gathered
#      global fields with `--dump`; the 2-, 4- and 8-rank runs read them back
#      with `--compare`. Gathering into *global* index order is what makes
#      that comparison about the halo exchange rather than about which rank
#      happened to own which cell.
#
# Both are run in 3-D (the `z` halo group and the `n_z^4` anisotropy branch
# only exist there) and once in 2-D, because `Axes2D()` skipping the `±z`
# faces is exactly the kind of thing that is right on one path and wrong on
# the other.
#
# Required in the job environment:
#   ALLOY_PARITY_BIN   absolute path to alloy_dendrite_hip_parity
# Optional:
#   ALLOY_N            cubic grid for the 3-D cases (default 96)
#   ALLOY_STEPS        steps per case (default 200)

set -euo pipefail

: "${ALLOY_PARITY_BIN:?set ALLOY_PARITY_BIN to the alloy_dendrite_hip_parity binary}"

N="${ALLOY_N:-96}"
STEPS="${ALLOY_STEPS:-200}"
OUT="${ALLOY_OUT:-/scratch/project_462001519/juaho/alloy-dendrite/parity_${SLURM_JOB_ID}}"
mkdir -p "${OUT}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
# Leave every GCD visible: the binary calls bind_local_device(), which picks
# local_rank % n_devices, and masking here breaks GPU-aware intra-node IPC.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true

# LUMI-G full-node CPU mask, 7 cores per GCD.
MASK8="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"

echo "=== alloy_dendrite HIP parity, job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="
echo "bin=${ALLOY_PARITY_BIN} N=${N} steps=${STEPS} out=${OUT}"
rocm-smi --showdriverversion 2>/dev/null | head -4 || true

CASE3D=(--nx="${N}" --ny="${N}" --nz="${N}" --dx=0.8 --fd-order=4 --steps="${STEPS}")
REF="${OUT}/ref_1rank.bin"

echo "--- 3-D, 1 rank: CPU vs GPU, and the reference dump ---"
srun --ntasks=1 --gpus-per-node=1 --cpu-bind=none \
  "${ALLOY_PARITY_BIN}" "${CASE3D[@]}" --run-id=np1 --dump="${REF}"

for np in 2 4 8; do
  echo "--- 3-D, ${np} ranks: CPU vs GPU per rank, and 1-vs-${np} on the GPU ---"
  if (( np == 8 )); then
    srun --ntasks=8 --gpus-per-node=8 --cpu-bind="${MASK8}" \
      "${ALLOY_PARITY_BIN}" "${CASE3D[@]}" --run-id="np${np}" --compare="${REF}"
  else
    srun --ntasks="${np}" --gpus-per-node="${np}" --cpu-bind=none \
      "${ALLOY_PARITY_BIN}" "${CASE3D[@]}" --run-id="np${np}" --compare="${REF}"
  fi
done

echo "--- 2-D slab, 1 and 4 ranks: the Axes2D() path ---"
CASE2D=(--nx=256 --ny=256 --nz=1 --dx=0.8 --fd-order=4 --steps="${STEPS}")
REF2="${OUT}/ref2d_1rank.bin"
srun --ntasks=1 --gpus-per-node=1 --cpu-bind=none \
  "${ALLOY_PARITY_BIN}" "${CASE2D[@]}" --run-id=np1-2d --dump="${REF2}"
srun --ntasks=4 --gpus-per-node=4 --cpu-bind=none \
  "${ALLOY_PARITY_BIN}" "${CASE2D[@]}" --run-id=np4-2d --compare="${REF2}"

echo "--- FD order sweep at 1 rank: the stencil payload, orders 2..8 ---"
for o in 2 6 8; do
  srun --ntasks=1 --gpus-per-node=1 --cpu-bind=none \
    "${ALLOY_PARITY_BIN}" --nx=64 --ny=64 --nz=64 --dx=0.8 --fd-order="${o}" \
    --steps="${STEPS}" --run-id="order${o}"
done

echo "--- drift with run length at 1 rank: is the CPU/GPU gap growing? ---"
for s in 50 200 800 3200; do
  srun --ntasks=1 --gpus-per-node=1 --cpu-bind=none \
    "${ALLOY_PARITY_BIN}" --nx=64 --ny=64 --nz=64 --dx=0.8 --fd-order=4 \
    --steps="${s}" --run-id="drift${s}"
done

echo "=== done ==="
