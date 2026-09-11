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
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# What coupling equations (5)-(7) to a GPU phase field actually costs.
#
# The route decision -- device port of the tensor solve, or host round-trip --
# turns on three ratios, and this job measures all three on grids the
# application would really use rather than on a 32^3 unit-test cube:
#
#   t_roundtrip / t_pf   is moving the fields across the bus affordable?
#   t_el / t_pf          is the host Eyre-Milton solve affordable?
#   iterations(warm)     does warm-starting in a time loop change the answer?
#
# Every case also runs with `--elastic=0` so the bare GPU step is measured in
# the same job on the same hardware, rather than inferred by subtraction.
#
# Required in the job environment:
#   ALLOY_COST_BIN   absolute path to alloy_dendrite_coupled_cost

set -euo pipefail

: "${ALLOY_COST_BIN:?set ALLOY_COST_BIN to the alloy_dendrite_coupled_cost binary}"

OUT="${ALLOY_OUT:-/scratch/project_462001519/juaho/alloy-dendrite/cost_${SLURM_JOB_ID}}"
mkdir -p "${OUT}"
CSV="${OUT}/coupled_cost.csv"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true

MASK8="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"

echo "=== alloy_dendrite coupled cost, job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="
echo "bin=${ALLOY_COST_BIN} out=${OUT}"

one() { # one <ntasks> <run-id> <args...>
  local np="$1"; shift
  local id="$1"; shift
  local bind=(--cpu-bind=none)
  if (( np == 8 )); then bind=(--cpu-bind="${MASK8}"); fi
  srun --ntasks="${np}" --gpus-per-node="${np}" "${bind[@]}" \
    "${ALLOY_COST_BIN}" --csv="${CSV}" --run-id="${id}" "$@"
}

# --- single GCD, three grid sizes -----------------------------------------
# 96^3 / 128^3 / 192^3 per GCD is the range a 3-D dendrite run would weak-scale
# at on this machine: below that the GPU is latency-bound, above it one GCD's
# 64 GB is the limit with sixteen double fields.
for n in 96 128 192; do
  one 1 "pf_only_${n}"   --nx=$n --ny=$n --nz=$n --steps=30 --warmup=5 --elastic=0
  one 1 "warm_${n}"      --nx=$n --ny=$n --nz=$n --steps=30 --warmup=5 --warm-start=1
  one 1 "cold_${n}"      --nx=$n --ny=$n --nz=$n --steps=30 --warmup=5 --warm-start=0
done

# --- lagging the quasi-static solve ---------------------------------------
for sub in 2 5 10; do
  one 1 "warm_128_sub${sub}" --nx=128 --ny=128 --nz=128 --steps=60 --warmup=10 \
      --warm-start=1 --n-el-substep="${sub}"
done

# --- softer liquid: the contrast the iteration count is most sensitive to --
for ls in 0.01 0.1; do
  one 1 "warm_128_mu${ls}" --nx=128 --ny=128 --nz=128 --steps=30 --warmup=5 \
      --warm-start=1 --liquid-shear="${ls}"
done

# --- eight GCDs: does the FFT's slab decomposition change the picture? -----
one 8 "pf_only_256_np8" --nx=256 --ny=256 --nz=256 --steps=30 --warmup=5 --elastic=0
one 8 "warm_256_np8"    --nx=256 --ny=256 --nz=256 --steps=30 --warmup=5 --warm-start=1
one 8 "cold_256_np8"    --nx=256 --ny=256 --nz=256 --steps=30 --warmup=5 --warm-start=0
# The FD-optimal brick decomposition, phase field only: the price of having
# forced the FD side onto the FFT's slab so the round-trip needs no MPI.
one 8 "pf_only_256_np8_brick" --nx=256 --ny=256 --nz=256 --steps=30 --warmup=5 \
    --elastic=0 --fd-grid=brick

# --- the coupling actually switched on, so the cost is of a real run ------
one 1 "feedback_128" --nx=128 --ny=128 --nz=128 --steps=30 --warmup=5 \
    --warm-start=1 --lambda-el=1.0

echo "=== per-step CSV ==="
cat "${CSV}"
echo "=== done ==="
