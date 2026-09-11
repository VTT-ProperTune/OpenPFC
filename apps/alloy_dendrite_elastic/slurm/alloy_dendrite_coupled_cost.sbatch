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
#SBATCH --time=02:00:00
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
# Every grid also runs with `--elastic=0` so the bare GPU step is measured in
# the same job on the same hardware, rather than inferred by subtraction.
#
# Provenance: the README's tables were filled in as this script was
# assembled, from jobs 21916676 (grids, warm/cold), 21916767 (tolerance and
# iteration ladders), 21917083 (eight GCDs) and 21917122 (lagging). Running
# this script reproduces all of them in one job id.
#
# OPENPFC_HIP_FORCE_PACKED_HALO=1 is set deliberately and is not a
# performance choice -- see "An intermittent 8-GCD fault" in the app README.
# With the GPU-aware device halo path *and* the host FFT solve both live on
# MPI_COMM_WORLD, eight-rank runs abort with a GPU memory fault in roughly a
# third of short runs. Forcing the packed halo path removes it. Unset it to
# reproduce the fault.
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
export OPENPFC_HIP_FORCE_PACKED_HALO="${OPENPFC_HIP_FORCE_PACKED_HALO:-1}"

MASK8="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"

echo "=== alloy_dendrite coupled cost, job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="
echo "bin=${ALLOY_COST_BIN} out=${OUT}"
echo "OPENPFC_HIP_FORCE_PACKED_HALO=${OPENPFC_HIP_FORCE_PACKED_HALO}"

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
# memory is the limit with sixteen double fields.
for n in 96 128 192; do
  one 1 "pf_only_${n}" --nx=$n --ny=$n --nz=$n --steps=30 --warmup=5 --elastic=0
  one 1 "warm_${n}"    --nx=$n --ny=$n --nz=$n --steps=30 --warmup=5 --warm-start=1
done
# Cold start only where it is affordable; the warm/cold ratio is the point
# and 192^3 cold is 30 x 36 s of host FFT for a number the pair already gives.
for n in 96 128; do
  one 1 "cold_${n}" --nx=$n --ny=$n --nz=$n --steps=30 --warmup=5 --warm-start=0
done

# --- what one Green application costs -------------------------------------
# The tolerance sweep says how many iterations a tolerance buys; the capped
# ladder prices one iteration directly, with tol_el = 0 so the count is
# exactly what is asked for and the fixed closing cost separates out.
for t in 1e-3 1e-4 1e-5 1e-6; do
  one 1 "tol_${t}" --nx=128 --ny=128 --nz=128 --steps=25 --warmup=5 --warm-start=1 --tol-el="${t}"
done
for it in 1 2 4 8; do
  one 1 "iter${it}_128" --nx=128 --ny=128 --nz=128 --steps=15 --warmup=3 \
      --warm-start=0 --n-el-iter="${it}" --tol-el=0
done

# --- lagging the quasi-static solve ---------------------------------------
for sub in 1 2 5 10; do
  one 1 "sub${sub}" --nx=128 --ny=128 --nz=128 --steps=40 --warmup=8 \
      --warm-start=1 --n-el-substep="${sub}"
done
for sub in 1 10; do
  one 1 "sub${sub}_tol3" --nx=128 --ny=128 --nz=128 --steps=40 --warmup=8 \
      --warm-start=1 --tol-el=1e-3 --n-el-substep="${sub}"
done

# --- liquid softness: the contrast the fixed point actually sees ----------
for ls in 0.01 0.1; do
  one 1 "mu${ls}" --nx=128 --ny=128 --nz=128 --steps=16 --warmup=4 \
      --warm-start=1 --liquid-shear="${ls}"
done

# --- the feedback switched on, so the cost is of a real coupled run -------
one 1 feedback_128 --nx=128 --ny=128 --nz=128 --steps=16 --warmup=4 \
    --warm-start=1 --lambda-el=1.0

# --- eight GCDs -----------------------------------------------------------
# 256^3 over 8 ranks is 128^3 per GCD, the middle of the single-GCD range
# above, so the two are directly comparable. OPENPFC_FFT_PROC_GRID forces the
# 1x1x8 z-slab: spectral_fft_proc_grid only returns a true slab at nine ranks
# or more (kSpectralSlabMinRanks), and at eight it hands back the same
# minimum-surface 2x2x2 brick, which halves the phase-field throughput here.
one 8 "pf_only_256_np8"       --nx=256 --ny=256 --nz=256 --steps=20 --warmup=4 --elastic=0
one 8 "pf_only_256_np8_brick" --nx=256 --ny=256 --nz=256 --steps=20 --warmup=4 \
    --elastic=0 --fd-grid=brick
one 8 "warm_128_np8"          --nx=128 --ny=128 --nz=128 --steps=12 --warmup=2 --warm-start=1
OPENPFC_FFT_PROC_GRID=1,1,8 one 8 "warm_256_np8_slab" \
    --nx=256 --ny=256 --nz=256 --steps=12 --warmup=2 --warm-start=1

echo "=== per-step CSV ==="
cat "${CSV}"
echo "=== done ==="
