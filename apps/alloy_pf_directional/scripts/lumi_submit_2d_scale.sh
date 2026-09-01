#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Submit the 2D campaign (steps 0–2, or the large CPU ladder).
# Run on LUMI, or from the laptop via:
#   ./apps/alloy_pf_directional/scripts/lumi_submit_2d_scale.sh          # ssh + sbatch
#   ./apps/alloy_pf_directional/scripts/lumi_submit_2d_scale.sh --local  # already on LUMI
#   ./apps/alloy_pf_directional/scripts/lumi_submit_2d_scale.sh --cpu-large
#   ./apps/alloy_pf_directional/scripts/lumi_submit_2d_scale.sh --cpu-large --nodes-max=16
#
# Does not start 3D bricks or moving-window jobs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lumi_paths.sh
source "${HERE}/lumi_paths.sh"

REMOTE=1
SYNC=1
DO_CPU=1
DO_GPU=1
DO_CPU_LARGE=0
NODES_MAX="${ALCU_CPU_NODES_MAX:-64}"
EXTRA_REMOTE=()
for a in "$@"; do
  case "$a" in
    --local) REMOTE=0 ;;
    --no-sync) SYNC=0 ;;
    --cpu-only) DO_GPU=0; EXTRA_REMOTE+=(--cpu-only) ;;
    --gpu-only) DO_CPU=0; EXTRA_REMOTE+=(--gpu-only) ;;
    --cpu-large)
      DO_CPU_LARGE=1
      DO_CPU=0
      DO_GPU=0
      EXTRA_REMOTE+=(--cpu-large)
      ;;
    --nodes-max=*)
      NODES_MAX="${a#*=}"
      EXTRA_REMOTE+=("$a")
      ;;
    --help|-h)
      echo "usage: $0 [--local] [--no-sync] [--cpu-only] [--gpu-only] [--cpu-large] [--nodes-max=N]"
      exit 0
      ;;
  esac
done

submit_on_lumi() {
  local script="$1"
  shift
  mkdir -p "${LUMI_LOGS}" "${LUMI_SCALE2D}"
  sbatch --export=ALL "$@" "${script}"
}

if [[ "${REMOTE}" -eq 1 ]]; then
  if [[ "${SYNC}" -eq 1 ]]; then
    "${HERE}/sync_to_lumi.sh"
  fi
  echo "ssh ${LUMI_USER}@${LUMI_HOST} to submit"
  extra=""
  if ((${#EXTRA_REMOTE[@]})); then
    extra="${EXTRA_REMOTE[*]}"
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=20 "${LUMI_USER}@${LUMI_HOST}" \
    "cd '${LUMI_SRC}' && bash apps/alloy_pf_directional/scripts/lumi_submit_2d_scale.sh --local --no-sync ${extra}"
  exit $?
fi

CPU="${LUMI_SRC}/apps/alloy_pf_directional/scripts/lumi_scale_cpu.sh"
GPU="${LUMI_SRC}/apps/alloy_pf_directional/scripts/lumi_scale_gpu.sh"
CHK_CPU="${LUMI_SRC}/apps/alloy_pf_directional/scripts/lumi_step0_cpu.sh"
CHK_GPU="${LUMI_SRC}/apps/alloy_pf_directional/scripts/check_hip_vs_cpu.sh"
mkdir -p "${LUMI_LOGS}" "${LUMI_SCALE2D}/step0_cpu" "${LUMI_SCALE2D}/step0_gpu"

echo "=== submitting 2D campaign under ${LUMI_SCALE2D} ==="
ids=()

if [[ "${DO_CPU}" -eq 1 ]]; then
  # Step 0 CPU
  id=$(sbatch --parsable --job-name=alcu-s0-c \
    --partition=small --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:25:00 \
    --export=ALL,GRID=1280x160,STEPS=800 \
    "${CHK_CPU}")
  ids+=("cpu-step0:${id}")
  echo "submitted CPU step 0 job ${id}"

  # OpenMP thread sweep (one node, not exclusive 128-core)
  for th in 8 16 32; do
    id=$(sbatch --parsable --job-name="alcu-omp-${th}" \
      --partition=small --nodes=1 --ntasks=1 --cpus-per-task="${th}" --time=00:20:00 \
      --export=ALL,MODE=omp,GRID=1280x160,NTHREADS="${th}",WARMUP=10,TIMED=80 \
      "${CPU}")
    ids+=("omp-${th}:${id}")
  done

  # Strong MPI: 1 rank/core. 1280×160 on 1/2/4 small nodes (32 ranks/node).
  for spec in "1:32" "2:64" "4:128"; do
    nodes="${spec%%:*}"; ntasks="${spec##*:}"
    id=$(sbatch --parsable --job-name="alcu-s1280-n${nodes}" \
      --partition=small --nodes="${nodes}" --ntasks="${ntasks}" --ntasks-per-node=32 \
      --cpus-per-task=1 --time=00:20:00 \
      --export=ALL,MODE=strong,GRID=1280x160,WARMUP=10,TIMED=80 \
      "${CPU}")
    ids+=("strong-1280-n${nodes}:${id}")
  done
  # Strong 2560×320: 1 and 2 nodes
  for spec in "1:32" "2:64"; do
    nodes="${spec%%:*}"; ntasks="${spec##*:}"
    id=$(sbatch --parsable --job-name="alcu-s2560-n${nodes}" \
      --partition=small --nodes="${nodes}" --ntasks="${ntasks}" --ntasks-per-node=32 \
      --cpus-per-task=1 --time=00:25:00 \
      --export=ALL,MODE=strong,GRID=2560x320,WARMUP=10,TIMED=50 \
      "${CPU}")
    ids+=("strong-2560-n${nodes}:${id}")
  done
  # Weak: 1 and 2 nodes from 1280×160 / rank
  for spec in "1:32" "2:64"; do
    nodes="${spec%%:*}"; ntasks="${spec##*:}"
    id=$(sbatch --parsable --job-name="alcu-w-n${nodes}" \
      --partition=small --nodes="${nodes}" --ntasks="${ntasks}" --ntasks-per-node=32 \
      --cpus-per-task=1 --time=00:20:00 \
      --export=ALL,MODE=weak,GRID=1280x160,WARMUP=10,TIMED=50 \
      "${CPU}")
    ids+=("weak-n${nodes}:${id}")
  done
fi

if [[ "${DO_CPU_LARGE}" -eq 1 ]]; then
  # Large 2D CPU ladder on LUMI-C standard (whole nodes, 128 cores each).
  # MPI is 1 rank/core. Cap with --nodes-max (default 64; partition max is 512).
  echo "large CPU campaign GRID=20480x2560 and 3600x1280, nodes_max=${NODES_MAX}"
  if [[ ! -x "${LUMI_BIN}" || ! -x "${LUMI_BIN_MPI}" ]]; then
    echo "missing ${LUMI_BIN} or ${LUMI_BIN_MPI} — build with lumi_build_cpu.sh" >&2
    exit 1
  fi
  for th in 16 32 64 128; do
    id=$(sbatch --parsable --job-name="alcu-omp-L-${th}" \
      --partition=standard --nodes=1 --ntasks=1 --cpus-per-task="${th}" \
      --time=00:25:00 \
      --export=ALL,MODE=omp,GRID=20480x2560,NTHREADS="${th}",WARMUP=10,TIMED=40 \
      "${CPU}")
    ids+=("omp-large-${th}:${id}")
  done
  for th in 16 32 64 128; do
    id=$(sbatch --parsable --job-name="alcu-omp-P-${th}" \
      --partition=standard --nodes=1 --ntasks=1 --cpus-per-task="${th}" \
      --time=00:20:00 \
      --export=ALL,MODE=omp,GRID=3600x1280,NTHREADS="${th}",WARMUP=10,TIMED=50 \
      "${CPU}")
    ids+=("omp-prodbox-${th}:${id}")
  done
  for nodes in 1 2 4 8 16 32 64 128 256 512; do
    if [[ "${nodes}" -gt "${NODES_MAX}" ]]; then
      continue
    fi
    ntasks=$((nodes * 128))
    id=$(sbatch --parsable --job-name="alcu-s20k-n${nodes}" \
      --partition=standard --nodes="${nodes}" --ntasks="${ntasks}" \
      --ntasks-per-node=128 --cpus-per-task=1 --time=00:25:00 \
      --export=ALL,MODE=strong,GRID=20480x2560,WARMUP=10,TIMED=40 \
      "${CPU}")
    ids+=("strong-20480-n${nodes}:${id}")
  done
fi

if [[ "${DO_GPU}" -eq 1 ]]; then
  if [[ ! -x "${LUMI_BIN_HIP}" ]]; then
    echo "HIP binary missing (${LUMI_BIN_HIP}) — skip GPU submits."
    echo "Build with: bash apps/alloy_pf_directional/scripts/lumi_build_hip.sh configure && sbatch apps/alloy_pf_directional/scripts/lumi_build_hip.sh"
    ids+=("gpu:SKIPPED_NO_HIP_BINARY")
  else
    id=$(sbatch --parsable --job-name=alcu-s0-g \
      --export=ALL,GRID=1280x160,STEPS=800 \
      "${CHK_GPU}")
    ids+=("gpu-step0:${id}")
    echo "submitted GPU step 0 job ${id}"

    # Strong: 1 GCD, 8 GCDs (1 node), 16 GCDs (2 nodes)
    id=$(sbatch --parsable --job-name=alcu-g1 \
      --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=1 --time=00:20:00 \
      --export=ALL,MODE=strong,GRID=1280x160,WARMUP=10,TIMED=80 \
      "${GPU}")
    ids+=("gpu-strong-1:${id}")
    id=$(sbatch --parsable --job-name=alcu-g8 \
      --nodes=1 --ntasks=8 --ntasks-per-node=8 --gpus-per-node=8 --time=00:20:00 \
      --export=ALL,MODE=strong,GRID=1280x160,WARMUP=10,TIMED=80 \
      "${GPU}")
    ids+=("gpu-strong-8:${id}")
    id=$(sbatch --parsable --job-name=alcu-g16 \
      --nodes=2 --ntasks=16 --ntasks-per-node=8 --gpus-per-node=8 --time=00:25:00 \
      --export=ALL,MODE=strong,GRID=2560x320,WARMUP=10,TIMED=50 \
      "${GPU}")
    ids+=("gpu-strong-16:${id}")
    id=$(sbatch --parsable --job-name=alcu-gw8 \
      --nodes=1 --ntasks=8 --ntasks-per-node=8 --gpus-per-node=8 --time=00:20:00 \
      --export=ALL,MODE=weak,GRID=1280x160,WARMUP=10,TIMED=50 \
      "${GPU}")
    ids+=("gpu-weak-8:${id}")
  fi
fi

printf '%s\n' "${ids[@]}" | tee "${LUMI_SCALE2D}/submitted_jobs.txt"
echo "Job ids written to ${LUMI_SCALE2D}/submitted_jobs.txt"
echo "After jobs finish: python3 apps/alloy_pf_directional/scripts/analyze_2d_scaling.py ${LUMI_SCALE2D}"
