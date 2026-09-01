#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LUMI-G 2D GPU scaling. One MPI rank per GCD, 8 GCDs per node.
# MPICH_GPU_SUPPORT_ENABLED=1. Packed fallback: OPENPFC_HIP_FORCE_PACKED_HALO=1
#
#   MODE=strong GRID=1280x160 sbatch --nodes=1 --ntasks=8 --gpus-per-node=8 ...
#   MODE=strong GRID=2560x320 sbatch --nodes=2 --ntasks=16 --gpus-per-node=8 ...
#   MODE=weak   GRID=1280x160 sbatch --nodes=1 --ntasks=8 ...
#
#SBATCH --job-name=alcu-2d-g
#SBATCH --account=project_462001519
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/alcu_fta/logs/scale2d-g-%j.out
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/lumi_paths.sh" ]]; then
  # shellcheck source=lumi_paths.sh
  source "${SCRIPT_DIR}/lumi_paths.sh"
else
  source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_directional/scripts/lumi_paths.sh
fi
# shellcheck source=alcu_2d_env.sh
source "${SCRIPT_DIR}/alcu_2d_env.sh"

MODE="${MODE:-strong}"
GRID="${GRID:-1280x160}"
WARMUP="${WARMUP:-10}"
TIMED="${TIMED:-80}"

module purge
module load "${LUMI_STACK}" partition/G cpeGNU cray-fftw lumi-CrayPath
lumi_gpu_runtime_env
export MPICH_GPU_SUPPORT_ENABLED=1

alcu_2d_apply_grid "${GRID}"
alcu_2d_apply_timing "${WARMUP}" "${TIMED}"

ntasks="${SLURM_NTASKS:-1}"
if [[ "${MODE}" == "weak" ]]; then
  local_ny_base=160
  case "${GRID}" in
    2560x320) local_ny_base=320 ;;
    5120x640) local_ny_base=640 ;;
  esac
  ny=$(( local_ny_base * ntasks ))
  export OPENPFC_ALCU_LY="$(python3 -c "print(${ny} * float('${OPENPFC_ALCU_W0}'))")"
fi

BIN="${LUMI_BIN_HIP}"
if [[ ! -x "${BIN}" ]]; then
  echo "missing ${BIN}" >&2
  echo "build HIP with apps/alloy_pf_directional/scripts/lumi_build_hip.sh" >&2
  exit 1
fi
TAG="${MODE}_${GRID}_g${ntasks}"
OUT="${LUMI_SCALE2D}/gpu/${SLURM_JOB_ID:-manual}_${TAG}"
mkdir -p "${OUT}" "${LUMI_LOGS}"
WRAP="${SCRIPT_DIR}/lumi_select_gpu.sh"
chmod +x "${WRAP}" || true

echo "ALCU_SCALE mode=${MODE} backend=hip nproc=${ntasks} grid=${GRID} Ly=${OPENPFC_ALCU_LY} packed=${OPENPFC_HIP_FORCE_PACKED_HALO:-0}"
echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
# LUMI-G rank↔GCD CPU map (same as docs/lumi_slurm/tungsten_gpu.sbatch)
srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 \
  "${WRAP}" "${BIN}" ds "${OUT}" | tee "${OUT}/run.log"
echo "done ${OUT}"
grep -E 'ALCU_PERF|ALCU_VERIFY|ALCU_SCALE' "${OUT}/run.log" || true
