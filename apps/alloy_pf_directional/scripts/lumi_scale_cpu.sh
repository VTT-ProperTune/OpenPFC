#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LUMI-C 2D CPU scaling (Nz=1). One allocation = one (MODE, GRID, ranks/threads).
# I/O off. Prints ALCU_PERF. Does not start 3D or moving-window work.
#
#   MODE=omp  GRID=1280x160 NTHREADS=16 sbatch --ntasks=1 --cpus-per-task=16 ...
#   MODE=strong GRID=20480x2560 sbatch --partition=standard --nodes=4 \
#       --ntasks=512 --ntasks-per-node=128 --cpus-per-task=1 ...
#   MODE=weak  GRID=1280x160  sbatch --ntasks=16 --cpus-per-task=1 ...
# MPI ranks are serial (OMP_NUM_THREADS=1). Do not request cpus-per-task>1.
#
# Prefer apps/alloy_pf_directional/scripts/lumi_submit_2d_scale.sh to submit the sweep.
#
#SBATCH --job-name=alcu-2d-c
#SBATCH --account=project_462001519
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/alcu_fta/logs/scale2d-c-%j.out
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
NTHREADS="${NTHREADS:-${SLURM_CPUS_PER_TASK:-8}}"
WARMUP="${WARMUP:-10}"
TIMED="${TIMED:-80}"

module purge
module load "${LUMI_STACK}" partition/C cpeGNU cray-fftw lumi-CrayPath
lumi_cpu_runtime_env
export OMP_PROC_BIND=close OMP_PLACES=cores
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-${NTHREADS}}"

alcu_2d_apply_grid "${GRID}"
alcu_2d_apply_timing "${WARMUP}" "${TIMED}"

ntasks="${SLURM_NTASKS:-1}"
# Weak: keep ~1280×160 cells/rank by growing Ny with ntasks (periodic y).
if [[ "${MODE}" == "weak" ]]; then
  local_ny_base=160
  case "${GRID}" in
    2560x320) local_ny_base=320 ;;
    5120x640) local_ny_base=640 ;;
    3600x1280) local_ny_base=1280 ;;
    7200x2560|20480x2560) local_ny_base=2560 ;;
    10240x1280) local_ny_base=1280 ;;
  esac
  ny=$(( local_ny_base * ntasks ))
  export OPENPFC_ALCU_LY="$(python3 -c "print(${ny} * float('${OPENPFC_ALCU_W0}'))")"
fi

TAG="${MODE}_${GRID}_n${ntasks}_t${NTHREADS}"
OUT="${LUMI_SCALE2D}/cpu/${SLURM_JOB_ID:-manual}_${TAG}"
mkdir -p "${OUT}" "${LUMI_LOGS}"
echo "ALCU_SCALE mode=${MODE} backend=cpu grid=${GRID} ntasks=${ntasks} nthreads=${NTHREADS} NxLy=${OPENPFC_ALCU_LX}x${OPENPFC_ALCU_LY} warmup=${WARMUP} timed=${TIMED}"

if [[ "${MODE}" == "omp" ]]; then
  export OMP_NUM_THREADS="${NTHREADS}"
  BIN="${LUMI_BIN}"
  if [[ ! -x "${BIN}" ]]; then
    echo "missing ${BIN}" >&2
    exit 1
  fi
  srun --ntasks=1 --cpus-per-task="${NTHREADS}" --cpu-bind=cores \
    "${BIN}" ds "${OUT}" "${NTHREADS}" | tee "${OUT}/run.log"
else
  BIN="${LUMI_BIN_MPI}"
  export OMP_NUM_THREADS=1
  if [[ "${SLURM_CPUS_PER_TASK:-1}" -gt 1 ]]; then
    echo "warning: MPI engine is not OpenMP-parallel; cpus-per-task=${SLURM_CPUS_PER_TASK} wastes cores. Use --cpus-per-task=1." >&2
  fi
  if [[ ! -x "${BIN}" ]]; then
    echo "missing ${BIN} (build alloy_pf_directional_mpi via lumi_build_cpu.sh)" >&2
    exit 1
  fi
  srun --cpu-bind=cores "${BIN}" ds "${OUT}" | tee "${OUT}/run.log"
fi
echo "done ${OUT}"
grep -E 'ALCU_PERF|ALCU_VERIFY|ALCU_SCALE' "${OUT}/run.log" || true
