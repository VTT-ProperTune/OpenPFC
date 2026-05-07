#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Strong scaling for apps/kobayashi/kobayashi_fd_hip on AMDGPU nodes: sweep MPI rank counts
# (default **1** and **2**) so each rank binds one GPU via `local_rank % hipGetDeviceCount()`.
#
# Requires a HIP-enabled build (`kobayashi_rebuild_openpfc_amdgpu.sbatch`).
#
#   export OPENPFC_BUILD_DIR=/path/to/build   # contains apps/kobayashi/kobayashi_fd_hip
#   sbatch apps/kobayashi/slurm/kobayashi_hip_scaling_amdgpu.sbatch
#
# Override rank list:  export KOBAYASHI_SCALING_NPROC="1 2"

#SBATCH --job-name=kob-hip-scale
#SBATCH --partition=amdgpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=kobayashi_hip_scaling_%j.out
#SBATCH --error=kobayashi_hip_scaling_%j.err

set -euo pipefail

if [[ -z "${OPENPFC_BUILD_DIR:-}" ]]; then
  echo "ERROR: set OPENPFC_BUILD_DIR to your HIP CMake build tree." >&2
  exit 1
fi

BIN="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_hip"
if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: missing executable: ${BIN}" >&2
  exit 1
fi

GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
ROCM_MODULE="${ROCM_MODULE:-rocm}"

module purge 2>/dev/null || true
module load "${GCC_MODULE}"
module load "${OPENMPI_MODULE}"
module load "${ROCM_MODULE}" 2>/dev/null || echo "WARN: ROCm module ${ROCM_MODULE} not loaded (may still work if ROCm is on PATH)." >&2
hash -r

if [[ "${KOBAYASHI_OMPI_KEEP_OFI:-0}" != "1" ]]; then
  export OMPI_MCA_mtl=^ofi
fi
export SLURM_MPI_TYPE="${SLURM_MPI_TYPE:-pmix}"
SRUN_MPI_TYPE="${KOBAYASHI_SRUN_MPI:-pmix}"
LAUNCHER="${KOBAYASHI_MPI_LAUNCHER:-srun}"

NX="${KOBAYASHI_SCALING_NX:-512}"
NY="${KOBAYASHI_SCALING_NY:-512}"
NSTEPS="${KOBAYASHI_SCALING_STEPS:-5000}"
DT="${KOBAYASHI_SCALING_DT:-1e-4}"
DX="${KOBAYASHI_SCALING_DX:-0.03}"

JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_ROOT="${SLURM_SUBMIT_DIR:-.}/kobayashi_hip_scaling_${JOB_TAG}"
mkdir -p "${OUT_ROOT}"

export OPENPFC_KOBAYASHI_SKIP_PNG=1
export OPENPFC_KOBAYASHI_QUIET=1

NPROC_LIST="${KOBAYASHI_SCALING_NPROC:-1 2}"

echo "OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "BIN=${BIN}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "problem: Nx=${NX} Ny=${NY} steps=${NSTEPS} dt=${DT} dx=${DX}"
echo "NPROC_LIST=${NPROC_LIST}"

SUMMARY="${OUT_ROOT}/summary.tsv"
echo -e "nproc\twall_loop_max_s\tjob_tag" >"${SUMMARY}"

for np in ${NPROC_LIST}; do
  echo ""
  echo "================================================================================"
  echo " np = ${np}  (MPI ranks / GPUs)"
  echo "================================================================================"
  outdir="${OUT_ROOT}/np_${np}"
  mkdir -p "${outdir}"
  logf="${OUT_ROOT}/run_np_${np}.log"

  if [[ "${LAUNCHER}" == "mpirun" ]]; then
    mpirun -np "${np}" --bind-to core \
      "${BIN}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${outdir}" 2>&1 | tee "${logf}"
  else
    srun --mpi="${SRUN_MPI_TYPE}" -n "${np}" --gpus-per-task=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
      "${BIN}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${outdir}" 2>&1 | tee "${logf}"
  fi

  n_verify="$(grep -cE '^KOBAYASHI_VERIFY ' "${logf}" 2>/dev/null || echo 0)"
  if [[ "${n_verify}" -ne 1 ]]; then
    echo "WARN(np=${np}): expected exactly 1 KOBAYASHI_VERIFY line, got ${n_verify}"
  fi
  wall="$(grep -E '^KOBAYASHI_VERIFY ' "${logf}" | grep -E "\\bnproc=${np}\\b" | head -1 | sed -n 's/.*wall_loop_max_s=//'p | awk '{print $1}')"
  if [[ -z "${wall}" ]]; then
    echo "WARN(np=${np}): could not parse wall_loop_max_s; writing nan"
    wall="nan"
  fi
  echo -e "${np}\t${wall}\t${JOB_TAG}" >>"${SUMMARY}"
done

echo ""
echo "Wrote logs under ${OUT_ROOT}/run_np_*.log"
echo "Wrote ${SUMMARY}"
echo "Summarize:"
echo "  python3 $(dirname "$0")/summarize_scaling.py ${OUT_ROOT}"
