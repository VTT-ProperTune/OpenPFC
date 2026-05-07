#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Compare **MPI CPU** (`kobayashi_fd_manual`) vs **MPI + CUDA** (`kobayashi_fd_cuda`) on:
#   **512×512** cells, **5000** steps, **dt=1e-4**, **dx=0.03**
# Partition **nvidia_h100** (VTT example: node **g0005**).
#
# **No exclusive node.** One allocation requests **32 CPUs** (**32 Slurm tasks × 1 CPU/task** so
# **`kobayashi_fd_manual`** runs **32 MPI ranks at 1 CPU per rank**) plus **4 GPUs** for CUDA.
# Then runs **sequentially**: CPU **`mpirun -np 32`** → CUDA **`np=1`** → **`np=2`** → **`np=4`**.
# Not four separate Slurm allocations; one shared slice of the node.
#
#   export OPENPFC_BUILD_DIR=$PWD/builds/kobayashi-cuda-h100
#   sbatch apps/kobayashi/slurm/kobayashi_compare_cpu_gpu_h100.sbatch
#
# Override problem / CPU rank count: KOBAYASHI_CMP_* / KOBAYASHI_SCALING_*.
# To request more GPUs from Slurm, edit the next line (**#SBATCH does not expand env vars**).

#SBATCH --job-name=kob-cmp-h100
#SBATCH --partition=nvidia_h100
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=kobayashi_compare_cpu_gpu_h100_%j.out
#SBATCH --error=kobayashi_compare_cpu_gpu_h100_%j.err

set -euo pipefail

OPENPFC_BUILD_DIR="${OPENPFC_BUILD_DIR:-${KOBAYASHI_CUDA_OPENPFC_BUILD_DIR:-}}"
if [[ -z "${OPENPFC_BUILD_DIR}" ]]; then
  _d="${SLURM_SUBMIT_DIR:-.}/builds/kobayashi-cuda-h100"
  if [[ -d "${_d}" ]]; then
    OPENPFC_BUILD_DIR="${_d}"
  fi
fi
if [[ -z "${OPENPFC_BUILD_DIR}" ]]; then
  echo "ERROR: set OPENPFC_BUILD_DIR to your CMake build (must contain kobayashi_fd_manual + kobayashi_fd_cuda)." >&2
  exit 1
fi

BIN_CPU="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_manual"
BIN_GPU="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_cuda"
for _b in "${BIN_CPU}" "${BIN_GPU}"; do
  if [[ ! -x "${_b}" ]]; then
    echo "ERROR: missing executable: ${_b}" >&2
    exit 1
  fi
done

GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.0}"

module purge 2>/dev/null || true
module load "${GCC_MODULE}"
module load "${OPENMPI_MODULE}"
module load "${CUDA_MODULE}"
hash -r

if [[ "${KOBAYASHI_OMPI_KEEP_OFI:-0}" != "1" ]]; then
  export OMPI_MCA_mtl=^ofi
fi
export SLURM_MPI_TYPE="${SLURM_MPI_TYPE:-pmix}"

NX="${KOBAYASHI_CMP_NX:-${KOBAYASHI_SCALING_NX:-512}}"
NY="${KOBAYASHI_CMP_NY:-${KOBAYASHI_SCALING_NY:-512}}"
NSTEPS="${KOBAYASHI_CMP_STEPS:-${KOBAYASHI_SCALING_STEPS:-5000}}"
DT="${KOBAYASHI_CMP_DT:-${KOBAYASHI_SCALING_DT:-1e-4}}"
DX="${KOBAYASHI_CMP_DX:-${KOBAYASHI_SCALING_DX:-0.03}}"
NCPU="${KOBAYASHI_CMP_NPROC_CPU:-32}"

JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_ROOT="${SLURM_SUBMIT_DIR:-.}/kobayashi_compare_cpu_gpu_h100_${JOB_TAG}"
mkdir -p "${OUT_ROOT}"

export OPENPFC_KOBAYASHI_SKIP_PNG=1
export OPENPFC_KOBAYASHI_QUIET=1

echo "host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "problem: Nx=${NX} Ny=${NY} steps=${NSTEPS} dt=${DT} dx=${DX}"
echo "CPU: kobayashi_fd_manual mpirun -np ${NCPU} (1 CPU per MPI rank via Slurm); CUDA: np=1,2,4"
echo "OUT_ROOT=${OUT_ROOT}"
echo ""

nvidia-smi -L 2>/dev/null || true
echo ""

run_one() {
  local tag="$1"
  shift
  local logf="${OUT_ROOT}/${tag}.log"
  echo "================================================================================"
  echo " ${tag}"
  echo "================================================================================"
  "$@" 2>&1 | tee "${logf}"
  wall="$(grep -E '^KOBAYASHI_VERIFY ' "${logf}" | head -1 | sed -n 's/.*wall_loop_max_s=//'p | awk '{print $1}')"
  nproc="$(grep -E '^KOBAYASHI_VERIFY ' "${logf}" | head -1 | sed -n 's/.*nproc=//'p | awk '{print $1}')"
  echo -e "${tag}\t${wall}\t${nproc}" >>"${OUT_ROOT}/summary.tsv"
  echo ""
}

echo -e "case\twall_loop_max_s\tnproc" >"${OUT_ROOT}/summary.tsv"

run_one "cpu_np${NCPU}" mpirun -np "${NCPU}" --bind-to core \
  "${BIN_CPU}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${OUT_ROOT}/cpu_np${NCPU}"
run_one "gpu_np1" mpirun -np 1 --bind-to core \
  "${BIN_GPU}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${OUT_ROOT}/gpu_np1"
run_one "gpu_np2" mpirun -np 2 --bind-to core \
  "${BIN_GPU}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${OUT_ROOT}/gpu_np2"
run_one "gpu_np4" mpirun -np 4 --bind-to core \
  "${BIN_GPU}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${OUT_ROOT}/gpu_np4"

echo "Wrote ${OUT_ROOT}/summary.tsv"
column -t "${OUT_ROOT}/summary.tsv" 2>/dev/null || cat "${OUT_ROOT}/summary.tsv"
