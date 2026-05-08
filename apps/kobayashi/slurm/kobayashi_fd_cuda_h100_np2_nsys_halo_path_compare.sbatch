#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# **A/B profile:** same problem and `nsys profile` setup twice in one job:
#   1) default **GPU-aware MPI** halos (`OPENPFC_CUDA_FORCE_PACKED_HALO` unset)
#   2) **packed face** halos (`OPENPFC_CUDA_FORCE_PACKED_HALO=1` → host MPI)
#
# Use this to confirm that the GPU-aware stack is generating a storm of
# **`cuMemcpyAsync` / `cuStreamSynchronize`** (driver-level) while the packed
# path uses **O(faces × fields)** narrow memcpy calls per step. See
# `apps/kobayashi/docs/scalability_cuda_h100.md` — “CUDA API call counts”.
#
# Submit from repo root (**defaults: 8192×4096**, 200 steps — matches large-grid halo
# investigation). Optional **1 CPU/task** for binding parity with `np2_1cpu`:
#   sbatch --cpus-per-task=1 apps/kobayashi/slurm/kobayashi_fd_cuda_h100_np2_nsys_halo_path_compare.sbatch
# Quick **512²** smoke (small `.nsys-rep`, faster):
#   sbatch --export=ALL,KOBAYASHI_NSYS_NX=512,KOBAYASHI_NSYS_NY=512 \
#     apps/kobayashi/slurm/kobayashi_fd_cuda_h100_np2_nsys_halo_path_compare.sbatch
#
# Large `.nsys-rep` files are sqlite-heavy: if **`/WRKDIR/$USER/openpfc`** exists
# (fast NVMe), artifacts go there; else under **`SLURM_SUBMIT_DIR`**. Override with
# **`KOBAYASHI_NSYS_ARTIFACT_BASE`**. See **`pick_nsys_artifact_base.bash`**.

#SBATCH --job-name=kob-nsys-halo-ab
#SBATCH --partition=nvidia_h100
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=kobayashi_fd_cuda_h100_np2_nsys_halo_ab_%j.out
#SBATCH --error=kobayashi_fd_cuda_h100_np2_nsys_halo_ab_%j.err

set -euo pipefail

SLURM_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=pick_nsys_artifact_base.bash
source "${SLURM_SCRIPT_DIR}/pick_nsys_artifact_base.bash"

OPENPFC_BUILD_DIR="${OPENPFC_BUILD_DIR:-${KOBAYASHI_CUDA_OPENPFC_BUILD_DIR:-}}"
if [[ -z "${OPENPFC_BUILD_DIR}" ]]; then
  _d="${SLURM_SUBMIT_DIR:-.}/builds/kobayashi-cuda-h100"
  if [[ -d "${_d}" ]]; then
    OPENPFC_BUILD_DIR="${_d}"
  fi
fi
if [[ -z "${OPENPFC_BUILD_DIR}" ]]; then
  echo "ERROR: set OPENPFC_BUILD_DIR (CUDA build with kobayashi_fd_cuda)." >&2
  exit 1
fi

BIN_GPU="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_cuda"
if [[ ! -x "${BIN_GPU}" ]]; then
  echo "ERROR: missing ${BIN_GPU}" >&2
  exit 1
fi

GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.0}"

module purge 2>/dev/null || true
module load "${GCC_MODULE}"
module load "${OPENMPI_MODULE}"
module load "${CUDA_MODULE}"
hash -r

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not on PATH after loading CUDA module ${CUDA_MODULE}" >&2
  exit 1
fi

if [[ "${KOBAYASHI_OMPI_KEEP_OFI:-0}" != "1" ]]; then
  export OMPI_MCA_mtl=^ofi
fi
export SLURM_MPI_TYPE="${SLURM_MPI_TYPE:-pmix}"

# Defaults **8192×4096** for apples-to-apples with `kobayashi_fd_cuda_h100_np2_nsys`
# large-grid runs (e.g. job 1236668). Override with **`KOBAYASHI_NSYS_NX` /
# `KOBAYASHI_NSYS_NY`** (and optionally **`KOBAYASHI_NSYS_STEPS`**).
NX="${KOBAYASHI_NSYS_NX:-8192}"
NY="${KOBAYASHI_NSYS_NY:-4096}"
NSTEPS="${KOBAYASHI_NSYS_STEPS:-200}"
DT="${KOBAYASHI_CMP_DT:-${KOBAYASHI_SCALING_DT:-1e-4}}"
DX="${KOBAYASHI_CMP_DX:-${KOBAYASHI_SCALING_DX:-0.03}}"

JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_ROOT="${KOBAYASHI_NSYS_ARTIFACT_BASE}/kobayashi_fd_cuda_h100_nsys_halo_ab_${JOB_TAG}"
mkdir -p "${OUT_ROOT}"

export OPENPFC_KOBAYASHI_SKIP_PNG=1
export OPENPFC_KOBAYASHI_QUIET=1
export OPENPFC_KOBAYASHI_PERF=1

echo "host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "KOBAYASHI_NSYS_ARTIFACT_BASE=${KOBAYASHI_NSYS_ARTIFACT_BASE} (${KOBAYASHI_NSYS_ARTIFACT_BASE_REASON})"
echo "problem: Nx=${NX} Ny=${NY} steps=${NSTEPS}"
echo "OUT_ROOT=${OUT_ROOT}"
echo ""

run_one_profile() {
  local label="$1"
  local subdir="$2"
  local -a env_prefix=("${@:3}")

  local dir="${OUT_ROOT}/${subdir}"
  mkdir -p "${dir}"
  local logf="${dir}/gpu_np2_nsys.log"
  local nsys_base="${dir}/kob_nsys"

  echo "========== PROFILE: ${label} =========="
  (
    cd "${dir}"
    "${env_prefix[@]}" nsys profile \
      --trace=cuda,nvtx,osrt,mpi \
      --mpi-impl=openmpi \
      --force-overwrite=true \
      -o "${nsys_base}" \
      -- \
      mpirun -np 2 --bind-to core \
        "${BIN_GPU}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${dir}/gpu_np2" \
      2>&1 | tee "${logf}"
  )

  shopt -s nullglob
  local rep=( "${dir}"/kob_nsys*.nsys-rep )
  if [[ ${#rep[@]} -eq 0 ]]; then
    echo "(no .nsys-rep for ${label})"
    return 0
  fi
  echo "--- nsys cuda_api_sum (top lines, ${label}) ---"
  nsys stats -q --report cuda_api_sum "${rep[0]}" 2>/dev/null | head -35 || true
  echo ""
  grep -E '^(KOBAYASHI_CUDA device|KOBAYASHI_CUDA_HALO|KOBAYASHI_PERF_LOOP|OPENPFC_CUDA_PROFILE_HALO_SUMMARY)' "${logf}" >&2 || true
  echo ""
}

# A: GPU-aware path (do not force packed)
unset OPENPFC_CUDA_FORCE_PACKED_HALO || true
run_one_profile "gpu_aware_default" "A_gpu_aware" env -u OPENPFC_CUDA_FORCE_PACKED_HALO

# B: Packed faces (MPI on pinned host; app uses cudaMemcpyAsync per face slab)
export OPENPFC_CUDA_FORCE_PACKED_HALO=1
run_one_profile "packed_pcie_forced" "B_packed_forced"

echo "DONE: compare Num Calls for cuMemcpyAsync / cuStreamSynchronize in the two blocks above."
echo "Artifacts: ${OUT_ROOT}/A_gpu_aware/ and ${OUT_ROOT}/B_packed_forced/"
