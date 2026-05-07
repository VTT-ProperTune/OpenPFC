#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Strong scaling for apps/kobayashi/kobayashi_fd_manual on partition gen05_epyc.
#
# Submit from any directory (logs land in SLURM_SUBMIT_DIR):
#   export OPENPFC_BUILD_DIR=/path/to/openpfc/build   # must contain kobayashi_fd_manual
#   sbatch apps/kobayashi/slurm/kobayashi_scaling_gen05_epyc.sbatch
#
# Optional: SBATCH_ACCOUNT=my_project  sbatch ...
#
# Requires one node with >=192 cores if you use np=192 (adjust loop below for your site).

#SBATCH --job-name=kob-scale
#SBATCH --partition=gen05_epyc
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=kobayashi_scaling_%j.out
#SBATCH --error=kobayashi_scaling_%j.err

set -euo pipefail

if [[ -z "${OPENPFC_BUILD_DIR:-}" ]]; then
  echo "ERROR: set OPENPFC_BUILD_DIR to your CMake build tree (contains apps/kobayashi/kobayashi_fd_manual)." >&2
  exit 1
fi

BIN="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_manual"
if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: missing executable: ${BIN}" >&2
  exit 1
fi

GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda}"

module purge 2>/dev/null || true
module load "${GCC_MODULE}"
module load "${OPENMPI_MODULE}"
if module load "${CUDA_MODULE}" 2>/dev/null; then
  echo "Loaded CUDA module: ${CUDA_MODULE}"
else
  echo "WARN: CUDA module '${CUDA_MODULE}' unavailable; continuing CPU-only."
fi
hash -r

# Open MPI loads CUDA accelerator MCA plugins if built with CUDA; CPU nodes often lack the NVIDIA
# driver lib (libcuda.so.1). CUDA toolkit stubs satisfy dlopen without enabling GPU work.
for _stub in "${CUDA_HOME:-}" "${CUDA_PATH:-}" "${CUDA_ROOT:-}"; do
  if [[ -n "${_stub}" && -f "${_stub}/lib64/stubs/libcuda.so" ]]; then
    export LD_LIBRARY_PATH="${_stub}/lib64/stubs:${LD_LIBRARY_PATH:-}"
    echo "Prepended ${_stub}/lib64/stubs to LD_LIBRARY_PATH (CUDA stubs for Open MPI MCA)."
    break
  fi
done
unset _stub

# Match rebuild script: avoid readlink -f on mpicc (opal_wrapper prefix split on some sites).
RUNTIME_OPENMPI_ROOT=""
if command -v mpicc >/dev/null 2>&1; then
  RUNTIME_OPENMPI_ROOT="$(mpicc --showme:prefix 2>/dev/null || true)"
fi
if [[ -z "${RUNTIME_OPENMPI_ROOT}" || ! -d "${RUNTIME_OPENMPI_ROOT}/lib" ]]; then
  RUNTIME_OPENMPI_ROOT="$(dirname "$(dirname "$(command -v mpicc)")")"
fi
if [[ ! -d "${RUNTIME_OPENMPI_ROOT}/lib" ]]; then
  echo "ERROR: RUNTIME_OPENMPI_ROOT=${RUNTIME_OPENMPI_ROOT} missing lib/ after module load." >&2
  exit 1
fi

# Fail fast if the binary was built against a different Open MPI tree than this job loads (singleton MPI / wrong PMI).
if command -v ldd >/dev/null 2>&1; then
  echo "linked libmpi (ldd ${BIN} | grep mpi):"
  ldd "${BIN}" 2>/dev/null | grep -E 'libmpi|openmpi' || true
  if ! ldd "${BIN}" 2>/dev/null | grep -E 'libmpi\.so' | grep -qF "${RUNTIME_OPENMPI_ROOT}"; then
    echo "ERROR: ${BIN} does not resolve libmpi under loaded prefix ${RUNTIME_OPENMPI_ROOT}." >&2
    echo "Rebuild on a node with ${OPENMPI_MODULE} (see kobayashi_rebuild_openpfc_gen05_epyc.sbatch) or fix MODULEPATH." >&2
    exit 1
  fi
fi

# Same workload as recent interactive runs: 512², 5000 steps (PNG off for fair timing).
NX="${KOBAYASHI_SCALING_NX:-512}"
NY="${KOBAYASHI_SCALING_NY:-512}"
NSTEPS="${KOBAYASHI_SCALING_STEPS:-5000}"
DT="${KOBAYASHI_SCALING_DT:-1e-4}"
DX="${KOBAYASHI_SCALING_DX:-0.03}"

JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_ROOT="${SLURM_SUBMIT_DIR:-.}/kobayashi_scaling_${JOB_TAG}"
mkdir -p "${OUT_ROOT}"

export OPENPFC_KOBAYASHI_SKIP_PNG=1
export OPENPFC_KOBAYASHI_QUIET=1

# OpenFabrics / Libfabric OFI MTL: fi_domain failures ("No data available") on some compute images break or
# noisy-init multi-rank jobs. Disable the OFI MTL so MPI uses ob1/BTL (self,vader on one node).
# Override at submit time if your centre requires OFI: unset OMPI_MCA_mtl or set KOBAYASHI_OMPI_KEEP_OFI=1.
if [[ "${KOBAYASHI_OMPI_KEEP_OFI:-0}" != "1" ]]; then
  export OMPI_MCA_mtl=^ofi
fi

# Single MPI job across ranks: Slurm must wire PMIx to Open MPI. Default is generic **pmix** (maps to the
# site PMIx plugin, e.g. pmix_v2). Do not assume pmix_v3 — run **`srun --mpi=list`** on the cluster if unsure.
export SLURM_MPI_TYPE="${SLURM_MPI_TYPE:-pmix}"
SRUN_MPI_TYPE="${KOBAYASHI_SRUN_MPI:-pmix}"
LAUNCHER="${KOBAYASHI_MPI_LAUNCHER:-srun}"

echo "MPI runtime: OPENMPI_MODULE=${OPENMPI_MODULE} RUNTIME_OPENMPI_ROOT=${RUNTIME_OPENMPI_ROOT}"
echo "MPI launch: LAUNCHER=${LAUNCHER} SLURM_MPI_TYPE=${SLURM_MPI_TYPE} KOBAYASHI_SRUN_MPI=${SRUN_MPI_TYPE}"
mpirun --version 2>/dev/null | head -n 2 || true

# 1, 2, 4, 8, 16, 32, 64, 128, 192 MPI ranks (sixteen — not "6 CPUs").
NPROC_LIST="${KOBAYASHI_SCALING_NPROC:-1 2 4 8 16 32 64 128 192}"

echo "OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "BIN=${BIN}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "problem: Nx=${NX} Ny=${NY} steps=${NSTEPS} dt=${DT} dx=${DX}"
echo "SLURM_JOB_ID=${JOB_TAG} SLURM_NTASKS(job)=${SLURM_NTASKS:-}"

SUMMARY="${OUT_ROOT}/summary.tsv"
echo -e "nproc\twall_loop_max_s\tjob_tag" >"${SUMMARY}"

for np in ${NPROC_LIST}; do
  echo ""
  echo "================================================================================"
  echo " np = ${np}"
  echo "================================================================================"
  outdir="${OUT_ROOT}/np_${np}"
  mkdir -p "${outdir}"
  logf="${OUT_ROOT}/run_np_${np}.log"

  # Launch exactly np MPI ranks (srun+PMIx for Slurm; or mpirun if KOBAYASHI_MPI_LAUNCHER=mpirun).
  if [[ "${LAUNCHER}" == "mpirun" ]]; then
    mpirun -np "${np}" --bind-to core \
      "${BIN}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${outdir}" 2>&1 | tee "${logf}"
  else
    srun --mpi="${SRUN_MPI_TYPE}" -n "${np}" --cpu-bind=cores \
      "${BIN}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${outdir}" 2>&1 | tee "${logf}"
  fi

  # Do not match KOBAYASHI_VERIFY_HEX (lines start with the same prefix for naive '^KOBAYASHI_VERIFY').
  n_verify="$(grep -cE '^KOBAYASHI_VERIFY ' "${logf}" 2>/dev/null || echo 0)"
  if [[ "${n_verify}" -ne 1 ]]; then
    echo "WARN(np=${np}): expected exactly 1 KOBAYASHI_VERIFY summary line, got ${n_verify}" \
      "(singleton MPI if >1 task each prints nproc=1 — check PMIx: SLURM_MPI_TYPE / KOBAYASHI_SRUN_MPI, or try KOBAYASHI_MPI_LAUNCHER=mpirun)."
  fi
  wall="$(grep -E '^KOBAYASHI_VERIFY ' "${logf}" | grep -E "\\bnproc=${np}\\b" | head -1 | sed -n 's/.*wall_loop_max_s=//'p | awk '{print $1}')"
  if [[ -z "${wall}" ]]; then
    echo "WARN(np=${np}): no KOBAYASHI_VERIFY with nproc=${np}; writing nan to summary.tsv (check log for singleton MPI)."
    wall="nan"
  fi
  echo -e "${np}\t${wall}\t${JOB_TAG}" >>"${SUMMARY}"
done

echo ""
echo "Wrote per-nproc logs under ${OUT_ROOT}/run_np_*.log"
echo "Wrote ${SUMMARY}"
echo "Parse / verify bitwise equality:"
echo "  python3 $(dirname "$0")/summarize_scaling.py ${OUT_ROOT}"
