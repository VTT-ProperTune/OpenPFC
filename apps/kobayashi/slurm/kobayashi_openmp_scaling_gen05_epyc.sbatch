#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Strong threading scaling for apps/kobayashi/kobayashi_fd_openmp on gen05_epyc.
# Single Slurm task; sweep OMP_NUM_THREADS (and argv thread override).
#
# Optional tuning:
#   - **KOBAYASHI_OPENMP_USE_SRUN=1**: wrap each sweep with **srun --cpu-bind=cores**. Uses
#     **--cpus-per-task=<OMP_NUM_THREADS>** for that iteration so the affinity mask matches the
#     thread team (binding all **SLURM_CPUS_PER_TASK** cpus while using fewer OpenMP threads interacts
#     badly with **OMP_PROC_BIND=spread** on some hosts). Default is **off** after observed regressions.
#   - **OMP_DYNAMIC=false** by default; set **OMP_WAIT_POLICY** yourself if needed (do not default to
#     **passive** here: thousands of barriers/step magnify passive wake latency).
#   - **KOBAYASHI_NUMA_INTERLEAVE=1**: **numactl --interleave=all** (optional experiment).
#
#   export OPENPFC_BUILD_DIR=/path/to/build   # must contain kobayashi_fd_openmp
#   sbatch apps/kobayashi/slurm/kobayashi_openmp_scaling_gen05_epyc.sbatch
#
# If the binary is missing, run kobayashi_rebuild_openpfc_gen05_epyc.sbatch first (it builds
# kobayashi_fd_openmp when CMake finds OpenMP).

#SBATCH --job-name=kob-omp-scale
#SBATCH --partition=gen05_epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00
#SBATCH --output=kobayashi_openmp_scaling_%j.out
#SBATCH --error=kobayashi_openmp_scaling_%j.err

set -euo pipefail

if [[ -z "${OPENPFC_BUILD_DIR:-}" ]]; then
  echo "ERROR: set OPENPFC_BUILD_DIR to your CMake build tree (contains kobayashi_fd_openmp)." >&2
  exit 1
fi

BIN="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_openmp"
if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: missing executable: ${BIN} (build with OpenMP enabled)." >&2
  exit 1
fi

GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda}"

module purge 2>/dev/null || true
module load "${GCC_MODULE}"
if module load "${CUDA_MODULE}" 2>/dev/null; then
  echo "Loaded CUDA module: ${CUDA_MODULE}"
else
  echo "WARN: CUDA module '${CUDA_MODULE}' unavailable."
fi
hash -r

NX="${KOBAYASHI_SCALING_NX:-512}"
NY="${KOBAYASHI_SCALING_NY:-512}"
NSTEPS="${KOBAYASHI_SCALING_STEPS:-5000}"
DT="${KOBAYASHI_SCALING_DT:-1e-4}"
DX="${KOBAYASHI_SCALING_DX:-0.03}"

JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_ROOT="${SLURM_SUBMIT_DIR:-.}/kobayashi_openmp_scaling_${JOB_TAG}"
mkdir -p "${OUT_ROOT}"

export OPENPFC_KOBAYASHI_SKIP_PNG=1
export OPENPFC_KOBAYASHI_QUIET=1

export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-false}"

THREAD_LIST="${KOBAYASHI_OPENMP_THREADS:-1 2 4 8 16 32 64 128 192}"

kobayashi_openmp_launch() {
  local nt_launch="${1:?}"
  shift
  local -a cmd=( "$@" )
  if [[ "${KOBAYASHI_NUMA_INTERLEAVE:-0}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
    cmd=( numactl --interleave=all "${cmd[@]}" )
    echo "NUMA: numactl --interleave=all"
  elif [[ "${KOBAYASHI_NUMA_INTERLEAVE:-0}" == "1" ]]; then
    echo "WARN: KOBAYASHI_NUMA_INTERLEAVE=1 but numactl not on PATH; ignoring." >&2
  fi
  if [[ "${KOBAYASHI_OPENMP_USE_SRUN:-0}" == "1" ]] && [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    local alloc="${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK unset}"
    local cpus="${KOBAYASHI_OPENMP_SRUN_CPUS:-${nt_launch}}"
    if [[ "${cpus}" -gt "${alloc}" ]]; then
      echo "ERROR: thread count ${cpus} exceeds SLURM_CPUS_PER_TASK=${alloc}" >&2
      exit 1
    fi
    echo "launcher: srun --ntasks=1 --cpus-per-task=${cpus} --cpu-bind=cores"
    srun --ntasks=1 --cpus-per-task="${cpus}" --cpu-bind=cores "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

echo "OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "BIN=${BIN}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "problem: Nx=${NX} Ny=${NY} steps=${NSTEPS} dt=${DT} dx=${DX}"
echo "OMP_PROC_BIND=${OMP_PROC_BIND} OMP_PLACES=${OMP_PLACES} OMP_DYNAMIC=${OMP_DYNAMIC} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-<unset>}"
echo "KOBAYASHI_OPENMP_USE_SRUN=${KOBAYASHI_OPENMP_USE_SRUN:-0} KOBAYASHI_NUMA_INTERLEAVE=${KOBAYASHI_NUMA_INTERLEAVE:-0}"

SUMMARY="${OUT_ROOT}/summary.tsv"
echo -e "nthreads\twall_loop_max_s\tjob_tag" >"${SUMMARY}"

for nt in ${THREAD_LIST}; do
  echo ""
  echo "================================================================================"
  echo " OMP_NUM_THREADS = ${nt}"
  echo "================================================================================"
  export OMP_NUM_THREADS="${nt}"
  outdir="${OUT_ROOT}/threads_${nt}"
  mkdir -p "${outdir}"
  logf="${OUT_ROOT}/run_threads_${nt}.log"

  # Explicit argv thread count matches OMP_NUM_THREADS (redundant but logged in VERIFY).
  kobayashi_openmp_launch "${nt}" "${BIN}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${outdir}" "${nt}" 2>&1 | tee "${logf}"

  n_verify="$(grep -cE '^KOBAYASHI_VERIFY ' "${logf}" 2>/dev/null || echo 0)"
  if [[ "${n_verify}" -ne 1 ]]; then
    echo "WARN(threads=${nt}): expected exactly 1 KOBAYASHI_VERIFY line, got ${n_verify}"
  fi
  wall="$(grep -E '^KOBAYASHI_VERIFY ' "${logf}" | grep -E "\\bnthreads=${nt}\\b" | head -1 | sed -n 's/.*wall_loop_max_s=//'p | awk '{print $1}')"
  if [[ -z "${wall}" ]]; then
    echo "WARN(threads=${nt}): could not parse wall_loop_max_s; writing nan"
    wall="nan"
  fi
  echo -e "${nt}\t${wall}\t${JOB_TAG}" >>"${SUMMARY}"
done

echo ""
echo "Wrote logs under ${OUT_ROOT}/run_threads_*.log"
echo "Wrote ${SUMMARY}"
echo "Summarize:"
echo "  python3 $(dirname "$0")/summarize_openmp_scaling.py ${OUT_ROOT}"
echo "Plot:"
echo "  python3 $(dirname "$0")/plot_strong_scaling.py ${OUT_ROOT}"
