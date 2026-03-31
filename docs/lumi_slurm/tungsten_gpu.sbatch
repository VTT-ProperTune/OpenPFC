#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001245
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_462001245/juaho/tungsten_perf_jobs/logs/%x-%j.out
#SBATCH --chdir=/scratch/project_462001245/juaho/tungsten_perf_jobs

# Tungsten HIP on LUMI-G: one MPI rank per GCD (8 per node), GPU selection via ROCR_VISIBLE_DEVICES.
# Prereqs: OpenPFC built with -DOpenPFC_MPI_HIP_AWARE=ON (default), HeFFTe with GPU-aware MPI,
# MPICH_GPU_SUPPORT_ENABLED=1 (set below). Run docs/lumi_slurm/verify_gpu_aware_mpi.sh once to confirm.
# CPU pinning matches LUMI docs (rank/GPU/NUMA alignment):
#   https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/
#   https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/#gpu-binding
#
# Submit:  sbatch --nodes=2 --job-name=tperf-gpu-2n tungsten_gpu.sbatch

set -euo pipefail

JOBROOT=/scratch/project_462001245/juaho/tungsten_perf_jobs
REPO_INPUT=/users/juaho/dev/openpfc/docs/lumi_slurm/tungsten_performance_gpu.toml
OPENPFC_BIN_DEFAULT=/projappl/project_462001245/openpfc/0.1.4-hip/bin/tungsten_hip
TUNGSTEN_BIN="${TUNGSTEN_HIP_BIN:-$OPENPFC_BIN_DEFAULT}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1

RUNDIR="${JOBROOT}/runs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RUNDIR}/data"
cp -f "${REPO_INPUT}" "${RUNDIR}/input.toml"
cd "${RUNDIR}"

cat << 'EOF' > "${RUNDIR}/select_gpu"
#!/bin/bash
export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}
exec "$@"
EOF
chmod +x "${RUNDIR}/select_gpu"

# From LUMI "MPI-based job" example (one rank per GCD)
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

echo "=== Tungsten HIP === nodes=${SLURM_NNODES} ntasks=${SLURM_NTASKS} bin=${TUNGSTEN_BIN}"
echo "=== cwd=${RUNDIR}"

srun --cpu-bind="${CPU_BIND}" "${RUNDIR}/select_gpu" "${TUNGSTEN_BIN}" input.toml

rm -f "${RUNDIR}/select_gpu"
echo "=== done"
