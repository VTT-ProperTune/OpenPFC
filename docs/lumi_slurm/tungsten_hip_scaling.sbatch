#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# tungsten_hip on LUMI-G: one MPI rank per GCD. Size one GCD, strong-scale
# 1/2/4/8 GCDs on one node, then 16/32 GCDs (2/4 nodes). I/O off. Submit with
# docs/lumi_slurm/submit_tungsten_hip_scaling.sh so ntasks and gpus-per-node
# match. Do not use the 0.1.4 / project_462001245 paths in tungsten_gpu.sbatch.
#
# Required in the job environment:
#   TUNGSTEN_HIP_BIN   absolute path to a 0.2 tungsten_hip
# Optional:
#   TUNGSTEN_LX        cubic grid (default 512)
#   TUNGSTEN_STEPS     accepted steps; t1 = steps with dt=1 (default 20)
#   OPENPFC_SCALING_ROOT  scratch run parent
#     (default /scratch/project_462001519/juaho/openpfc-scaling)

set -euo pipefail

: "${TUNGSTEN_HIP_BIN:?set TUNGSTEN_HIP_BIN to the 0.2 tungsten_hip binary}"

LX="${TUNGSTEN_LX:-512}"
STEPS="${TUNGSTEN_STEPS:-20}"
JOBROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling}"

# Slurm runs a spool copy of this script; do not resolve the TOML relative to
# BASH_SOURCE. The submit helper exports TUNGSTEN_SCALING_TEMPLATE.
TEMPLATE="${TUNGSTEN_SCALING_TEMPLATE:-}"
if [[ -z "${TEMPLATE}" || ! -f "${TEMPLATE}" ]]; then
  echo "set TUNGSTEN_SCALING_TEMPLATE to docs/lumi_slurm/tungsten_hip_scaling.toml" >&2
  exit 1
fi

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1

NTASKS="${SLURM_NTASKS:?SLURM_NTASKS is not set}"
RUNDIR="${JOBROOT}/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RUNDIR}"

sed -e "s/__LX__/${LX}/g" -e "s/__T1__/${STEPS}/g" "${TEMPLATE}" \
  > "${RUNDIR}/input.toml"
cp -f "${BASH_SOURCE[0]}" "${RUNDIR}/tungsten_hip_scaling.sbatch"
cd "${RUNDIR}"

cat << 'EOF' > "${RUNDIR}/select_gpu"
#!/bin/bash
export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}
exec "$@"
EOF
chmod +x "${RUNDIR}/select_gpu"

# Full-node 8-GCD map from LUMI-G docs, applied per node. Partial GCD
# allocations (dev-g / small-g with 1–4 GPUs) do not own those CPUs;
# map_cpu:49 then fails.
PER_NODE="${SLURM_NTASKS_PER_NODE:-${NTASKS}}"
SRUN_BIND=()
CPU_BIND="none"
if (( PER_NODE == 8 )); then
  CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
  SRUN_BIND=(--cpu-bind="${CPU_BIND}")
elif (( PER_NODE > 8 )); then
  echo "at most 8 ranks per node (one per GCD); ntasks-per-node=${PER_NODE}" >&2
  exit 1
fi

{
  echo "=== tungsten_hip scaling ==="
  echo "job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME}"
  echo "nodes=${SLURM_NNODES} ntasks=${NTASKS} partition=${SLURM_JOB_PARTITION}"
  echo "bin=${TUNGSTEN_HIP_BIN}"
  echo "Lx=Ly=Lz=${LX} steps=${STEPS} cwd=${RUNDIR}"
  echo "cpu-bind=${CPU_BIND}"
  echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
} | tee "${RUNDIR}/run_meta.txt"

srun "${SRUN_BIND[@]}" "${RUNDIR}/select_gpu" \
  "${TUNGSTEN_HIP_BIN}" "${RUNDIR}/input.toml"

rm -f "${RUNDIR}/select_gpu"
echo "=== done ==="
