#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Three repeats of the LUMI-C Heat3D control at N=512 (affordable
# variability; not a replacement for the N=1024 ranking table).
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=heat3d-reps
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-heat3d-cpu}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/results/heat3d_repeats}"
HEFFTE_LIB="/flash/project_462001519/juaho/build/heffte-2.3.0-milan/lib"
export LD_LIBRARY_PATH="${HEFFTE_LIB}:${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset LD_PRELOAD
export HEAT3D_WARMUP=5 OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
export CRAYPE_LINK_TYPE=dynamic MPICH_GPU_SUPPORT_ENABLED=0
mkdir -p "${OUT}"
N=512; STEPS=30; DT=0.01
run() {
  local tag="$1"; shift
  srun --ntasks=8 --cpus-per-task=16 --cpu-bind=cores "$@" \
    | tee "${OUT}/${tag}-${SLURM_JOB_ID}.log" \
    | grep HEAT3D_CPU_WALL_STEP_MS_MEDIAN || true
}
for r in 1 2 3; do
  echo "=== repeat ${r} ==="
  run "spectral-r${r}" "${BUILD}/apps/heat3d/heat3d_spectral" "${N}" "${STEPS}" "${DT}"
  run "fd2-r${r}" "${BUILD}/apps/heat3d/heat3d_fd" "${N}" "${STEPS}" "${DT}" 2
  run "fd8-r${r}" "${BUILD}/apps/heat3d/heat3d_fd" "${N}" "${STEPS}" "${DT}" 8
  run "fd12-r${r}" "${BUILD}/apps/heat3d/heat3d_fd" "${N}" "${STEPS}" "${DT}" 12
done
echo "=== done ==="
