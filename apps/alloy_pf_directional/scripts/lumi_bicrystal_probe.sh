#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Short two-grain probes on LUMI-C debug: overlap vs dt.
# Run after a CPU rebuild:
#   sbatch --dependency=afterok:$BUILD_ID apps/alloy_pf_directional/scripts/lumi_bicrystal_probe.sh
#SBATCH --job-name=alcu-bi-probe
#SBATCH --account=project_462001519
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1750
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/alcu_fta/logs/probe-%j.out
set -euo pipefail

# shellcheck source=lumi_paths.sh
source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_directional/scripts/lumi_paths.sh

module purge
module load "${LUMI_STACK}" partition/C cpeGNU cray-fftw lumi-CrayPath
lumi_cpu_runtime_env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export OPENPFC_ALCU_QUIET=1
export OPENPFC_ALCU_SKIP_VTK=1
export OPENPFC_ALCU_NOISE=0
export OPENPFC_ALCU_WINDOW=0
export OPENPFC_ALCU_STOP_FAR_C=0
export OPENPFC_ALCU_STOP_RIGHT=0
export OPENPFC_ALCU_PERIODIC_Y=1
export OPENPFC_ALCU_G=3.0e6
export OPENPFC_ALCU_VP=0.4
export OPENPFC_ALCU_LX=4.0e-6
export OPENPFC_ALCU_LY=3.20e-6
export OPENPFC_ALCU_TEND=20.0e-6
export OPENPFC_ALCU_SEED=0.20e-6
export OPENPFC_ALCU_THETA=30
export OPENPFC_ALCU_W0=20e-9
export OPENPFC_ALCU_DXW=1.0
export OPENPFC_ALCU_MAX_STEPS=800
export OPENPFC_ALCU_SAVE_EVERY=100
export OPENPFC_ALCU_LOG_EVERY=20

run_one() {
  local name="$1"
  shift
  local out="${LUMI_RUNS}/bicrystal/probe_${name}"
  rm -rf "${out}"
  mkdir -p "${out}"
  echo "=== ${name} ==="
  srun --cpu-bind=cores --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    "${LUMI_BIN}" ds "${out}" "${SLURM_CPUS_PER_TASK}" \
    --save-every 100 --log-every 20 "$@" || true
  echo "--- ${name} meta/overlap ---"
  grep -E "n_grains|r_seed|y_seed|seed_gap|dt_over_tau|phi1_g" "${out}/meta.txt" || true
  echo "# fields.log last"
  tail -5 "${out}/fields.log" || true
  if [[ -f "${out}/abort.txt" ]]; then
    echo "# abort"
    cat "${out}/abort.txt"
  fi
}

# 1-grain control at the production dt.
export OPENPFC_ALCU_NGRANS=1
export OPENPFC_ALCU_DT_OVER_TAU=0.2
export OPENPFC_ALCU_SKIP_PNG=1
run_one n1_dt02

# Two grains, production dt (0.2 τ0) and a tighter step.
export OPENPFC_ALCU_NGRANS=2
unset OPENPFC_ALCU_SKIP_PNG
run_one n2_dt02

export OPENPFC_ALCU_DT_OVER_TAU=0.05
export OPENPFC_ALCU_SKIP_PNG=1
run_one n2_dt005

echo "probe done"
squeue -u "$USER" -o '%.18i %.12P %.22j %.2t %.10M %R' || true
