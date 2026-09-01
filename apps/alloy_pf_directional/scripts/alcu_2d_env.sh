#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Shared 2D DS physics for the Al-Cu FTA scaling campaign.
# Source after lumi_paths.sh (or standalone). Sets I/O off, Ji iso on, Nz=1.
#
#   source apps/alloy_pf_directional/scripts/alcu_2d_env.sh
#   alcu_2d_apply_grid 1280x160
#   alcu_2d_apply_timing 10 80

_ALCU_2D_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! declare -F alcu_2d_apply_grid >/dev/null 2>&1; then
  # shellcheck source=lumi_paths.sh
  source "${_ALCU_2D_DIR}/lumi_paths.sh"
fi

alcu_2d_apply_timing() {
  export OPENPFC_ALCU_WARMUP="${1:-10}"
  export OPENPFC_ALCU_TIMED_STEPS="${2:-80}"
  local cap=$(( OPENPFC_ALCU_WARMUP + OPENPFC_ALCU_TIMED_STEPS ))
  export OPENPFC_ALCU_MAX_STEPS="${OPENPFC_ALCU_MAX_STEPS:-${cap}}"
  export OPENPFC_ALCU_TEND="${OPENPFC_ALCU_TEND:-1.0e-6}"
}

alcu_2d_launch_mpi() {
  local np="$1"
  shift
  if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks="${np}" --cpu-bind=cores "$@"
  elif command -v mpirun >/dev/null 2>&1; then
    mpirun -n "${np}" "$@"
  else
    echo "no srun/mpirun" >&2
    return 127
  fi
}
