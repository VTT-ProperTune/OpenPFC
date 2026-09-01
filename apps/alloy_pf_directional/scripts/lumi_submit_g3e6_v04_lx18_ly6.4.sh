#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Relative to the Ly=3.20 μm / Lx=12 μm series: double width, +50% length.
#   Lx = 18.0 μm, Ly = 6.40 μm, dx = W0, dt = 0.2 τ0
#   W0 = 20, 10, 5 nm. t_end = 180 μs so the extra length can be used.
#   Uses the installed CPU OpenMP binary (no rebuild).
#
#   ./apps/alloy_pf_directional/scripts/lumi_submit_g3e6_v04_lx18_ly6.4.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lumi_paths.sh
source "${HERE}/lumi_paths.sh"

REMOTE="${LUMI_USER}@${LUMI_HOST}"
SRC="${LUMI_SRC}"

ssh -o BatchMode=yes -o ConnectTimeout=25 "${REMOTE}" bash -s <<EOF
set -euo pipefail
cd '${SRC}'

PHYS="OPENPFC_ALCU_G=3.0e6,OPENPFC_ALCU_VP=0.4,OPENPFC_ALCU_LX=18.0e-6,OPENPFC_ALCU_LY=6.40e-6,OPENPFC_ALCU_TEND=180.0e-6,OPENPFC_ALCU_SEED=0.20e-6,OPENPFC_ALCU_NGRANS=1,OPENPFC_ALCU_NOISE=0,OPENPFC_ALCU_WINDOW=0,OPENPFC_ALCU_STOP_FAR_C=1,OPENPFC_ALCU_STOP_RIGHT=0,OPENPFC_ALCU_PERIODIC_Y=1,OPENPFC_ALCU_SKIP_PNG=1,OPENPFC_ALCU_SKIP_VTK=1"

submit_ds() {
  local name="\$1" w0="\$2" save="\$3" time="\$4" cpus="\$5" jn="\$6"
  sbatch --parsable \\
    --job-name="\${jn}" \\
    --time="\${time}" \\
    --cpus-per-task="\${cpus}" \\
    --export=ALL,CASE=\${name},W0=\${w0},DXW=1.0,SAVE_EVERY=\${save},OPENPFC_ALCU_DT_OVER_TAU=0.2,\${PHYS} \\
    apps/alloy_pf_directional/scripts/lumi_ds.sh
}

echo -n "W20nm "; submit_ds g3e6_V04_Lx18_Ly6.4_w0_20nm 20e-9 1000 01:00:00 16 alcu-L18y64-w20
echo -n "W10nm "; submit_ds g3e6_V04_Lx18_Ly6.4_w0_10nm 10e-9 4000 03:00:00 32 alcu-L18y64-w10
echo -n "W5nm "; submit_ds g3e6_V04_Lx18_Ly6.4_w0_5nm 5e-9 15000 16:00:00 64 alcu-L18y64-w5
squeue -u \$USER -o '%.18i %.12P %.22j %.2t %.10M %.6D %R'
EOF
