#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 4× wider channel (Ly = 3.20 μm) at dx = W0, dt = 0.2 τ0:
#   W0 = 20, 10, 5 nm. Same G, Vp, Lx, abort as the narrow series.
#   Uses the installed CPU OpenMP binary (no rebuild).
#
#   ./apps/alloy_pf_directional/scripts/lumi_submit_g3e6_v04_wide.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lumi_paths.sh
source "${HERE}/lumi_paths.sh"

REMOTE="${LUMI_USER}@${LUMI_HOST}"
SRC="${LUMI_SRC}"

ssh -o BatchMode=yes -o ConnectTimeout=25 "${REMOTE}" bash -s <<EOF
set -euo pipefail
cd '${SRC}'

PHYS="OPENPFC_ALCU_G=3.0e6,OPENPFC_ALCU_VP=0.4,OPENPFC_ALCU_LX=12.0e-6,OPENPFC_ALCU_LY=3.20e-6,OPENPFC_ALCU_TEND=120.0e-6,OPENPFC_ALCU_SEED=0.20e-6,OPENPFC_ALCU_NGRANS=1,OPENPFC_ALCU_NOISE=0,OPENPFC_ALCU_WINDOW=0,OPENPFC_ALCU_STOP_FAR_C=1,OPENPFC_ALCU_STOP_RIGHT=0,OPENPFC_ALCU_PERIODIC_Y=1,OPENPFC_ALCU_SKIP_PNG=1,OPENPFC_ALCU_SKIP_VTK=1"

submit_ds() {
  local name="\$1" w0="\$2" save="\$3" time="\$4" cpus="\$5" jn="\$6"
  sbatch --parsable \\
    --job-name="\${jn}" \\
    --time="\${time}" \\
    --cpus-per-task="\${cpus}" \\
    --export=ALL,CASE=\${name},W0=\${w0},DXW=1.0,SAVE_EVERY=\${save},OPENPFC_ALCU_DT_OVER_TAU=0.2,\${PHYS} \\
    apps/alloy_pf_directional/scripts/lumi_ds.sh
}

echo -n "W20nm "; submit_ds g3e6_V04_Ly3.2_w0_20nm 20e-9 500 01:00:00 8 alcu-Ly32-w20
echo -n "W10nm "; submit_ds g3e6_V04_Ly3.2_w0_10nm 10e-9 2000 02:00:00 16 alcu-Ly32-w10
echo -n "W5nm "; submit_ds g3e6_V04_Ly3.2_w0_5nm 5e-9 10000 08:00:00 32 alcu-Ly32-w5
squeue -u \$USER -o '%.18i %.12P %.22j %.2t %.10M %.6D %R'
EOF
