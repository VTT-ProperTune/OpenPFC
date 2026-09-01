#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Doubled box with Langevin noise on φ, two grains at ±30°:
#   G = 3e6 K/m, Vp = 0.4 m/s, Lx = 24 μm, Ly = 6.40 μm, dt = 0.2 τ0
#   W0 = 10 nm (2400x640) and 5 nm (4800x1280), F0 = 1e-3.
# Twice the box of lumi_submit_bicrystal_g3e6_v04_wide.sh, so t_end is
# doubled too; STOP_FAR_C ends each run when the front reaches the far wall.
#
#   ./apps/alloy_pf_directional/scripts/lumi_submit_bicrystal_2x_noise.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lumi_paths.sh
source "${HERE}/lumi_paths.sh"

"${HERE}/sync_to_lumi.sh"

REMOTE="${LUMI_USER}@${LUMI_HOST}"
SRC="${LUMI_SRC}"
F0="${F0:-1e-3}"

ssh -o BatchMode=yes -o ConnectTimeout=25 "${REMOTE}" bash -s <<EOF
set -euo pipefail
cd '${SRC}'
bash apps/alloy_pf_directional/scripts/lumi_build_cpu.sh configure
BUILD_ID=\$(sbatch --parsable apps/alloy_pf_directional/scripts/lumi_build_cpu.sh)
echo "build job \${BUILD_ID}"

PHYS="OPENPFC_ALCU_G=3.0e6,OPENPFC_ALCU_VP=0.4,OPENPFC_ALCU_LX=24.0e-6,OPENPFC_ALCU_LY=6.40e-6,OPENPFC_ALCU_TEND=240.0e-6,OPENPFC_ALCU_SEED=0.20e-6,OPENPFC_ALCU_NGRANS=2,OPENPFC_ALCU_THETA=30,OPENPFC_ALCU_NOISE=${F0},OPENPFC_ALCU_NOISE_SEED=1,OPENPFC_ALCU_WINDOW=0,OPENPFC_ALCU_STOP_FAR_C=1,OPENPFC_ALCU_STOP_RIGHT=0,OPENPFC_ALCU_PERIODIC_Y=1,OPENPFC_ALCU_SKIP_PNG=0,OPENPFC_ALCU_SKIP_VTK=1"

submit_ds() {
  local name="\$1" w0="\$2" save="\$3" time="\$4" cpus="\$5" jn="\$6"
  sbatch --parsable \\
    --job-name="\${jn}" \\
    --time="\${time}" \\
    --cpus-per-task="\${cpus}" \\
    --dependency=afterok:\${BUILD_ID} \\
    --export=ALL,CASE=\${name},W0=\${w0},DXW=1.0,SAVE_EVERY=\${save},OPENPFC_ALCU_DT_OVER_TAU=0.2,\${PHYS} \\
    apps/alloy_pf_directional/scripts/lumi_ds.sh
}

echo -n "W10nm "; submit_ds g3e6_V04_2x_noise_pm30_w0_10nm 10e-9 3000 06:00:00 32 alcu-2xn-w10
echo -n "W5nm "; submit_ds g3e6_V04_2x_noise_pm30_w0_5nm 5e-9 15000 12:00:00 64 alcu-2xn-w5
squeue -u \$USER -o '%.18i %.12P %.22j %.2t %.10M %.6D %R'
EOF
