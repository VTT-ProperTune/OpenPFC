#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# From the laptop: sync OpenPFC, rebuild alloy_pf_directional_openmp on LUMI-C, then
# submit static-box G=3e6 K/m, Vp=0.4 m/s cases at W0 = 5 nm and 2.5 nm.
# No moving window. Stop when any right-face liquid pixel leaves c_∞.
#
#   ./apps/alloy_pf_directional/scripts/lumi_submit_g3e6_v04.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lumi_paths.sh
source "${HERE}/lumi_paths.sh"

"${HERE}/sync_to_lumi.sh"

REMOTE="${LUMI_USER}@${LUMI_HOST}"
SRC="${LUMI_SRC}"

ssh -o BatchMode=yes -o ConnectTimeout=25 "${REMOTE}" bash -s <<EOF
set -euo pipefail
cd '${SRC}'
bash apps/alloy_pf_directional/scripts/lumi_build_cpu.sh configure
BUILD_ID=\$(sbatch --parsable apps/alloy_pf_directional/scripts/lumi_build_cpu.sh)
echo "build job \${BUILD_ID}"

PHYS="OPENPFC_ALCU_G=3.0e6,OPENPFC_ALCU_VP=0.4,OPENPFC_ALCU_LX=12.0e-6,OPENPFC_ALCU_LY=0.80e-6,OPENPFC_ALCU_TEND=120.0e-6,OPENPFC_ALCU_SEED=0.20e-6,OPENPFC_ALCU_NGRANS=1,OPENPFC_ALCU_NOISE=0,OPENPFC_ALCU_WINDOW=0,OPENPFC_ALCU_STOP_FAR_C=1,OPENPFC_ALCU_STOP_RIGHT=0,OPENPFC_ALCU_PERIODIC_Y=1,OPENPFC_ALCU_SKIP_PNG=1,OPENPFC_ALCU_SKIP_VTK=1"

submit_ds() {
  local name="\$1" w0="\$2" save="\$3" time="\$4" cpus="\$5" dtr="\$6"
  sbatch --parsable \\
    --job-name="alcu-\${name}" \\
    --time="\${time}" \\
    --cpus-per-task="\${cpus}" \\
    --dependency=afterok:\${BUILD_ID} \\
    --export=ALL,CASE=\${name},W0=\${w0},DXW=1.0,SAVE_EVERY=\${save},OPENPFC_ALCU_DT_OVER_TAU=\${dtr},\${PHYS} \\
    apps/alloy_pf_directional/scripts/lumi_ds.sh
}

echo -n "W5nm "; submit_ds g3e6_V04_w0_5nm 5e-9 20000 08:00:00 16 0.1
echo -n "W2.5nm_dt0.1 "; submit_ds g3e6_V04_w0_2.5nm_dt0.1 2.5e-9 25000 72:00:00 32 0.1
echo -n "W2.5nm_dt0.2 "; submit_ds g3e6_V04_w0_2.5nm_dt0.2 2.5e-9 12500 48:00:00 32 0.2
squeue -u \$USER -o '%.18i %.12P %.22j %.8u %.2t %.10M %.6D %R' | head -n 20
EOF
