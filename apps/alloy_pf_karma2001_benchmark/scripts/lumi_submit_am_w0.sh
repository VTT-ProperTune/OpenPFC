#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# From the laptop: sync OpenPFC, configure+build alloy_pf_karma2001_benchmark_openmp on LUMI-C,
# then submit decaying AM cooling jobs at W0 = 10, 5, 2.5 nm.
#
#   ./apps/alloy_pf_karma2001_benchmark/scripts/lumi_submit_am_w0.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lumi_paths.sh
source "${HERE}/lumi_paths.sh"

"${HERE}/sync_to_lumi.sh"

REMOTE="${LUMI_USER}@${LUMI_HOST}"
SRC="${LUMI_KARMA_SRC}"

ssh -o BatchMode=yes -o ConnectTimeout=25 "${REMOTE}" bash -s <<EOF
set -euo pipefail
cd '${SRC}'
bash apps/alloy_pf_karma2001_benchmark/scripts/lumi_build_cpu.sh configure
BUILD_ID=\$(sbatch --parsable apps/alloy_pf_karma2001_benchmark/scripts/lumi_build_cpu.sh)
echo "build job \${BUILD_ID}"

submit_am() {
  local w="\$1" time="\$2" cpus="\$3"
  local out="${LUMI_KARMA_RUNS}/tau12_eq7/W\${w}nm_th45"
  sbatch --parsable \\
    --job-name="karma-am-W\${w}" \\
    --time="\${time}" \\
    --cpus-per-task="\${cpus}" \\
    --dependency=afterok:\${BUILD_ID} \\
    --export=ALL,KARMA_W0=\${w},KARMA_OUT=\${out} \\
    apps/alloy_pf_karma2001_benchmark/scripts/lumi_am_w0.sh
}

echo -n "W10 "; submit_am 10 04:00:00 16
echo -n "W5 "; submit_am 5 12:00:00 32
echo -n "W2.5 "; submit_am 2.5 48:00:00 32
squeue -u \$USER -o '%.18i %.12P %.22j %.8u %.2t %.10M %.6D %R' | head -n 20
EOF
