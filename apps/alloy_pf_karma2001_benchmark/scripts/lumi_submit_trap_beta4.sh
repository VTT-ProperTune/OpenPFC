#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Re-run isothermal trap at β₀=4 s/m, Δt=0.1 τ₀ (W0=10 nm and 5 nm).
# Does not touch the running no-trap 5 nm job.
#
#   ./apps/alloy_pf_karma2001_benchmark/scripts/lumi_submit_trap_beta4.sh
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

submit_gl() {
  local d0w="\$1" time="\$2" cpus="\$3" name="\$4"
  local out="${LUMI_KARMA_RUNS}/trap_eq7_b4/d0W_\${d0w}_th45"
  sbatch --parsable \\
    --job-name="\${name}" \\
    --time="\${time}" \\
    --cpus-per-task="\${cpus}" \\
    --export=ALL,KARMA_D0W=\${d0w},KARMA_TRAP=1,KARMA_DT=0.1,KARMA_BETA0=4,KARMA_OUT=\${out} \\
    apps/alloy_pf_karma2001_benchmark/scripts/lumi_glasner_w0.sh
}

echo -n "W10 trap b4 "; submit_gl 1.217 02:00:00 16 karma-b4-W10
echo -n "W5 trap b4 "; submit_gl 2.434 08:00:00 32 karma-b4-W5
squeue -u \$USER -o '%.18i %.12P %.22j %.8u %.2t %.10M %.6D %R' | head -n 20
EOF
