#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# From the laptop: sync, then submit the Karma 2001 present-model paper suite
# on LUMI-C. Reuses an already-running karma-build if present.
#
#   ./apps/alloy_pf_karma2001_benchmark/scripts/lumi_submit_paper.sh
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

BUILD_ID=\$(squeue -u \$USER -h -n karma-build -o '%i' | head -n1 || true)
if [[ -z "\${BUILD_ID}" ]]; then
  if [[ ! -x '${LUMI_KARMA_BIN}' ]]; then
    bash apps/alloy_pf_karma2001_benchmark/scripts/lumi_build_cpu.sh configure
    BUILD_ID=\$(sbatch --parsable apps/alloy_pf_karma2001_benchmark/scripts/lumi_build_cpu.sh)
    echo "build job \${BUILD_ID}"
  fi
else
  echo "reuse build job \${BUILD_ID}"
fi

DEP=()
if [[ -n "\${BUILD_ID:-}" ]]; then
  DEP=(--dependency=afterok:\${BUILD_ID})
fi

sbatch --parsable \\
  --job-name=karma-paper \\
  "\${DEP[@]}" \\
  apps/alloy_pf_karma2001_benchmark/scripts/lumi_paper.sh
squeue -u \$USER -o '%.18i %.12P %.22j %.8u %.2t %.10M %.6D %R' | head -n 30
EOF
