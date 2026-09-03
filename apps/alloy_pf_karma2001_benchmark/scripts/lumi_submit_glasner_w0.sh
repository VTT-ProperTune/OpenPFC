#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# From the laptop: sync, then submit extra isothermal trapping W0=10 nm and 5 nm
# (trap + no-trap). Not the PRL paper suite — that is lumi_submit_paper.sh.
#
#   ./apps/alloy_pf_karma2001_benchmark/scripts/lumi_submit_glasner_w0.sh
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

submit_gl() {
  local d0w="\$1" trap="\$2" time="\$3" cpus="\$4" name="\$5"
  local tag=""
  [[ "\${trap}" == "0" ]] && tag="_notrap"
  local out="${LUMI_KARMA_RUNS}/trap_eq7/d0W_\${d0w}_th45\${tag}"
  sbatch --parsable \\
    --job-name="\${name}" \\
    --time="\${time}" \\
    --cpus-per-task="\${cpus}" \\
    "\${DEP[@]}" \\
    --export=ALL,KARMA_D0W=\${d0w},KARMA_TRAP=\${trap},KARMA_PHI1=45,KARMA_OUT=\${out} \\
    apps/alloy_pf_karma2001_benchmark/scripts/lumi_glasner_w0.sh
}

# d0=12.17 nm → W0=10 nm is d0/W=1.217; W0=5 nm is d0/W=2.434
echo -n "W10 trap "; submit_gl 1.217 1 04:00:00 16 karma-gl-W10
echo -n "W10 notrap "; submit_gl 1.217 0 08:00:00 32 karma-gl-W10n
echo -n "W5 trap "; submit_gl 2.434 1 12:00:00 32 karma-gl-W5
echo -n "W5 notrap "; submit_gl 2.434 0 24:00:00 32 karma-gl-W5n
squeue -u \$USER -o '%.18i %.12P %.22j %.8u %.2t %.10M %.6D %R' | head -n 30
EOF
