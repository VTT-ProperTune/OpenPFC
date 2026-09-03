# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Karma 2001 LUMI-C layout (CPU OpenMP). Source from laptop rsync/sbatch
# wrappers or from the job script on LUMI.
#
# Scratch I/O stays under karma2001/.

LUMI_PROJECT="${LUMI_PROJECT:-project_462001519}"
LUMI_USER="${LUMI_USER:-tpinomaa}"
LUMI_HOST="${LUMI_HOST:-lumi.csc.fi}"

LUMI_PROJAPPL="${LUMI_PROJAPPL:-/projappl/${LUMI_PROJECT}/${LUMI_USER}}"
LUMI_SCRATCH="${LUMI_SCRATCH:-/scratch/${LUMI_PROJECT}/${LUMI_USER}}"

LUMI_KARMA_SRC="${LUMI_KARMA_SRC:-${LUMI_PROJAPPL}/src/OpenPFC}"
LUMI_KARMA_BUILD="${LUMI_KARMA_BUILD:-${LUMI_PROJAPPL}/build/openpfc-cpu}"
LUMI_KARMA_BIN="${LUMI_KARMA_BIN:-${LUMI_KARMA_BUILD}/apps/alloy_pf_karma2001_benchmark/alloy_pf_karma2001_benchmark_openmp}"
LUMI_KARMA_RUNS="${LUMI_KARMA_RUNS:-${LUMI_SCRATCH}/karma2001}"
LUMI_KARMA_LOGS="${LUMI_KARMA_LOGS:-${LUMI_KARMA_RUNS}/logs}"

# Reuse the Al-Cu FTA HeFFTe CPU prefix if present.
LUMI_HEFFTE_PREFIX="${LUMI_HEFFTE_PREFIX:-${LUMI_PROJAPPL}/opt/heffte/2.4.1-cpu}"
LUMI_STACK="${LUMI_STACK:-LUMI/25.09}"

lumi_cpu_runtime_env() {
  export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  local fab
  for fab in /opt/cray/libfabric/*/lib64; do
    if [[ -e "${fab}/libfabric.so.1" ]]; then
      export LD_LIBRARY_PATH="${fab}:${LD_LIBRARY_PATH}"
      break
    fi
  done
}
