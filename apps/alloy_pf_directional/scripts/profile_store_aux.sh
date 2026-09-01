#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 2D OpenMP A/B: STORE_EU=0 (recompute u) vs default (store eu/u) vs STORE_AUX=1 (eu/u + fluxes).
# Laptop DS box, I/O off in the timed window. Does not submit to LUMI.
#
#   BUILD=builds/macos-cpu-release NTHREADS=8 ./apps/alloy_pf_directional/scripts/profile_store_aux.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD="${BUILD:-${OPENPFC_BUILD_DIR:-${ROOT}/builds/release}}"
OMP="${BUILD}/apps/alloy_pf_directional/alloy_pf_directional_openmp"
OUT="${ROOT}/results/alloy_pf_directional_profile"
NTHREADS="${NTHREADS:-8}"
mkdir -p "${OUT}/recompute" "${OUT}/store_eu" "${OUT}/store_aux"
if [[ ! -x "${OMP}" ]]; then
  echo "missing ${OMP}" >&2
  exit 1
fi
export OPENPFC_ALCU_SKIP_PNG=1 OPENPFC_ALCU_SKIP_VTK=1 OPENPFC_ALCU_QUIET=1
export OPENPFC_ALCU_NOISE=0 OPENPFC_ALCU_NGRANS=1 OPENPFC_ALCU_STOP_RIGHT=0
export OPENPFC_ALCU_NDIM=2 OPENPFC_ALCU_PERIODIC_Y=1
unset OPENPFC_ALCU_W0 OPENPFC_ALCU_DXW OPENPFC_ALCU_LX OPENPFC_ALCU_LY OPENPFC_ALCU_LZ || true
export OPENPFC_ALCU_WARMUP="${OPENPFC_ALCU_WARMUP:-15}"
export OPENPFC_ALCU_TIMED_STEPS="${OPENPFC_ALCU_TIMED_STEPS:-80}"
export OPENPFC_ALCU_MAX_STEPS="${OPENPFC_ALCU_MAX_STEPS:-100}"
echo "== B recompute STORE_EU=0 =="
OPENPFC_ALCU_STORE_EU=0 OPENPFC_ALCU_STORE_AUX=0 "${OMP}" ds "${OUT}/recompute" "${NTHREADS}" | tee "${OUT}/recompute.log"
echo "== A store eu/u (default) =="
OPENPFC_ALCU_STORE_EU=1 OPENPFC_ALCU_STORE_AUX=0 "${OMP}" ds "${OUT}/store_eu" "${NTHREADS}" | tee "${OUT}/store_eu.log"
echo "== C store aux eu/u + fluxes =="
OPENPFC_ALCU_STORE_AUX=1 "${OMP}" ds "${OUT}/store_aux" "${NTHREADS}" | tee "${OUT}/store_aux.log"
echo
grep ALCU_PERF "${OUT}/recompute.log" "${OUT}/store_eu.log" "${OUT}/store_aux.log" || true
