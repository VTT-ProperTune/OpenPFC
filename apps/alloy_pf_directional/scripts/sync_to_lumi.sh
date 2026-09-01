#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copy this OpenPFC working tree to LUMI projappl (includes uncommitted
# apps/alloy_pf_directional). Run from the laptop, not on LUMI.
#
#   ./apps/alloy_pf_directional/scripts/sync_to_lumi.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lumi_paths.sh
source "${HERE}/lumi_paths.sh"
ROOT="$(cd "${HERE}/../../.." && pwd)"

DEST="${LUMI_USER}@${LUMI_HOST}:${LUMI_SRC}/"

echo "rsync ${ROOT}/ -> ${DEST}"
ssh -o BatchMode=yes -o ConnectTimeout=20 "${LUMI_USER}@${LUMI_HOST}" \
  "mkdir -p '${LUMI_SRC}' '${LUMI_PROJAPPL}/opt' '${LUMI_PROJAPPL}/build' '${LUMI_PROJAPPL}/src' '${LUMI_RUNS}' '${LUMI_LOGS}'"

rsync -az --info=stats2 \
  --exclude '.git/' \
  --exclude 'builds/' \
  --exclude 'build/' \
  --exclude 'results/' \
  --exclude '.DS_Store' \
  --exclude '*.raw' \
  --exclude '.cursor/' \
  --exclude 'docs/html/' \
  --exclude 'docs/latex/' \
  "${ROOT}/" "${DEST}"

echo "synced to ${DEST}"
