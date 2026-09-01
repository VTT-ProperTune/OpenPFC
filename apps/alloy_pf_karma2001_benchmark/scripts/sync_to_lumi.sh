#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Rsync this OpenPFC tree to LUMI projappl. Run from the laptop.
#
#   ./apps/alloy_pf_karma2001_benchmark/scripts/sync_to_lumi.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lumi_paths.sh
source "${HERE}/lumi_paths.sh"
ROOT="$(cd "${HERE}/../../.." && pwd)"
DEST="${LUMI_USER}@${LUMI_HOST}:${LUMI_KARMA_SRC}/"

echo "rsync ${ROOT}/ -> ${DEST}"
ssh -o BatchMode=yes -o ConnectTimeout=25 "${LUMI_USER}@${LUMI_HOST}" \
  "mkdir -p '${LUMI_KARMA_SRC}' '${LUMI_KARMA_BUILD}' '${LUMI_KARMA_RUNS}' '${LUMI_KARMA_LOGS}'"

rsync -az --info=stats2 \
  --exclude '.git/' \
  --exclude 'builds/' \
  --exclude 'build/' \
  --exclude 'results/' \
  --exclude '.DS_Store' \
  --exclude '.cursor/' \
  --exclude 'docs/html/' \
  --exclude 'docs/latex/' \
  "${ROOT}/" "${DEST}"

echo "synced to ${DEST}"
