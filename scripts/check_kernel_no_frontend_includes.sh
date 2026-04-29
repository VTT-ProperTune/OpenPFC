#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Fail if kernel sources include openpfc/frontend headers (layering rule).
# See docs/architecture.md (Include audit).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INC_KERNEL="$ROOT/include/openpfc/kernel"
SRC_KERNEL="$ROOT/src/openpfc/kernel"

if ! command -v rg >/dev/null 2>&1; then
  echo "check_kernel_no_frontend_includes: ripgrep (rg) not found; install ripgrep or skip in minimal environments."
  exit 0
fi

matches="$(rg -n --no-heading '#include\s*[<"]openpfc/frontend/' "$INC_KERNEL" "$SRC_KERNEL" 2>/dev/null || true)"
if [[ -n "${matches}" ]]; then
  echo "ERROR: kernel tree must not #include openpfc/frontend headers."
  echo "${matches}"
  exit 1
fi

echo "OK: no openpfc/frontend includes under include/openpfc/kernel or src/openpfc/kernel"
