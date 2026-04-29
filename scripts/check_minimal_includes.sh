#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Optional guard: kernel unit tests should not pull the umbrella openpfc.hpp.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
fail=0
while IFS= read -r -d '' f; do
  if grep -qE '#include\s*[<"]openpfc/openpfc\.hpp[>"]' "$f"; then
    echo "check_minimal_includes: avoid umbrella include in $f" >&2
    fail=1
  fi
done < <(find tests/unit/kernel -name '*.cpp' -print0 2>/dev/null)

exit "$fail"
