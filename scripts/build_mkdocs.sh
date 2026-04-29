#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Wrapper: suppress Material’s optional MkDocs 2.0 banner (see material/templates/__init__.py).
# Usage (from repository root): ./scripts/build_mkdocs.sh build --strict
set -euo pipefail
export NO_MKDOCS_2_WARNING=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"
exec uv run --project docs mkdocs "$@"
