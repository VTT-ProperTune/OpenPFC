#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Verify no deprecated symbols remain in include/ and src/ except A0 shim
# Exit code 1 on any match -> CI fails if violations exist
set -euo pipefail

grep -rnE 'Box3D|world_types\.hpp|CartesianTag' \
  --exclude='backward_compat.hpp' \
  --exclude-dir='v0' \
  include/ src/ && \
  echo "ERROR: Found deprecated symbols in include/ or src/ - see output above" && exit 1 || \
  echo "OK: No deprecated symbols found in include/ or src/"
