#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Verify no deprecated symbols remain in include/ and src/ except A0 shim
# Exit code 1 on any match -> CI fails if violations exist
set -euo pipefail

# Check if include/ and src/ directories exist
if [ ! -d "include" ] && [ ! -d "src" ]; then
  echo "OK: No include/ or src/ directories to check"
  exit 0
fi

# Build grep arguments to avoid errors on missing directories
grep_args=()
[ -d "include" ] && grep_args+=("include/")
[ -d "src" ] && grep_args+=("src/")

if [ ${#grep_args[@]} -eq 0 ]; then
  echo "OK: No include/ or src/ directories to check"
  exit 0
fi

# Search for deprecated symbols, excluding A0 backward compatibility shim
# world_types.hpp is escaped as world_types\.hpp to match the literal filename
if grep -rnE 'Box3D|world_types\.hpp|CartesianTag' \
  --exclude='backward_compat.hpp' \
  --exclude-dir='v0' \
  "${grep_args[@]}"; then
  
  echo "ERROR: Found deprecated symbols in include/ or src/ - see output above"
  exit 1
else
  echo "OK: No deprecated symbols found in include/ or src/"
  exit 0
fi
