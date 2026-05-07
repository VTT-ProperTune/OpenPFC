#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Summarize an A/B nsys run of `kobayashi_fd_cuda` with the **batched halo**
# (KOBAYASHI_HALO_BATCH=1) vs the **per-field baseline** (=0), comparing
# `KOBAYASHI_PERF_LOOP` and `OPENPFC_CUDA_PROFILE_HALO_SUMMARY` lines from the
# two Slurm `.out` files.
#
# Usage:
#   apps/kobayashi/slurm/summarize_halo_batch_ab.sh <off_jobid> <on_jobid>
#
# Reads `kobayashi_fd_cuda_h100_np2_nsys_<jobid>.out` for both runs (must be in
# the current directory).

set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <off_jobid> <on_jobid>" >&2
  exit 2
fi
OFF="kobayashi_fd_cuda_h100_np2_nsys_$1.out"
ON="kobayashi_fd_cuda_h100_np2_nsys_$2.out"
for f in "$OFF" "$ON"; do
  if [[ ! -f "$f" ]]; then
    echo "missing: $f" >&2
    exit 1
  fi
done

extract() {
  local f="$1" tag="$2"
  grep -E "^${tag}" "$f" | head -1 || true
}

echo "=== KOBAYASHI_PERF_LOOP ==="
echo "[off] $(extract "$OFF" KOBAYASHI_PERF_LOOP)"
echo "[on ] $(extract "$ON"  KOBAYASHI_PERF_LOOP)"

echo ""
echo "=== OPENPFC_CUDA_PROFILE_HALO_SUMMARY ==="
echo "[off] $(extract "$OFF" OPENPFC_CUDA_PROFILE_HALO_SUMMARY)"
echo "[on ] $(extract "$ON"  OPENPFC_CUDA_PROFILE_HALO_SUMMARY)"

echo ""
echo "=== KOBAYASHI_VERIFY_HEX (must match between off/on) ==="
echo "[off] $(extract "$OFF" KOBAYASHI_VERIFY_HEX)"
echo "[on ] $(extract "$ON"  KOBAYASHI_VERIFY_HEX)"

echo ""
echo "=== KOBAYASHI_CUDA_HALO_MODE ==="
echo "[off] $(extract "$OFF" KOBAYASHI_CUDA_HALO_MODE)"
echo "[on ] $(extract "$ON"  KOBAYASHI_CUDA_HALO_MODE)"
