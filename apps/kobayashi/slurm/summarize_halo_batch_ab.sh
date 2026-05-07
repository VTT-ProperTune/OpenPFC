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
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <off_jobid> <on_jobid> [<extended_jobid> [<extra_jobid> ...]]" >&2
  exit 2
fi

extract() {
  local f="$1" tag="$2"
  grep -E "^${tag}" "$f" | head -1 || true
}

declare -a TAGS=( "off" "on" "ext" )
declare -a FILES=()
declare -a LABELS=()
i=0
for jid in "$@"; do
  f="kobayashi_fd_cuda_h100_np2_nsys_${jid}.out"
  if [[ ! -f "$f" ]]; then
    echo "missing: $f" >&2
    exit 1
  fi
  FILES+=( "$f" )
  if [[ $i -lt ${#TAGS[@]} ]]; then
    LABELS+=( "${TAGS[$i]}" )
  else
    LABELS+=( "j${jid}" )
  fi
  ((i++))
done

print_block() {
  local title="$1" tag="$2"
  echo "=== ${title} ==="
  for k in "${!FILES[@]}"; do
    printf "[%s] %s\n" "${LABELS[$k]}" "$(extract "${FILES[$k]}" "${tag}")"
  done
  echo ""
}

print_block "KOBAYASHI_CUDA_HALO_MODE"            "KOBAYASHI_CUDA_HALO_MODE"
print_block "KOBAYASHI_PERF_LOOP"                 "KOBAYASHI_PERF_LOOP"
print_block "OPENPFC_CUDA_PROFILE_HALO_SUMMARY"   "OPENPFC_CUDA_PROFILE_HALO_SUMMARY"
print_block "KOBAYASHI_VERIFY_HEX (should match for non-boundary-activating runs)" "KOBAYASHI_VERIFY_HEX"
