#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# M3 single-source GPU runtime: native cudaMemcpy / hipMemcpy in include/ and
# src/ must live under runtime/gpu/. Vendor trees are thin includes or
# re-exports (plus FFT until M5) and must not grow their own memcpy calls.
#
# Matches the DoD grep:
#   grep -rn "hipMemcpy\|cudaMemcpy" include/ src/ | grep -v runtime/gpu
#
# Usage:
#   check_gpu_memcpy_single_source.sh             # run the check
#   check_gpu_memcpy_single_source.sh --self-test # verify the checker

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v rg >/dev/null 2>&1; then
  echo "check_gpu_memcpy_single_source: ripgrep (rg) not found; install ripgrep or skip in minimal environments."
  exit 0
fi

# Print hipMemcpy/cudaMemcpy hits under include/ and src/ that are not in
# runtime/gpu/. Paths are relative to the scan root.
scan() {
  local base="$1"
  local hits
  hits="$(
    rg -n --no-heading -e 'hipMemcpy|cudaMemcpy' \
      "${base}/include" "${base}/src" 2>/dev/null || true
  )"
  if [[ -z "${hits}" ]]; then
    return 0
  fi
  # Allow include/openpfc/runtime/gpu/ and src/openpfc/runtime/gpu/.
  printf '%s\n' "${hits}" | rg -v '/runtime/gpu/' || true
}

run_checks() {
  local base="$1"
  local m
  m="$(scan "${base}")"
  if [[ -n "${m}" ]]; then
    echo "ERROR: cudaMemcpy/hipMemcpy outside runtime/gpu/ (include/ and src/):"
    echo "${m}"
    return 1
  fi
  return 0
}

self_test() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp}"' RETURN
  mkdir -p "${tmp}/include/openpfc/runtime/gpu" \
           "${tmp}/include/openpfc/kernel/data" \
           "${tmp}/src/openpfc/runtime/gpu" \
           "${tmp}/src/openpfc/runtime/cuda"

  cat >"${tmp}/include/openpfc/runtime/gpu/ok.hpp" <<'EOF'
#pragma once
inline void copy(void *d, const void *s, std::size_t n) {
  cudaMemcpy(d, s, n, cudaMemcpyHostToDevice);
}
EOF
  cat >"${tmp}/src/openpfc/runtime/gpu/ok.inc" <<'EOF'
hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
EOF
  if ! run_checks "${tmp}" >/dev/null; then
    echo "SELF-TEST FAILED: allowed runtime/gpu memcpy was flagged."; return 1
  fi

  cat >"${tmp}/include/openpfc/kernel/data/bad.hpp" <<'EOF'
#pragma once
inline void leak() { cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyDefault); }
EOF
  if run_checks "${tmp}" >/dev/null 2>&1; then
    echo "SELF-TEST FAILED: kernel cudaMemcpy was NOT detected."; return 1
  fi
  rm -f "${tmp}/include/openpfc/kernel/data/bad.hpp"

  cat >"${tmp}/src/openpfc/runtime/cuda/bad.cpp" <<'EOF'
#include <cuda_runtime.h>
void leak() { cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyDefault); }
EOF
  if run_checks "${tmp}" >/dev/null 2>&1; then
    echo "SELF-TEST FAILED: vendor-tree cudaMemcpy was NOT detected."; return 1
  fi

  echo "SELF-TEST OK: runtime/gpu memcpy allowed; leaks outside it detected."
  return 0
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi

if run_checks "${ROOT}"; then
  echo "OK: cudaMemcpy/hipMemcpy in include/ and src/ stay under runtime/gpu/."
else
  exit 1
fi
