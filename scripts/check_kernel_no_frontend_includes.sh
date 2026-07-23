#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Architecture layering enforcement (audit §11 / M0). Layers, top to bottom:
#
#   frontend  ->  runtime  ->  kernel
#
# A lower layer must never #include a higher one. This script enforces all three
# downward-only rules:
#   1. kernel   must not #include openpfc/frontend/...
#   2. kernel   must not #include openpfc/runtime/...   (kernel defines tags;
#      runtime injects specializations, so the dependency is runtime -> kernel)
#   3. runtime  must not #include openpfc/frontend/...
#
# Only real preprocessor directives count: the pattern anchors to the start of a
# line (optional leading whitespace), so guidance strings such as
#   static_assert(false, "... requires #include <openpfc/runtime/...>")
# are NOT flagged.
#
# Usage:
#   check_kernel_no_frontend_includes.sh             # run the checks
#   check_kernel_no_frontend_includes.sh --self-test # verify the checker detects
#                                                     # a deliberate violation
# See docs/concepts/architecture.md (Include audit) and docs/adr/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v rg >/dev/null 2>&1; then
  echo "check_kernel_no_frontend_includes: ripgrep (rg) not found; install ripgrep or skip in minimal environments."
  exit 0
fi

# scan <higher-layer-name> <dir...> -- print real "#include <openpfc/<higher>/...>"
# lines under the given directories (anchored to line start; ignores strings).
scan() {
  local higher="$1"; shift
  local pattern="^[[:space:]]*#[[:space:]]*include[[:space:]]*[<\"]openpfc/${higher}/"
  rg -n --no-heading -e "${pattern}" "$@" 2>/dev/null || true
}

run_checks() {
  local base="$1"
  local inc_kernel="${base}/include/openpfc/kernel"
  local src_kernel="${base}/src/openpfc/kernel"
  local inc_runtime="${base}/include/openpfc/runtime"
  local src_runtime="${base}/src/openpfc/runtime"
  local failed=0
  local m

  m="$(scan frontend "${inc_kernel}" "${src_kernel}")"
  if [[ -n "${m}" ]]; then
    echo "ERROR: kernel must not #include openpfc/frontend headers:"; echo "${m}"; failed=1
  fi

  m="$(scan runtime "${inc_kernel}" "${src_kernel}")"
  if [[ -n "${m}" ]]; then
    echo "ERROR: kernel must not #include openpfc/runtime headers (dependency is runtime -> kernel):"; echo "${m}"; failed=1
  fi

  m="$(scan frontend "${inc_runtime}" "${src_runtime}")"
  if [[ -n "${m}" ]]; then
    echo "ERROR: runtime must not #include openpfc/frontend headers:"; echo "${m}"; failed=1
  fi

  return "${failed}"
}

self_test() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp}"' RETURN
  mkdir -p "${tmp}/include/openpfc/kernel/data" \
           "${tmp}/include/openpfc/runtime/cuda" \
           "${tmp}/src/openpfc/kernel"

  # A clean kernel header (including a guidance STRING mentioning runtime) must pass.
  cat >"${tmp}/include/openpfc/kernel/data/clean.hpp" <<'EOF'
#pragma once
#include <vector>
// static_assert(false, "CudaTag requires #include <openpfc/runtime/cuda/x.hpp>");
EOF
  if [[ -n "$(run_checks "${tmp}" 2>/dev/null || true)" ]]; then
    echo "SELF-TEST FAILED: clean tree (with guidance string) was flagged."; return 1
  fi

  # A real violation must be detected.
  cat >"${tmp}/include/openpfc/kernel/data/bad.hpp" <<'EOF'
#pragma once
#include <openpfc/frontend/ui/app.hpp>
EOF
  if run_checks "${tmp}" >/dev/null 2>&1; then
    echo "SELF-TEST FAILED: a kernel->frontend violation was NOT detected."; return 1
  fi

  echo "SELF-TEST OK: clean tree passes, deliberate violation detected."
  return 0
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi

if run_checks "${ROOT}"; then
  echo "OK: layering respected (kernel !-> frontend/runtime; runtime !-> frontend)."
else
  exit 1
fi
