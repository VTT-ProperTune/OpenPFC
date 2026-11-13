#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Guard test for scripts/pre-commit-hook: in a throwaway repository that uses
# the tracked hook, a plain text file commits, while run output (*.bin),
# an editor backup (*.bak) and an oversized file are rejected.
# Exit code 1 on any deviation -> CI fails.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null HOME="$TMP"
export GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.invalid
export GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.invalid

REPO="$TMP/repo"
mkdir -p "$REPO/scripts" "$REPO/.githooks"
cp "$SRC/scripts/pre-commit-hook" "$REPO/scripts/pre-commit-hook"
cp "$SRC/.githooks/pre-commit" "$REPO/.githooks/pre-commit"
chmod +x "$REPO/scripts/pre-commit-hook" "$REPO/.githooks/pre-commit"
git -C "$REPO" init -q
git -C "$REPO" config core.hooksPath .githooks
git -C "$REPO" config commit.gpgsign false

failures=0

expect_commit() {  # <message> <expected: ok|reject> <stderr must contain> -- files...
  local msg="$1" expected="$2" needle="$3"; shift 3
  git -C "$REPO" add -- "$@"
  local err rc=0
  err="$(git -C "$REPO" commit -q -m "$msg" 2>&1 >/dev/null)" || rc=$?
  if [[ "$expected" == ok && "$rc" -ne 0 ]]; then
    echo "FAIL: '$msg' should commit, got rc=$rc: $err" >&2; failures=$((failures+1))
  elif [[ "$expected" == reject ]]; then
    if [[ "$rc" -eq 0 ]]; then
      echo "FAIL: '$msg' should be rejected but committed" >&2; failures=$((failures+1))
    elif [[ "$err" != *"$needle"* ]]; then
      echo "FAIL: '$msg' rejected without '$needle': $err" >&2; failures=$((failures+1))
    fi
    git -C "$REPO" reset -q -- "$@"
  fi
}

echo "note" > "$REPO/note.txt"
expect_commit "plain text file commits" ok "" note.txt

printf 'x' > "$REPO/u_0000.bin"
expect_commit "binary run output is rejected" reject "run output or editor backup" u_0000.bin

printf 'x' > "$REPO/field.hpp.bak"
expect_commit "editor backup is rejected" reject "run output or editor backup" field.hpp.bak

head -c 2048 /dev/zero > "$REPO/big.dat"
OPENPFC_PRECOMMIT_MAX_BYTES=1024 expect_commit "oversized file is rejected" reject "exceeds 1024 bytes" big.dat

OPENPFC_PRECOMMIT_MAX_BYTES=4096 expect_commit "size limit is adjustable" ok "" big.dat

if [[ "$failures" -ne 0 ]]; then
  echo "pre-commit guard: $failures check(s) failed" >&2
  exit 1
fi
echo "pre-commit guard: all checks passed"
