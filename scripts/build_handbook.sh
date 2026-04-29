#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Optional: concatenate docs from docs/handbook_manifest.txt and run pandoc.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUTDIR="${ROOT}/build-handbook"
MD="${OUTDIR}/openpfc-handbook.md"
MANIFEST="${ROOT}/docs/handbook_manifest.txt"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "build_handbook: pandoc not found; install pandoc to generate the handbook." >&2
  exit 0
fi

mkdir -p "${OUTDIR}"
: >"${MD}"

while IFS= read -r line || [[ -n "${line}" ]]; do
  [[ -z "${line}" || "${line}" =~ ^# ]] && continue
  f="${ROOT}/${line}"
  if [[ ! -f "${f}" ]]; then
    echo "build_handbook: skip missing ${line}" >&2
    continue
  fi
  {
    echo ""
    echo "<!-- ${line} -->"
    echo ""
    cat "${f}"
  } >>"${MD}"
done <"${MANIFEST}"

if pandoc "${MD}" -o "${OUTDIR}/openpfc-handbook.pdf" --pdf-engine=pdflatex -V geometry:margin=1in 2>/dev/null; then
  echo "build_handbook: wrote ${OUTDIR}/openpfc-handbook.pdf"
else
  pandoc "${MD}" -o "${OUTDIR}/openpfc-handbook.html" -s
  echo "build_handbook: PDF engine unavailable; wrote ${OUTDIR}/openpfc-handbook.html"
fi
