#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:-build}
DOXYGEN_BUILD_DIR=${OPENPFC_DOXYGEN_BUILD_DIR:-"${ROOT}/build/docs"}
SITE_DIR=${OPENPFC_DOCS_SITE_DIR:-"${ROOT}/site"}
SPHINX_LOG=${OPENPFC_SPHINX_LOG:-"${DOXYGEN_BUILD_DIR}/sphinx.log"}

mkdir -p "${DOXYGEN_BUILD_DIR}"
cmake -S "${ROOT}/docs" -B "${DOXYGEN_BUILD_DIR}" -GNinja
cmake --build "${DOXYGEN_BUILD_DIR}" --target openpfc-doxygen-xml

export OPENPFC_DOXYGEN_XML="${DOXYGEN_BUILD_DIR}/xml"
export OPENPFC_REPOSITORY=${OPENPFC_REPOSITORY:-VTT-ProperTune/OpenPFC}
export OPENPFC_REVISION=${OPENPFC_REVISION:-master}

case "${MODE}" in
  build)
    rm -f "${SPHINX_LOG}"
    set +e
    uv run --project "${ROOT}/docs" --frozen \
      sphinx-build --keep-going -b dirhtml \
      "${ROOT}/docs" "${SITE_DIR}" 2>&1 | tee "${SPHINX_LOG}"
    sphinx_status=${PIPESTATUS[0]}
    set -e

    if [[ ${sphinx_status} -ne 0 ]]; then
      exit "${sphinx_status}"
    fi

    python3 "${ROOT}/scripts/check_sphinx_log.py" "${SPHINX_LOG}"
    ;;
  serve)
    exec uv run --project "${ROOT}/docs" --frozen \
      sphinx-autobuild -b dirhtml \
      "${ROOT}/docs" "${SITE_DIR}"
    ;;
  *)
    echo "usage: $0 [build|serve]" >&2
    exit 2
    ;;
esac
