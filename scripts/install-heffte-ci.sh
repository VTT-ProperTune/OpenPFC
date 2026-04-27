#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <cmake-build-type> <temporary-build-dir>" >&2
  exit 2
fi

build_type=$1
build_dir=$2
heffte_version=2.4.1
archive="/tmp/heffte-v${heffte_version}.tar.gz"
source_dir="/tmp/heffte-${heffte_version}"
install_prefix="${HEFFTE_INSTALL_PREFIX:-$HOME/opt/heffte/${heffte_version}-cpu}"

cleanup() {
  rm -rf "${build_dir}" "${archive}" "${source_dir}"
}
trap cleanup EXIT

cleanup
wget -q -O "${archive}" \
  "https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${heffte_version}.tar.gz"
tar xzf "${archive}" -C /tmp

cmake -S "${source_dir}" -B "${build_dir}" \
  -GNinja \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DHeffte_ENABLE_FFTW=ON \
  -DBUILD_SHARED_LIBS=ON

cmake --build "${build_dir}" -j2
cmake --install "${build_dir}"
