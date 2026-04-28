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

# When OPENPFC_CI_HEFFTE_ADDRESS_SANITIZER=1, build HeFFTe with AddressSanitizer
# so it matches OpenPFC configured with -DOpenPFC_ENABLE_ADDRESS_SANITIZER=ON.
heffte_extra_cmake_flags=()
if [[ "${OPENPFC_CI_HEFFTE_ADDRESS_SANITIZER:-0}" == "1" ]]; then
  heffte_extra_cmake_flags+=(
    "-DCMAKE_C_FLAGS=-fsanitize=address -fno-omit-frame-pointer -g"
    "-DCMAKE_CXX_FLAGS=-fsanitize=address -fno-omit-frame-pointer -g"
    "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address"
    "-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address"
  )
fi

cmake -S "${source_dir}" -B "${build_dir}" \
  -GNinja \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DHeffte_ENABLE_FFTW=ON \
  -DBUILD_SHARED_LIBS=ON \
  "${heffte_extra_cmake_flags[@]}"

cmake --build "${build_dir}" -j2
cmake --install "${build_dir}"
