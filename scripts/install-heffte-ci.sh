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
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
heffte_version=2.4.1
# Match the immutable source and correction used by cmake/FetchHeffte.cmake.
heffte_revision=6e16996e5a3eb65c60fd62fd4d93c4b2df833033
archive_sha256=543cc06336140d6eddbc7a0955fe795bf400440578131121fb7b03ca4922fba6
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/openpfc-heffte-ci.XXXXXX")"
archive="${work_dir}/heffte.tar.gz"
source_dir="${work_dir}/heffte-${heffte_revision}"
install_prefix="${HEFFTE_INSTALL_PREFIX:-$HOME/opt/heffte/${heffte_version}-cpu}"

cleanup() {
  rm -rf "${build_dir}" "${work_dir}"
}
trap cleanup EXIT

rm -rf "${build_dir}"
wget -q -O "${archive}" \
  "https://codeload.github.com/icl-utk-edu/heffte/tar.gz/${heffte_revision}"
printf '%s  %s\n' "${archive_sha256}" "${archive}" | sha256sum --check -
tar xzf "${archive}" -C "${work_dir}"
cmake "-DSOURCE_DIR=${source_dir}" -P "${script_dir}/../cmake/PatchHeffteFftw.cmake"

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
