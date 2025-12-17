# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

{ lib, stdenv, ninja, cmake, git, mpi, heffte, tomlplusplus, nlohmann_json, catch2_3 ? null
, doxygen ? null, version, src, buildType ? "Release", enableDocs ? true
, enableTests ? true, enableExamples ? true, enableApps ? true }:

stdenv.mkDerivation {
  pname = "openpfc";
  inherit src version;

  meta = {
    description = "Phase Field Crystal simulation framework";
    license = lib.licenses.agpl3;
    platforms = lib.platforms.linux;
  };

  nativeBuildInputs = [ ninja cmake git ];

  buildInputs = [ mpi heffte tomlplusplus nlohmann_json ] ++ lib.optional enableDocs doxygen
    ++ lib.optional enableTests catch2_3;

  cmakeFlags = [
    "-GNinja"
    "-DCMAKE_BUILD_TYPE=${buildType}"
    "-DOpenPFC_BUILD_TESTS=${if enableTests then "ON" else "OFF"}"
    "-DOpenPFC_BUILD_EXAMPLES=${if enableExamples then "ON" else "OFF"}"
    "-DOpenPFC_BUILD_APPS=${if enableApps then "ON" else "OFF"}"
    "-DOpenPFC_BUILD_DOCUMENTATION=${if enableDocs then "ON" else "OFF"}"
    "-DOpenPFC_ENABLE_CODE_COVERAGE=OFF"
    "-DHeffte_DIR=${heffte}/lib/cmake/Heffte"
    "-Dtomlplusplus_DIR=${tomlplusplus}/lib/cmake/tomlplusplus"
  ];
}
