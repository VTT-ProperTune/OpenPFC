# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

{ lib, stdenv, cmake, mpi, heffte, nlohmann_json, catch2_3 ? null
, doxygen ? null, version, src, buildType ? "Release", enableDocs ? true
, enableTests ? true, enableExamples ? true, enableApps ? true }:

let
  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=${buildType}"
    "-DOpenPFC_BUILD_TESTS=${if enableTests then "ON" else "OFF"}"
    "-DOpenPFC_BUILD_EXAMPLES=${if enableExamples then "ON" else "OFF"}"
    "-DOpenPFC_BUILD_APPS=${if enableApps then "ON" else "OFF"}"
    "-DOpenPFC_BUILD_DOCUMENTATION=${if enableDocs then "ON" else "OFF"}"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" # Enable compile_commands.json generation for clang-tidy
    "-DHeffte_DIR=${heffte}/lib/cmake/Heffte"
  ];

  cmakeFlagsString = lib.concatStringsSep " \\\n  " cmakeFlags;

in stdenv.mkDerivation {
  pname = "openpfc";
  inherit src version;

  meta = {
    description = "Phase Field Crystal simulation framework";
    license = lib.licenses.agpl3;
    platforms = lib.platforms.linux;
  };

  # Tell Nix that we want multiple outputs
  outputs = [ "out" "builddir" "coverage" ];

  nativeBuildInputs = [ cmake ];

  buildInputs = [ mpi heffte nlohmann_json ] ++ lib.optional enableDocs doxygen
    ++ lib.optional enableTests catch2_3;

  buildPhase = ''
    cmake -S $src -B build ${cmakeFlagsString}
    cmake --build build
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out
    mkdir -p $builddir
    mkdir -p $coverage

    # Install compiled outputs
    cmake --install build --prefix=$out

    # Copy build artifacts
    cp -r build/* $builddir/

    # Dummy coverage for now
    # echo "Coverage not generated yet" > $coverageOutput/README.txt

    runHook postInstall
  '';

}
