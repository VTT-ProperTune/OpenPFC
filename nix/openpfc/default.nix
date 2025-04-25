# nix/openpfc/default.nix

{ lib
, stdenv
, cmake
, mpi
, heffte
, nlohmann_json
, catch2_3 ? null
, doxygen ? null
, enableDocs ? false
, enableTests ? true
, enableExamples ? true
, enableApps ? true
, fetchFromGitHub
, versions
, version ? "0.1.1"
, src ? null }:

let
  inherit (versions.openpfc.${version}) rev sha256;

  realSrc = if src != null then src else fetchFromGitHub {
    owner = "VTT-ProperTune";
    repo = "OpenPFC";
    inherit (versions.openpfc.${version}) rev sha256;
  };
in

stdenv.mkDerivation {
  pname = "openpfc";
  inherit version;
   src = realSrc;

  meta = {
    description = "Phase Field Crystal simulation framework";
    license = lib.licenses.agpl3;
    platforms = lib.platforms.linux;
  };

  nativeBuildInputs = [ cmake ];

  buildInputs = [ mpi heffte nlohmann_json ]
    ++ lib.optional enableDocs doxygen
    ++ lib.optional enableTests catch2_3;

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DOpenPFC_BUILD_TESTS=${if enableTests then "ON" else "OFF"}"
    "-DOpenPFC_BUILD_EXAMPLES=${if enableExamples then "ON" else "OFF"}"
    "-DOpenPFC_BUILD_APPS=${if enableApps then "ON" else "OFF"}"
    "-DOpenPFC_BUILD_DOCUMENTATION=${if enableDocs then "ON" else "OFF"}"
    "-DHeffte_DIR=${heffte}/lib/cmake/Heffte"
  ];

}
