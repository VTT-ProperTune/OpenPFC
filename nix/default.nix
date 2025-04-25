{ lib
, stdenv
, cmake
, git
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
, version ? "dev"
, src ? ./
}:

stdenv.mkDerivation rec {
  pname = "openpfc";
  inherit version;

  inherit src;

  nativeBuildInputs = [ cmake git ];
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

  doCheck = enableTests;

  meta = {
    description = "Phase Field Crystal simulation framework";
    homepage = "https://github.com/VTT-ProperTune/OpenPFC";
    license = lib.licenses.agpl3;
    platforms = lib.platforms.linux;
  };
}
