# nix/heffte/default.nix

{ lib, stdenv, cmake, fftw, fftwFloat, openmpi, fetchFromGitHub, versions, version ? "2.4.1" }:

let
  inherit (versions.heffte.${version}) rev sha256;
in
stdenv.mkDerivation {
  pname = "heffte";
  inherit version;

  meta = {
    description = "Highly Efficient FFT for Exascale";
    license = lib.licenses.bsd3;
    platforms = lib.platforms.linux;
  };

  src = fetchFromGitHub {
    owner = "icl-utk-edu";
    repo = "heffte";
    inherit rev sha256;
  };

  nativeBuildInputs = [ cmake ];

  buildInputs = [ fftw fftwFloat openmpi ];

  cmakeFlags = [
    "-DHeffte_ENABLE_FFTW=ON"
    "-DHeffte_ENABLE_CUDA=OFF"
    "-DHeffte_ENABLE_ROCM=OFF"
    "-DHeffte_ENABLE_ONEAPI=OFF"
  ];
}
