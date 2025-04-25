# heffte.nix
{ stdenv, cmake, fftw, fftwFloat, openmpi, fetchFromGitHub }:

stdenv.mkDerivation {
  pname = "heffte";
  version = "2.4.1";

  src = fetchFromGitHub {
    owner = "icl-utk-edu";
    repo = "heffte";
    rev = "v2.4.1";
    sha256 = "3qCF3nsxjjj3OOks8f4Uu1L3+budRPX1i+iwXy8hLhE=";
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
