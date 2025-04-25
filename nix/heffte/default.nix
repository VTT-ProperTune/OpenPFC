# nix/heffte/default.nix

{ lib
  , stdenv
  , cmake
  , fftw
  , fftwFloat
  , openmpi
  , fetchFromGitHub
  , version
  , src
}:

stdenv.mkDerivation {
  pname = "heffte";
  inherit src version;

  meta = {
    description = "Highly Efficient FFT for Exascale";
    license = lib.licenses.bsd3;
    platforms = lib.platforms.linux;
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
