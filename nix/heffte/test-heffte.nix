# test-heffte.nix
let
  pkgs = import <nixpkgs> {};

  heffte = pkgs.callPackage ./default.nix {
    cmake = pkgs.cmake;
    fftw = pkgs.fftw;
    fftwFloat = pkgs.fftwFloat;
    openmpi = pkgs.openmpi;
    fetchFromGitHub = pkgs.fetchFromGitHub;
  };
in
heffte
