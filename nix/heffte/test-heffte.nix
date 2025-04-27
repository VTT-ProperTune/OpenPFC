# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

let
  pkgs = import <nixpkgs> { };

  heffte = pkgs.callPackage ./default.nix {
    cmake = pkgs.cmake;
    fftw = pkgs.fftw;
    fftwFloat = pkgs.fftwFloat;
    openmpi = pkgs.openmpi;
    fetchFromGitHub = pkgs.fetchFromGitHub;
  };
in heffte
