{
  description = "OpenPFC built with Nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # HeFFTe for development (master branch)
        heffteDev = pkgs.stdenv.mkDerivation {
          pname = "heffte";
          version = "dev";
          src = pkgs.fetchFromGitHub {
            owner = "icl-utk-edu";
            repo = "heffte";
            rev = "master";
            sha256 = null; # Allow Nix to compute the hash dynamically
          };
          nativeBuildInputs = [ pkgs.cmake pkgs.openmpi pkgs.fftw pkgs.fftwFloat ];
        };

        # HeFFTe for releases (specific versions)
        heffteRelease = pkgs.callPackage ./nix/heffte.nix {
          fftw = pkgs.fftw;
          fftwFloat = pkgs.fftwFloat;
          openmpi = pkgs.openmpi;
          fetchFromGitHub = pkgs.fetchFromGitHub;
          cmake = pkgs.cmake;
        };

      in {
        # Development environment
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.openmpi
            pkgs.gcc
            pkgs.fftw
            pkgs.fftwFloat
            pkgs.nlohmann_json
            heffteDev
          ];
          src = ./; # Use the local source for OpenPFC during development
        };

        # Release builds
        packages.default = pkgs.callPackage ./nix/default.nix {
          heffte = heffteRelease;
          enableDocs = true;
          enableTests = true;
          catch2_3 = pkgs.catch2_3;
        };
      });
}
