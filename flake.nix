# flake.nix

{
  description = "OpenPFC and HeFFTe builder";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let

        pkgs = import nixpkgs { inherit system; };
        versions = import ./nix/versions.nix;
        hefftePath = ./nix/heffte/default.nix;
        openpfcPath = ./nix/openpfc/default.nix;

      in
      {

        packages = rec {
          # HeFFTe version 2.4.1 from GitHub
          heffte = pkgs.callPackage hefftePath {
            inherit versions;
            version = "2.4.1";
          };

          # OpenPFC versioned release (e.g., 0.1.1)
          openpfc = pkgs.callPackage openpfcPath {
            inherit versions;
            version = "0.1.1";
            heffte = self.packages.${system}.heffte;
          };

          # OpenPFC from local checkout (your dev version)
          openpfc-dev = pkgs.callPackage openpfcPath {
            inherit versions;
            version = "dev";
            src = ./.;
            heffte = self.packages.${system}.heffte;
          };

        default = openpfc-dev;

        };

        devShells.default = pkgs.mkShell {

          packages = [
            pkgs.cmake
            pkgs.git
            pkgs.openmpi
            pkgs.nlohmann_json
            pkgs.doxygen
            pkgs.catch2_3
          ];

          inputsFrom = [
            self.packages.${system}.heffte
            self.packages.${system}.openpfc-dev
          ];

          shellHook = ''
            export Heffte_DIR=${self.packages.${system}.heffte}/lib/cmake/Heffte
            echo "🎉 Welcome to the OpenPFC development shell!"
            echo "👉 To configure the project, run: cmake -S . -B build"
            echo "👉 To build the project, run: cmake --build build"
          '';

        };
      }
    );
}
