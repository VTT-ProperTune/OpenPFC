# flake.nix

{
  description = "OpenPFC and HeFFTe builder";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let

        pkgs = import nixpkgs { inherit system; };

        hefftePath = ./nix/heffte/default.nix;
        heffteVersions = builtins.fromJSON (builtins.readFile ./nix/heffte/versions.json);
        heffteVersion = heffteVersions.current;

        openpfcPath = ./nix/openpfc/default.nix;
        openpfcVersions = builtins.fromJSON (builtins.readFile ./nix/openpfc/versions.json);
        openpfcVersion = openpfcVersions.current;

      in

      {

        packages = rec {

          heffte = pkgs.callPackage hefftePath {
            version = heffteVersion;
            src = pkgs.fetchFromGitHub {
              owner = "icl-utk-edu";
              repo = "heffte";
              inherit (heffteVersions.versions.${heffteVersion}) rev sha256;
            };
          };

          # OpenPFC versioned release
          openpfc = pkgs.callPackage openpfcPath {
            version = openpfcVersion;
            src = pkgs.fetchFromGitHub {
              owner = "VTT-ProperTune";
              repo = "OpenPFC";
              inherit (openpfcVersions.versions.${openpfcVersion}) rev sha256;
            };
            heffte = self.packages.${system}.heffte;
          };

          # OpenPFC from local checkout (your dev version)
          openpfc-dev = pkgs.callPackage openpfcPath {
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
