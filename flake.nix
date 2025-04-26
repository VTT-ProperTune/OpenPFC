# flake.nix
# To list all variations, use the following command:
# nix flake show | grep "openpfcVariations"

{
  description = "OpenPFC and HeFFTe builder";

  # Inputs define external dependencies for this flake.
  inputs.nixpkgs.url =
    "github:NixOS/nixpkgs/nixos-23.11"; # Nixpkgs repository for system packages.
  inputs.flake-utils.url =
    "github:numtide/flake-utils"; # Utility library for flakes.

  # Outputs define what this flake provides.
  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Import the Nixpkgs package set for the current system.
        pkgs = import nixpkgs { inherit system; };

        # Define paths and versions for HeFFTe.
        hefftePath = ./nix/heffte/default.nix;
        heffteVersions =
          builtins.fromJSON (builtins.readFile ./nix/heffte/versions.json);
        heffteVersion = heffteVersions.current;

        # Define paths and versions for OpenPFC.
        openpfcPath = ./nix/openpfc/default.nix;
        openpfcVersions =
          builtins.fromJSON (builtins.readFile ./nix/openpfc/versions.json);
        openpfcVersion = openpfcVersions.current;

        heffte = pkgs.callPackage hefftePath {
          version = heffteVersion;
          src = pkgs.fetchFromGitHub {
            owner = "icl-utk-edu";
            repo = "heffte";
            inherit (heffteVersions.versions.${heffteVersion}) rev sha256;
          };
        };

      in {
        # Combine dynamically generated OpenPFC packages with their dependencies.
        packages = {
          # openpfc: Fetches the current version of OpenPFC.
          openpfc = pkgs.callPackage openpfcPath {
            version = openpfcVersion;
            buildType = "Release";
            src = pkgs.fetchFromGitHub {
              owner = "VTT-ProperTune";
              repo = "OpenPFC";
              inherit (openpfcVersions.versions.${openpfcVersion}) rev sha256;
            };
            enableTests = true;
            enableDocs = true;
            enableExamples = true;
            enableApps = true;
            heffte = heffte;
          };

          # openpfc-dev: Uses the local source directory (./).
          openpfc-dev = pkgs.callPackage openpfcPath {
            version = "dev";
            buildType = "Debug";
            src = ./.;
            enableTests = true;
            enableDocs = true;
            enableExamples = true;
            enableApps = true;
            heffte = heffte;
          };

          # openpfc-tests: Uses the local source directory (./).
          openpfc-tests = pkgs.callPackage openpfcPath {
            version = "dev";
            buildType = "Debug";
            src = ./.;
            enableTests = true;
            enableDocs = false;
            enableExamples = false;
            enableApps = false;
            heffte = heffte;
          };
        };

        # Development shell configuration.
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
      });
}
