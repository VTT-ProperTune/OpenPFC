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

        ### Packages ###

        packages = {

          # heffte: Fetches the current version of HeFFTe.
          heffte = heffte;

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

        ### Applications ###

        apps = {

          openpfc-tests = {
            type = "app";
            program =
              "${self.packages.${system}.openpfc-tests}/bin/openpfc-tests";
            meta = with pkgs.lib; {
              description = "OpenPFC tests";
              license = licenses.agpl3;
              platforms = platforms.linux;
            };
          };

        };

        ### DevShells ###

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

        ### Tests ###

        checks = {

          # Check that the license of the project is compatible with AGPL-3.0.
          license-check = pkgs.runCommand "license-check" {
            buildInputs = [ pkgs.nodejs ];
            src = ./.;
          } ''
            # TODO: Uncomment the following lines to check the license.
            # cp -r $src/* .
            # npx license-checker --production
            touch $out
          '';

          # Check that the code is formatted correctly using clang-format.
          format-check = pkgs.runCommand "format-check" {
            buildInputs = [ pkgs.clang-tools_17 ];
            src = ./.;
          } ''
            # TODO: Uncomment the following lines to check the code format.
            # cp -r $src/* .
            # clang-format --dry-run --Werror $(find ./apps ./include ./examples ./tests ./docs -name '*.hpp' -o -name '*.cpp')
            touch $out
          '';

          static-analysis = pkgs.runCommand "clang-tidy-check" {
            buildInputs = [ pkgs.clang-tools ];
            src = ./.;
          } ''
            # clang-tidy $(find src/ include/ tests/ -name '*.cpp')
            touch $out
          '';

          doxygen = pkgs.runCommand "doxygen-docs-check" {
            buildInputs = [ pkgs.doxygen ];
            src = ./.;
          } ''
            # doxygen docs/Doxyfile
            touch $out
          '';

          coverage = pkgs.runCommand "coverage-report" {
            buildInputs = [ pkgs.gcovr ];
            src = ./.;
          } ''
            # gcovr -r . --fail-under-line 80
            touch $out
          '';

          openpfc-tests = pkgs.runCommand "openpfc-tests" {
            buildInputs = [ self.packages.${system}.openpfc-tests ];
            src = ./.;
          } ''
            # ${self.packages.${system}.openpfc-tests}/bin/openpfc-tests
            touch $out
          '';

        };

      });
}
