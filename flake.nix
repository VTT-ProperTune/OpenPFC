# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

{
  description = "OpenPFC and HeFFTe builder";

  # Inputs define external dependencies for this flake.
  inputs.nixpkgs.url =
    "github:NixOS/nixpkgs/nixos-24.05"; # Nixpkgs repository for system packages.
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
          openpfc = pkgs.callPackage ./nix/openpfc/default.nix {
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
          openpfc-dev = pkgs.callPackage ./nix/openpfc/default.nix {
            version = "dev";
            buildType = "Debug";
            src = ./.;
            enableTests = true;
            enableDocs = true;
            enableExamples = true;
            enableApps = true;
            heffte = heffte;
          };

          # openpfc-tests: compile only the tests.
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

          # openpfc-docs: compile only the docs.
          openpfc-docs = pkgs.callPackage openpfcPath {
            version = "dev";
            buildType = "Debug";
            src = ./.;
            enableTests = false;
            enableDocs = true;
            enableExamples = false;
            enableApps = false;
            heffte = heffte;
          };

          # default: build dev version of OpenPFC.
          default = pkgs.callPackage openpfcPath {
            version = "dev";
            buildType = "Debug";
            src = ./.;
            enableTests = true;
            enableDocs = true;
            enableExamples = true;
            enableApps = true;
            heffte = heffte;
          };
        };

        ### Applications ###

        apps = {

          test = {
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
            pkgs.ninja
            pkgs.cmake
            pkgs.git
            pkgs.openmpi
            pkgs.nlohmann_json
            pkgs.doxygen
            pkgs.catch2_3

            # 🆕 Add these for better code hygiene
            pkgs.clang-tools # Provides clang-tidy, clang-format, etc. (latest version)
            pkgs.reuse # License checking
            pkgs.coreutils # Needed for realpath and other shell utilities
          ];

          inputsFrom = [
            self.packages.${system}.heffte
            self.packages.${system}.openpfc-dev
          ];

          shellHook = ''
            export Heffte_DIR=${self.packages.${system}.heffte}/lib/cmake/Heffte

            echo ""
            echo "🎉 Welcome to the OpenPFC development shell!"
            echo "👉 To configure the project:  cmake -S . -B build -DOpenPFC_DEVELOPMENT=yes"
            echo "👉 To build the project:      cmake --build build"
            echo "👉 To format code:            nix run .#clang-format"
            echo "👉 To run static analysis:    nix run .#clang-tidy"
            echo "👉 To check licenses:         nix run .#reuse"
            echo ""
          '';
        };

        ### Tests ###

        checks = {

          # Check that the license of the project is compatible with AGPL-3.0.
          license-check = pkgs.runCommand "license-check" {
            nativeBuildInputs = [ pkgs.reuse ];
            src = ./.;
          } ''
            reuse --root $src lint
            touch $out
          '';

          # Check that the code is formatted correctly using clang-format.
          format-check = pkgs.runCommand "format-check" {
            buildInputs = [ pkgs.clang-tools ];
            src = ./.;
          } ''
            cp -r $src/. .

            # Find only real source files (*.cpp, *.hpp) in apps, include, examples, and tests
            files=$(find ./apps ./include ./examples ./tests ./docs \( -name '*.hpp' -o -name '*.cpp' \))

            # Run clang-format check
            clang-format --dry-run --Werror $files

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

        };

      });
}
