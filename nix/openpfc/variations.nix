{ pkgs, system, self, openpfcVersions }:

let
  # mkVariation: A low-level function to create a specific OpenPFC variation.
  # This function directly calls the default.nix derivation with the provided
  # parameters, such as version, build type, and feature flags.
  # Parameters:
  # - version: The version of OpenPFC to build (e.g., "0.1.0", "dev").
  # - buildType: The build type (e.g., "Debug" or "Release"). Defaults to "Debug" for "dev" versions.
  # - enableTests: Boolean to enable or disable tests.
  # - enableExamples: Boolean to enable or disable examples.
  # - enableApps: Boolean to enable or disable apps.
  # - enableDocs: Boolean to enable or disable documentation.
  # Returns:
  # - A derivation for the specified OpenPFC variation.
  mkVariation = { version, heffte
    , buildType ? if version == "dev" then "Debug" else "Release", enableTests
    , enableExamples, enableApps, enableDocs }:
    pkgs.callPackage ./default.nix {
      inherit version buildType heffte;
      src = if version == "dev" then
        ./.
      else if version == "master" then
        builtins.fetchGit {
          url = "https://github.com/VTT-ProperTune/OpenPFC.git";
          ref = "master";
        }
      else
        pkgs.fetchFromGitHub {
          owner = "VTT-ProperTune";
          repo = "OpenPFC";
          inherit (openpfcVersions.versions.${version}) rev sha256;
        };
      inherit enableTests enableExamples enableApps enableDocs;
    };

  # dynamicVariation: A high-level wrapper around mkVariation to simplify the creation of variations.
  # This function applies default values for parameters like build type and feature flags,
  # making it easier to dynamically generate variations with consistent defaults.
  # Parameters:
  # - version: The version of OpenPFC to build (e.g., "0.1.0", "dev").
  # - buildType: The build type (e.g., "Debug" or "Release"). Defaults to "Debug" for "dev" versions.
  # - enableTests: Boolean to enable or disable tests. Defaults to false.
  # - enableExamples: Boolean to enable or disable examples. Defaults to false.
  # - enableApps: Boolean to enable or disable apps. Defaults to false.
  # - enableDocs: Boolean to enable or disable documentation. Defaults to false.
  # Returns:
  # - A derivation for the specified OpenPFC variation with default settings applied.
  dynamicVariation = { version
    , buildType ? if version == "dev" then "Debug" else "Release", enableTests
    , enableExamples, enableApps, enableDocs, heffte }:
    mkVariation {
      inherit version buildType enableTests enableExamples enableApps enableDocs
        heffte;
    };

  # Generate attributes for all versions
  versionedVariations = builtins.listToAttrs (map (version: {
    name = "openpfc-${version}";
    value = dynamicVariation {
      version = version;
      enableTests = true;
      enableExamples = true;
      enableApps = true;
      enableDocs = true;
    };
  }) (builtins.attrNames openpfcVersions.versions));
in {
  inherit dynamicVariation;

  # Default current version
  openpfc = dynamicVariation {
    version = openpfcVersions.current;
    enableTests = true;
    enableExamples = true;
    enableApps = true;
    enableDocs = true;
  };

  # Development version
  "openpfc-dev" = dynamicVariation {
    version = "dev";
    enableTests = true;
    enableExamples = true;
    enableApps = true;
    enableDocs = true;
  };

  # Master version
  "openpfc-master" = dynamicVariation {
    version = "master";
    enableTests = true;
    enableExamples = true;
    enableApps = true;
    enableDocs = true;
  };

}
