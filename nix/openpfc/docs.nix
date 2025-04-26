{ openpfcVariations, openpfcVersions }:

builtins.listToAttrs (map (version: {
  name = "openpfc-docs-${version}";
  value = openpfcVariations.dynamicVariation {
    version = version;
    enableTests = false;
    enableExamples = false;
    enableApps = false;
    enableDocs = true;
  };
}) (builtins.attrNames openpfcVersions.versions))
