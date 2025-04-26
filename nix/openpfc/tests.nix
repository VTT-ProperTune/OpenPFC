{ openpfcVariations }:

builtins.listToAttrs (map (version: {
  name = "openpfc-tests-${version}";
  value = openpfcVariations.dynamicVariation {
    version = version;
    enableTests = true;
    enableExamples = false;
    enableApps = false;
    enableDocs = false;
  };
}) (builtins.attrNames openpfcVariations.self.inputs.openpfcVersions.versions))
