{ openpfcVariations }:

builtins.listToAttrs (map (version: {
  name = "openpfc-examples-${version}";
  value = openpfcVariations.dynamicVariation {
    version = version;
    enableTests = false;
    enableExamples = true;
    enableApps = false;
    enableDocs = false;
  };
}) (builtins.attrNames openpfcVariations.self.inputs.openpfcVersions.versions))
