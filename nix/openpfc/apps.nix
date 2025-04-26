{ openpfcVariations }:

builtins.listToAttrs (map (version: {
  name = "openpfc-apps-${version}";
  value = openpfcVariations.dynamicVariation {
    version = version;
    enableTests = false;
    enableExamples = false;
    enableApps = true;
    enableDocs = false;
  };
}) (builtins.attrNames openpfcVariations.self.inputs.openpfcVersions.versions))
