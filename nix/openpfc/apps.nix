# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

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
