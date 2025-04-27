# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

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
