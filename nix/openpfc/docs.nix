# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

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
