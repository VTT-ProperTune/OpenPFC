// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <openpfc/frontend/io/snapshot_series.hpp>

int main() {
  const nlohmann::json document{{"value", 42}};
  return document.at("value").get<int>() == 42 ? 0 : 1;
}
