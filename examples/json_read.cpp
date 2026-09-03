// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file json_read.cpp
 * @brief Minimal nlohmann::json stream parsing example.
 *
 * Production OpenPFC applications load settings with
 * `pfc::ui::load_settings_file(<config.json|config.toml>)` and build a session
 * from the parsed document. This standalone example only
 * demonstrates how raw JSON can be read from a file or stdin.
 */

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char *argv[]) {
  std::cout << "JSON stream read example\n";
  json settings;
  if (argc > 1) {
    std::cout << "Reading JSON from file " << argv[1] << "\n";
    std::ifstream input_file(argv[1]);
    input_file >> settings;
  } else {
    std::cout << "Reading JSON from standard input:\n";
    std::cin >> settings;
  }
  std::cout << "Simulation settings:\n\n";
  std::cout << settings.dump(4) << "\n\n";
  return 0;
}
