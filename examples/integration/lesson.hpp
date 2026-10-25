// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace lesson {
inline nlohmann::json input(int argc, char **argv) {
  if (argc < 2) return nlohmann::json::object();
  std::ifstream file(argv[1]);
  if (!file) throw std::runtime_error("Cannot open input JSON");
  nlohmann::json config;
  file >> config;
  return config;
}
inline void positive(double value, const char *name) {
  if (!(value > 0) || !std::isfinite(value))
    throw std::invalid_argument(std::string(name) + " must be finite and positive");
}
inline std::ofstream csv(const std::string &name) {
  std::filesystem::create_directories("results");
  std::ofstream file("results/" + name);
  if (!file) throw std::runtime_error("Cannot open output CSV");
  file.exceptions(std::ios::badbit | std::ios::failbit);
  file << std::setprecision(17);
  return file;
}
} // namespace lesson
