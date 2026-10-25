// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>

namespace output_lesson {
inline pfc::Domain domain(int argc, char **argv) {
  nlohmann::json config = nlohmann::json::object();
  if (argc > 1) {
    std::ifstream file(argv[1]);
    if (!file) throw std::runtime_error("Cannot open input JSON");
    file >> config;
  }
  const int n = config.value("n", 8);
  if (n < 8 || n > 64) throw std::invalid_argument("Use n in [8,64]");
  std::filesystem::create_directories("results");
  return pfc::domain::create(pfc::GridSize{{n, 4, 2}},
                             pfc::PhysicalOrigin{{-1, 2, 3}},
                             pfc::GridSpacing{{.25, .5, .75}});
}
inline double value(double x, double y, double z, double t = 0) {
  return 1 + x + 2 * y + 3 * z + t;
}
} // namespace output_lesson
