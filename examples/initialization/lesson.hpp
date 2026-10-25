// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <stdexcept>

namespace initialization {
inline nlohmann::json input(int argc, char **argv) {
  if (argc < 2) return nlohmann::json::object();
  std::ifstream file(argv[1]);
  if (!file) throw std::runtime_error("Cannot open input JSON");
  nlohmann::json config;
  file >> config;
  return config;
}

inline pfc::Domain domain(const nlohmann::json &config) {
  const int n = config.value("n", 24), nz = config.value("nz", 1);
  if (n < 8 || n > 128 || nz < 1 || nz > 32)
    throw std::invalid_argument("Use n in [8,128] and nz in [1,32]");
  return pfc::domain::create(pfc::GridSize{{n, n, nz}},
                             pfc::PhysicalOrigin{{-4, -4, 0}},
                             pfc::GridSpacing{{8.0 / n, 8.0 / n, 1.0 / nz}});
}

inline std::ofstream output(int rank) {
  std::filesystem::create_directories("results");
  std::ofstream file("results/fields_rank_" + std::to_string(rank) + ".csv");
  if (!file) throw std::runtime_error("Cannot open field CSV");
  file.exceptions(std::ios::badbit | std::ios::failbit);
  file << std::setprecision(17) << "pattern,x,y,z,u\n";
  return file;
}

inline void write(std::ofstream &file, const std::string &name,
                  const pfc::data::Field<double> &field) {
  field.for_each_owned([&](double x, double y, double z, double value) {
    if (!std::isfinite(value)) throw std::runtime_error("Non-finite initial field");
    file << name << ',' << x << ',' << y << ',' << z << ',' << value << '\n';
  });
}
} // namespace initialization
