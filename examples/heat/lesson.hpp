// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

// Shared input and observation for the three heat lessons. The numerical
// method remains in each model.cpp; this header never advances the solution.
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>

namespace heat {

struct Derivatives {
  double xx{}, yy{}, zz{};
};

struct Parameters {
  int n = 16;
  int steps = 100;
  double dt = 0.001;
  double diffusivity = 1.0;

  double spacing() const { return 2.0 * std::numbers::pi / n; }

  static Parameters read(int argc, char **argv) {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    Parameters p{input.value("n", 16), input.value("steps", 100),
                 input.value("dt", 0.001), input.value("diffusivity", 1.0)};
    if (p.n < 8 || p.n > 128 || p.n % 2 != 0 || p.steps < 1 || p.steps > 10000 ||
        !std::isfinite(p.dt) || p.dt <= 0.0 || !std::isfinite(p.diffusivity) ||
        p.diffusivity <= 0.0 || !std::isfinite(p.dt * p.diffusivity * p.steps)) {
      throw std::invalid_argument("Use even n in [8,128], steps in [1,10000], "
                                  "finite positive dt and diffusivity");
    }
    return p;
  }
};

inline pfc::Domain domain(const Parameters &p) {
  const double h = p.spacing();
  return pfc::domain::create(pfc::GridSize{{p.n, p.n, p.n}},
                             pfc::PhysicalOrigin{{0.0, 0.0, 0.0}},
                             pfc::GridSpacing{{h, h, h}});
}

// start snippet exact
inline double exact(double x, double y, double z, double t, double diffusivity) {
  return 1.0 + 0.2 * std::exp(-3.0 * diffusivity * t) * std::cos(x) * std::cos(y) *
                   std::cos(z);
}
// end snippet exact

inline std::ofstream open_diagnostics(int rank) {
  std::filesystem::create_directories("results");
  std::ofstream file;
  if (rank == 0) {
    file.open("results/diagnostics.csv");
    if (!file) throw std::runtime_error("Cannot write results/diagnostics.csv");
    file << std::setprecision(17) << "time,mean,rms_fluctuation,rms_error\n";
  }
  return file;
}

inline void observe(const pfc::data::Field<double> &u, const Parameters &p, double t,
                    std::ofstream &file, int rank) {
  std::array<double, 3> local{};
  u.for_each_owned([&](double x, double y, double z, double value) {
    const double error = value - exact(x, y, z, t, p.diffusivity);
    local[0] += value;
    local[1] += (value - 1.0) * (value - 1.0);
    local[2] += error * error;
  });
  std::array<double, 3> global{};
  MPI_Allreduce(local.data(), global.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double cells = static_cast<double>(p.n) * p.n * p.n;
  if (rank == 0) {
    file << t << ',' << global[0] / cells << ',' << std::sqrt(global[1] / cells)
         << ',' << std::sqrt(global[2] / cells) << '\n';
    if (!file) throw std::runtime_error("Cannot write heat diagnostics");
    if (t == p.steps * p.dt) {
      std::cout << std::setprecision(8) << "t=" << t
                << "  mean=" << global[0] / cells
                << "  RMS error=" << std::sqrt(global[2] / cells) << '\n';
    }
  }
}

inline void write_field(const pfc::data::Field<double> &u, const Parameters &p,
                        int rank) {
  std::ofstream file("results/field_rank_" + std::to_string(rank) + ".csv");
  if (!file) throw std::runtime_error("Cannot write rank field CSV");
  file << std::setprecision(17) << "x,y,z,u,exact\n";
  u.for_each_owned([&](double x, double y, double z, double value) {
    file << x << ',' << y << ',' << z << ',' << value << ','
         << exact(x, y, z, p.steps * p.dt, p.diffusivity) << '\n';
  });
  if (!file) throw std::runtime_error("Cannot finish rank field CSV");
}

} // namespace heat
