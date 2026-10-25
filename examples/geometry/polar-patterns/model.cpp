// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>

// start snippet mapping
struct Polar {
  double radius, angle;
};

Polar polar(double x, double y) {
  const double r = std::hypot(x, y);
  return {r, r == 0 ? std::numeric_limits<double>::quiet_NaN() : std::atan2(y, x)};
}
// end snippet mapping

// start snippet pattern
struct AngularGaussian {
  double width;
  int harmonic;

  double operator()(double x, double y, double) const {
    const auto [r, theta] = polar(x, y);
    if (r == 0) return harmonic == 0 ? 1 : 0;
    return std::pow(r / width, harmonic) * std::exp(-r * r / (2 * width * width)) *
           std::cos(harmonic * theta);
  }
};
// end snippet pattern

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    auto input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    const int n = input.value("n", 64), harmonic = input.value("harmonic", 3);
    const double width = input.value("width", .3);
    if (n < 8 || n > 128 || harmonic < 0 || harmonic > 8 || !std::isfinite(width) ||
        width < .05 || width > 1)
      throw std::invalid_argument(
          "Use n in [8,128], harmonic in [0,8], width in [0.05,1]");
    // start snippet initialize
    const auto domain = pfc::domain::create(
        pfc::GridSize{{n, n, 1}}, pfc::PhysicalOrigin{{-1, -1, 0}},
        pfc::GridSpacing{{2.0 / n, 2.0 / n, 1}}, pfc::Bool3{false, false, false});
    const auto partition = pfc::decomposition::create(domain, comm.size());
    auto field = pfc::data::field_from_subdomain<double>(partition, comm.rank(), 0);
    field.apply(AngularGaussian{width, harmonic});
    // end snippet initialize
    std::filesystem::create_directories("results");
    std::ofstream output("results/polar_rank_" + std::to_string(comm.rank()) +
                         ".csv");
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << std::setprecision(17) << "i,j,x,y,r,theta,u\n";
    const auto low = field.box().low;
    field.for_each_owned([&](int i, int j, int k) {
      const pfc::Int3 global{i + low[0], j + low[1], k + low[2]};
      const auto x = pfc::domain::to_coords(domain, global);
      const auto [r, theta] = polar(x[0], x[1]);
      output << global[0] << ',' << global[1] << ',' << x[0] << ',' << x[1] << ','
             << r << ',' << theta << ',' << field(i, j, k) << '\n';
    });
    if (comm.rank() == 0)
      std::cout << "Polar pattern on a Cartesian field, harmonic " << harmonic
                << '\n';
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
