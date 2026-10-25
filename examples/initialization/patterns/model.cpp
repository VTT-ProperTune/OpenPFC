// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <iostream>
#include <numbers>

// start snippet gaussian
struct Gaussian {
  double width;
  double operator()(double x, double y, double z) const {
    return std::exp(-(x * x + y * y + z * z) / (2 * width * width));
  }
};
// end snippet gaussian

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    const auto config = initialization::input(argc, argv);
    const auto domain = initialization::domain(config);
    const double width = config.value("width", .8), core = config.value("core", .5);
    const double circulation = config.value("circulation", 5.0);
    if (!std::isfinite(width) || width <= 0 || !std::isfinite(core) || core <= 0 ||
        !std::isfinite(circulation))
      throw std::invalid_argument(
          "Use positive finite width/core and finite circulation");
    const auto partition = pfc::decomposition::create(domain, comm.size());
    auto field = pfc::data::field_from_subdomain<double>(partition, comm.rank(), 0);
    auto output = initialization::output(comm.rank());
    field.apply(Gaussian{width});
    initialization::write(output, "gaussian", field);

    // start snippet vortex
    field.apply([&](double x, double y, double) {
      const double r = std::hypot(x, y);
      return r == 0 ? 0
                    : -circulation * std::expm1(-r * r / (core * core)) /
                          (2 * std::numbers::pi * r);
    });
    // end snippet vortex
    initialization::write(output, "vortex_speed", field);

    // start snippet checkerboard
    field.apply([](double x, double y, double z) {
      const int cell =
          static_cast<int>(std::floor(x) + std::floor(y) + std::floor(z));
      return cell % 2 ? -1.0 : 1.0;
    });
    // end snippet checkerboard
    initialization::write(output, "checkerboard", field);
    if (comm.rank() == 0) std::cout << "Three coordinate-defined patterns written\n";
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
