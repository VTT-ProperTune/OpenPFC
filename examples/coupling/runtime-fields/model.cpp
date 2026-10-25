// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/steppers/runtime_field_rk.hpp>

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input");
      file >> input;
    }
    const int steps = input.value("steps", 20);
    const double dt = input.value("dt", .05);
    if (steps < 1 || steps > 1000 || !std::isfinite(dt) || !(dt > 0) || dt > .5)
      throw std::invalid_argument("Use 1..1000 steps and 0 < dt <= .5");
    const auto domain = pfc::domain::create(
        pfc::GridSize{{16, 4, 4}}, pfc::PhysicalOrigin{{0, 0, 0}},
        pfc::GridSpacing{{2 * std::numbers::pi / 16, 1, 1}});
    const auto decomposition = pfc::decomposition::create(domain, comm.size());
    auto u = pfc::data::field_from_subdomain<double>(decomposition, comm.rank(), 0);
    auto v = pfc::data::field_from_subdomain<double>(decomposition, comm.rank(), 0);
    u.apply([](double x, double, double) { return std::cos(x); });
    v.apply([](double, double, double) { return 0.; });
    // start snippet fields
    std::array fields{&u, &v};
    auto rhs = [&](double, pfc::sim::steppers::RuntimeFieldRKStepper::Rates rates) {
      std::size_t offset = 0;
      u.for_each_owned([&](int i, int j, int k) {
        rates[0][offset] = v(i, j, k);
        rates[1][offset] = -u(i, j, k);
        ++offset;
      });
    };
    auto stepper = pfc::sim::steppers::create_runtime(
        fields, dt, pfc::sim::steppers::make_rk4_classical<double>(), rhs, comm);
    for (int step = 0; step < steps; ++step) {
      const auto candidate = stepper.attempt(step * dt);
      stepper.commit(candidate);
    }
    // end snippet fields
    std::filesystem::create_directories("results");
    std::ofstream output("results/oscillator_rank_" + std::to_string(comm.rank()) +
                         ".csv");
    if (!output) throw std::runtime_error("Cannot open oscillator CSV");
    output << std::setprecision(17) << "i,j,k,x,time,u,v\n";
    u.for_each_owned([&](int i, int j, int k) {
      const auto low = u.box().low;
      output << i + low[0] << ',' << j + low[1] << ',' << k + low[2] << ','
             << (i + low[0]) * domain.spacing[0] << ',' << steps * dt << ','
             << u(i, j, k) << ',' << v(i, j, k) << '\n';
    });
    if (!output) throw std::runtime_error("Cannot write oscillator CSV");
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
