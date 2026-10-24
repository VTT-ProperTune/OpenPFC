// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <numbers>
#include <stdexcept>
#include <string>
#include <tuple>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

// Edit this callable to change the physics: du/dt = D * Laplacian(u).
struct Derivatives {
  double xx{}, yy{}, zz{};
};
struct Diffusion {
  double diffusivity;
  double rhs(double /*time*/, const Derivatives &g) const {
    return diffusivity * (g.xx + g.yy + g.zz);
  }
};

int integer(const nlohmann::json &input, const std::string &name, int lower,
            int upper) {
  const auto &value = input.at(name);
  if (!value.is_number_integer() || value.get<double>() < lower ||
      value.get<double>() > upper)
    throw std::invalid_argument(name + " must be an integer in [" +
                                std::to_string(lower) + ", " +
                                std::to_string(upper) + "]");
  return value.get<int>();
}

void simulate(const char *input_path, int rank, int ranks) {
  std::ifstream stream(input_path);
  if (!stream) throw std::runtime_error("cannot open input JSON");
  nlohmann::json input;
  stream >> input;
  const int cells = integer(input, "cells", 4, 256);
  const int steps = integer(input, "steps", 1, 1000000);
  const double dt = input.at("dt").get<double>();
  const double diffusivity = input.at("diffusivity").get<double>();
  // Unit spacing, seven-point Laplacian and forward Euler in three dimensions.
  if (!std::isfinite(dt) || !std::isfinite(diffusivity) || dt <= 0 ||
      diffusivity <= 0 || dt > 1.0 / (6.0 * diffusivity) ||
      !std::isfinite(dt * steps))
    throw std::invalid_argument(
        "require finite dt > 0, diffusivity > 0 and dt <= 1/(6*diffusivity)");

  const auto domain = pfc::domain::create(pfc::Int3{cells, cells, cells});
  const auto decomposition = pfc::decomposition::create(domain, ranks);
  auto u = pfc::data::field_from_subdomain<double>(decomposition, rank, 1);
  pfc::comm::HaloExchange<pfc::HostSpace, double> halo(u, decomposition, rank,
                                                       MPI_COMM_WORLD);
  u.apply([&](double x, double, double) {
    return 1.0 + 0.1 * std::cos(2.0 * std::numbers::pi * x / cells);
  });
  pfc::gradient::FDGradient<Derivatives> gradient(u, 2);
  const Diffusion physics{diffusivity};
  // Derivatives use owned coordinates; rates use the field's padded storage.
  auto rhs = [&](double time, auto &rates) {
    u.for_each_owned([&](int i, int j, int k) {
      std::get<0>(rates)[u.idx(i, j, k)] = physics.rhs(time, gradient(i, j, k));
    });
  };
  pfc::sim::steppers::OwnedEulerStepper<decltype(rhs), 1> stepper(dt, {u.size()},
                                                                  rhs);

  std::filesystem::create_directories("results");
  std::ofstream diagnostics;
  if (rank == 0) {
    diagnostics.exceptions(std::ios::failbit | std::ios::badbit);
    diagnostics.open("results/diagnostics.csv");
    diagnostics << "step,time,mean,l2\n" << std::setprecision(17);
  }
  const double count = static_cast<double>(cells) * cells * cells;
  double initial_l2 = 0, final_l2 = 0, mean = 0;
  for (int step = 0; step <= steps; ++step) {
    double local[2]{};
    u.for_each_owned([&](double, double, double, double value) {
      local[0] += value;
      local[1] += value * value;
    });
    double global[2]{};
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mean = global[0] / count;
    final_l2 = std::sqrt(global[1] / count);
    if (step == 0) initial_l2 = final_l2;
    if (rank == 0)
      diagnostics << step << ',' << step * dt << ',' << mean << ',' << final_l2
                  << '\n';
    if (step == steps) break;
    halo.exchange();
    (void)stepper.step(step * dt, u);
  }

  std::ofstream field;
  field.exceptions(std::ios::failbit | std::ios::badbit);
  field.open("results/field_rank_" + std::to_string(rank) + ".csv");
  field << "x,y,z,u\n" << std::setprecision(17);
  u.for_each_owned([&](double x, double y, double z, double value) {
    field << x << ',' << y << ',' << z << ',' << value << '\n';
  });
  field.close();
  if (rank == 0) {
    diagnostics.close();
    std::ofstream summary;
    summary.exceptions(std::ios::failbit | std::ios::badbit);
    summary.open("results/summary.json");
    summary << nlohmann::json{{"ranks", ranks},           {"steps", steps},
                              {"time", steps * dt},       {"mean", mean},
                              {"initial_l2", initial_l2}, {"final_l2", final_l2}}
                   .dump(2)
            << '\n';
    summary.close();
    std::cout << "Diffusion complete: " << steps << " steps on " << ranks
              << " rank(s); mean=" << mean << ", L2 " << initial_l2 << " -> "
              << final_l2 << ". Results: results/\n";
  }
} // MPI-owning objects are destroyed before MPI_Finalize().

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (argc != 2) throw std::invalid_argument("usage: model input.json");
    simulate(argv[1], rank, ranks);
    MPI_Finalize();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "diffusion (rank " << rank << "): " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
