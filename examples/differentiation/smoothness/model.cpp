// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "../profiles.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <vector>

struct Derivative {
  double x{};
};

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    std::vector<int> sizes;
    for (int n = 6; n <= 50; n += 2) sizes.push_back(n);
    std::vector<std::string> names{"finite-smoothness", "smooth-bump",
                                   "analytic-strip", "band-limited"};
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      nlohmann::json input;
      file >> input;
      sizes = input.value("sizes", sizes);
      names = input.value("profiles", names);
    }
    if (sizes.empty() || sizes.size() > 64)
      throw std::invalid_argument("Use 1 to 64 resolutions");
    for (int n : sizes) {
      if (n < 2 || n > 256 || n % 2)
        throw std::invalid_argument("Use even sizes in [2,256]");
    }
    if (names.empty() || names.size() > 8)
      throw std::invalid_argument("Use 1 to 8 profiles");
    for (const auto &name : names) (void)profiles::evaluate(name, 0.3);
    std::filesystem::create_directories("results");
    std::ofstream output;
    if (comm.rank() == 0) {
      output.open("results/smoothness.csv");
      output << std::setprecision(17)
             << "n,profile,matrix_error,fourier_error,agreement\n";
    }
    double worst_agreement = 0;
    // start snippet regularity
    for (int n : sizes) {
      const double h = 2 * std::numbers::pi / n;
      const auto domain = pfc::domain::create(pfc::GridSize{{n, 4, 4}},
                                              pfc::PhysicalOrigin{{0, 0, 0}},
                                              pfc::GridSpacing{{h, 1, 1}});
      pfc::sim::stacks::SpectralCPUStack stack(domain, comm.rank(), comm.size());
      auto &u = stack.u();
      auto derivative = pfc::field::create<Derivative>(u, stack.fft());
      for (const auto &name : names) {
        std::vector<double> samples(n);
        for (int j = 0; j < n; ++j)
          samples[j] = profiles::evaluate(name, j * h).value;
        u.apply([&](double x, double, double) {
          return profiles::evaluate(name, x).value;
        });
        derivative.prepare();
        std::array<double, 3> local{};
        u.for_each_owned([&](int i, int j, int k) {
          const int row = i + u.box().low[0];
          const double exact = profiles::evaluate(name, row * h).derivative;
          const double matrix = profiles::matrix_derivative(samples, row);
          const double fourier = derivative(i, j, k).x;
          local[0] = std::max(local[0], std::abs(matrix - exact));
          local[1] = std::max(local[1], std::abs(fourier - exact));
          local[2] = std::max(local[2], std::abs(matrix - fourier));
        });
        std::array<double, 3> error{};
        MPI_Allreduce(local.data(), error.data(), 3, MPI_DOUBLE, MPI_MAX, comm);
        worst_agreement = std::max(worst_agreement, error[2]);
        if (comm.rank() == 0) {
          output << n << ',' << name << ',' << error[0] << ',' << error[1] << ','
                 << error[2] << '\n';
        }
      }
    }
    // end snippet regularity
    if (comm.rank() == 0 && !output)
      throw std::runtime_error("Cannot write smoothness.csv");
    return worst_agreement < 1e-10 ? 0 : 1;
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
