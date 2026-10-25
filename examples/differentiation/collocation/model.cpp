// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// The dense circulant formula is example-local mathematics. OpenPFC owns
// the sampled field, distributed FFT and spectral derivative evaluator.
#include "../profiles.hpp"
#include <algorithm>
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
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    const int n = input.value("n", 24);
    if (n < 4 || n > 128 || n % 2)
      throw std::invalid_argument("Use even n in [4,128]");
    const double h = 2 * std::numbers::pi / n;
    const auto domain =
        pfc::domain::create(pfc::GridSize{{n, 4, 4}}, pfc::PhysicalOrigin{{0, 0, 0}},
                            pfc::GridSpacing{{h, 1, 1}});
    pfc::sim::stacks::SpectralCPUStack stack(domain, comm.rank(), comm.size());
    auto &u = stack.u();
    auto derivative = pfc::field::create<Derivative>(u, stack.fft());
    std::filesystem::create_directories("results");
    std::ofstream output("results/derivative_rank_" + std::to_string(comm.rank()) +
                         ".csv");
    output << std::setprecision(17)
           << "profile,x,value,matrix,fourier,exact,differentiable\n";
    double agreement = 0;
    // start snippet compare
    for (const auto name : {"hat", "analytic"}) {
      std::vector<double> samples(n);
      for (int j = 0; j < n; ++j) samples[j] = profiles::evaluate(name, j * h).value;
      u.apply([&](double x, double, double) {
        return profiles::evaluate(name, x).value;
      });
      derivative.prepare();
      u.for_each_owned([&](int i, int j, int k) {
        const auto global = u.box().low;
        if (j + global[1] != 0 || k + global[2] != 0) return;
        const double x = (i + global[0]) * h;
        const auto exact = profiles::evaluate(name, x);
        const double dense = profiles::matrix_derivative(samples, i + global[0]);
        const double fourier = derivative(i, j, k).x;
        agreement = std::max(agreement, std::abs(dense - fourier));
        output << name << ',' << x << ',' << u(i, j, k) << ',' << dense << ','
               << fourier << ',' << exact.derivative << ',' << exact.differentiable
               << '\n';
      });
    }
    // end snippet compare
    if (!output) throw std::runtime_error("Cannot write derivative CSV");
    double global_agreement = 0;
    MPI_Allreduce(&agreement, &global_agreement, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (comm.rank() == 0)
      std::cout << "Matrix/FFT agreement=" << global_agreement << '\n';
    return global_agreement < 1e-10 ? 0 : 1;
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
