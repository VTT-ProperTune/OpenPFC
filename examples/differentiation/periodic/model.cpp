// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// An OpenPFC experiment inspired by Trefethen, Spectral Methods in MATLAB,
// programs 1 and 5. This is an independent C++ implementation, not a port.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

struct Derivative {
  double x{};
};

double value(double x) { return std::exp(std::sin(x)); }

double exact_derivative(double x) { return std::cos(x) * std::exp(std::sin(x)); }

// start snippet differentiate
template <class Evaluator>
double derivative_error(const pfc::data::Field<double> &u, Evaluator &derivative) {
  derivative.prepare();
  double local_error = 0.0;
  u.for_each_owned([&](int i, int j, int k) {
    const double x = u.coords(i, j, k)[0];
    const double exact = exact_derivative(x);
    local_error = std::max(local_error, std::abs(derivative(i, j, k).x - exact));
  });
  double error = 0.0;
  MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return error;
}
// end snippet differentiate

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
    const auto sizes = input.value("sizes", std::vector<int>{8, 16, 32, 64, 128});
    if (sizes.empty() || sizes.size() > 16) {
      throw std::invalid_argument("Use between 1 and 16 grid sizes");
    }
    for (int n : sizes) {
      if (n < 8 || n > 8192 || n % 2 != 0) {
        throw std::invalid_argument("Use even grid sizes in [8,8192]");
      }
    }
    std::filesystem::create_directories("results");
    std::ofstream file;
    if (comm.rank() == 0) {
      file.open("results/convergence.csv");
      if (!file) throw std::runtime_error("Cannot write convergence.csv");
      file << std::setprecision(17) << "n,fd2,fd4,fourier\n";
      std::cout << "N         FD2             FD4             Fourier\n";
    }

    // start snippet compare
    for (int n : sizes) {
      const double h = 2.0 * std::numbers::pi / n;
      const auto domain = pfc::domain::create(pfc::GridSize{{n, 8, 8}},
                                              pfc::PhysicalOrigin{{0.0, 0.0, 0.0}},
                                              pfc::GridSpacing{{h, 1.0, 1.0}});
      const auto decomp = pfc::decomposition::create(domain, comm.size());
      auto u = pfc::data::field_from_subdomain<double>(decomp, comm.rank(), 2);
      pfc::comm::HaloExchange<pfc::HostSpace, double> halo(u, decomp, comm.rank(),
                                                           comm);
      u.apply([](double x, double, double) { return value(x); });
      halo.exchange();
      auto fd2 = pfc::field::create<Derivative>(u, 2);
      auto fd4 = pfc::field::create<Derivative>(u, 4);
      pfc::sim::stacks::SpectralCPUStack fourier(domain, comm.rank(), comm.size());
      fourier.u().apply([](double x, double, double) { return value(x); });
      auto spectral = pfc::field::create<Derivative>(fourier.u(), fourier.fft());
      const double e2 = derivative_error(u, fd2);
      const double e4 = derivative_error(u, fd4);
      const double ef = derivative_error(fourier.u(), spectral);
      if (comm.rank() == 0) {
        file << n << ',' << e2 << ',' << e4 << ',' << ef << '\n';
        std::cout << n << "  " << std::scientific << e2 << "  " << e4 << "  " << ef
                  << '\n';
      }
    }
    // end snippet compare
    if (comm.rank() == 0 && !file)
      throw std::runtime_error("Cannot finish convergence.csv");
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
