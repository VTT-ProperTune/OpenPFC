// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <vector>

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
    const auto backend = input.value("implementation", std::string("heffte"));
    if (backend != "heffte" && backend != "local") {
      throw std::invalid_argument("implementation must be heffte or local");
    }
    const auto implementation = backend == "local"
                                    ? pfc::fft::Implementation::Local
                                    : pfc::fft::Implementation::Heffte;
    // start snippet layouts
    const auto domain = pfc::domain::create(
        pfc::GridSize{{16, 8, 4}}, pfc::PhysicalOrigin{{0, 0, 0}},
        pfc::GridSpacing{{2 * std::numbers::pi / 16, 1, 1}});
    pfc::sim::stacks::SpectralCPUStack stack(domain, comm.rank(), comm.size(), comm,
                                             implementation);
    auto &u = stack.u();
    auto &fft = stack.fft();
    u.apply([](double x, double, double) { return 1 + std::cos(3 * x); });
    std::vector<std::complex<double>> spectrum(fft.size_outbox());
    const auto original = u.vec();
    fft.forward(u.vec(), spectrum);
    fft.backward(spectrum, u.vec());
    // end snippet layouts
    double local_error = 0;
    for (std::size_t i = 0; i < original.size(); ++i) {
      local_error = std::max(local_error, std::abs(original[i] - u.vec()[i]));
    }
    double error = 0;
    MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, comm);
    const auto real = static_cast<unsigned long long>(u.size());
    const auto complex = static_cast<unsigned long long>(spectrum.size());
    unsigned long long real_total = 0, complex_total = 0;
    MPI_Allreduce(&real, &real_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(&complex, &complex_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  comm);
    if (comm.rank() == 0) {
      std::cout << "implementation=" << backend << " real=" << real_total
                << " complex=" << complex_total << " error=" << error << '\n';
    }
    return real_total == 512 && complex_total == 288 && error < 1e-12 ? 0 : 1;
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
