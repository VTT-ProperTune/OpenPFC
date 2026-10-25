// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <vector>

double signal(double x) { return 1.0 + 0.5 * std::cos(2 * x) - 0.25 * std::sin(x); }

int main() {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    constexpr int nx = 8, ny = 4, nz = 2;
    constexpr double two_pi = 2 * std::numbers::pi;
    const auto domain = pfc::domain::create(pfc::GridSize{{nx, ny, nz}},
                                            pfc::PhysicalOrigin{{0, 0, 0}},
                                            pfc::GridSpacing{{two_pi / nx, 1, 1}});
    // start snippet transform
    pfc::sim::stacks::SpectralCPUStack stack(domain, comm.rank(), comm.size());
    auto &u = stack.u();
    auto &fft = stack.fft();
    u.apply([](double x, double, double) { return signal(x); });
    const auto original = u.vec();
    std::vector<std::complex<double>> spectrum(fft.size_outbox());
    fft.forward(u.vec(), spectrum);
    // end snippet transform

    std::filesystem::create_directories("results");
    std::ofstream output("results/spectrum_rank_" + std::to_string(comm.rank()) +
                         ".csv");
    output << std::setprecision(17)
           << "i,j,k,real,imag,reference_real,reference_imag\n";
    double coefficient_error = 0.0;
    // start snippet reference
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t index, double, double, double, int i, int j, int k) {
          std::complex<double> reference{};
          for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
              for (int x = 0; x < nx; ++x) {
                const double angle =
                    -two_pi *
                    (double(i * x) / nx + double(j * y) / ny + double(k * z) / nz);
                reference += signal(two_pi * x / nx) * std::polar(1.0, angle);
              }
            }
          }
          coefficient_error =
              std::max(coefficient_error, std::abs(spectrum[index] - reference));
          output << i << ',' << j << ',' << k << ',' << spectrum[index].real() << ','
                 << spectrum[index].imag() << ',' << reference.real() << ','
                 << reference.imag() << '\n';
        });
    // end snippet reference
    if (!output) throw std::runtime_error("Cannot write spectrum CSV");
    // start snippet inverse
    fft.backward(spectrum, u.vec());
    double round_trip_error = 0.0;
    for (std::size_t i = 0; i < original.size(); ++i) {
      round_trip_error =
          std::max(round_trip_error, std::abs(u.vec()[i] - original[i]));
    }
    const double local = std::max(coefficient_error, round_trip_error);
    double error = 0.0;
    MPI_Allreduce(&local, &error, 1, MPI_DOUBLE, MPI_MAX, comm);
    // end snippet inverse
    if (comm.rank() == 0)
      std::cout << "Maximum coefficient/round-trip error=" << error << '\n';
    return std::isfinite(error) && error < 1e-10 ? 0 : 1;
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
