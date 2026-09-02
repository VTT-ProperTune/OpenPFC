// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "diffusion_spectral_helpers.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

/**
 * \example 05_simulator.cpp
 *
 * Same spectral implicit-Euler diffusion as example 04, but the initial
 * condition is a host-buffer callable (not baked into the physics) and the
 * time loop is `pfc::sim::run` over `Time`. No `Model`, `Simulator`, or
 * `FieldModifier`.
 */

using namespace pfc;

void apply_gaussian_ic(data::Field<double> &field, double D) {
  field.apply([&](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / (4.0 * D));
  });
}

int main(int argc, char *argv[]) {
  std::cout << std::fixed;
  std::cout.precision(12);
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  constexpr int L = 64;
  const double h = 2.0 * constants::pi / 8.0;
  const double o = -0.5 * L * h;
  Domain domain = domain::create(GridSize{{L, L, L}}, PhysicalOrigin{{o, o, o}},
                                 GridSpacing{{h, h, h}});
  sim::stacks::SpectralCPUStack stack(std::move(domain), rank, nproc);
  auto &fft = stack.fft();
  auto &psi = stack.u();

  if (rank == 0) std::cout << "Create initial condition" << std::endl;
  apply_gaussian_ic(psi, 1.0);

  const double t0 = 0.0;
  const double t1 = 0.5874010519681994;
  const double dt = (t1 - t0) / 42;
  if (rank == 0) std::cout << "Prepare operators" << std::endl;
  std::vector<double> opL(fft.size_outbox());
  std::vector<std::complex<double>> psi_F(fft.size_outbox());
  pfc::fft::kspace::for_each_kpoint(
      get_outbox(fft), psi.domain(),
      [&](std::size_t idx, double ki, double kj, double kk, int, int, int) {
        const double kLap = -(ki * ki + kj * kj + kk * kk);
        opL[idx] = 1.0 / (1.0 - dt * kLap);
      });

  Time time({t0, t1, dt}, dt);
  double psi_min = 0.0;
  double psi_max = 1.0;
  auto print_statline = [&](const Time &clock) {
    if (rank != 0) return;
    std::cout << "n = " << time::increment(clock) << ", t = " << time::current(clock)
              << ", min = " << psi_min << ", max = " << psi_max << std::endl;
  };

  pfc::sim::run(
      time,
      [&](double) {
        diffusion_example::spectral_diffusion_step(fft, psi.vec(), psi_F, opL);
        diffusion_example::reduce_psi_min_max_mpi(psi.vec(), psi_min, psi_max);
      },
      {}, {}, print_statline);

  if (rank == 0) {
    if (std::abs(psi_max - 0.5) < 0.01) {
      std::cout << "Test pass!" << std::endl;
    } else {
      std::cerr << "Test failed!" << std::endl;
    }
  }

  MPI_Finalize();
  return 0;
}
