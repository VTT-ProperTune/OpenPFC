// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "diffusion_model.hpp"
#include "diffusion_spectral_helpers.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using namespace std;

void print_stats(const pfc::Time &clock, const pfc::data::Field<double> &field,
                 int midpoint_idx, int rank) {
  if (rank != 0 || midpoint_idx < 0) return;
  cout << "n = " << pfc::time::increment(clock)
       << ", t = " << pfc::time::current(clock) << ", psi[" << midpoint_idx
       << "] = " << field.vec()[static_cast<size_t>(midpoint_idx)] << endl;
}

int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(12);
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  constexpr int L = 64;
  const double dx = 2.0 * pfc::constants::pi / 8.0;
  const double x0 = -0.5 * L * dx;
  pfc::Domain domain = pfc::domain::create(pfc::GridSize{{L, L, L}},
                                           pfc::PhysicalOrigin{{x0, x0, x0}},
                                           pfc::GridSpacing{{dx, dx, dx}});
  pfc::sim::stacks::SpectralCPUStack stack(std::move(domain), rank, nproc);
  auto &fft = stack.fft();
  auto &psi = stack.u();

  if (rank == 0) {
    cout << "Applying custom initial condition at time 0" << endl;
  }
  diffusion_example::fill_gaussian(psi, 1.0);
  const int midpoint_idx = diffusion_example::find_midpoint_idx(psi);

  const double t0 = 0.0;
  const double t1 = 0.5874010519681994;
  const double dt = (t1 - t0) / 42;
  std::vector<double> opL;
  std::vector<complex<double>> psi_F(fft.size_outbox());
  diffusion_example::prepare_implicit_euler_opL(fft, psi.domain(), dt, opL);

  pfc::Time time({t0, t1, dt}, dt);
  print_stats(time, psi, midpoint_idx, rank);
  pfc::sim::run(
      time,
      [&](double) {
        diffusion_example::spectral_diffusion_step(fft, psi.vec(), psi_F, opL);
      },
      {}, {},
      [&](const pfc::Time &clock) { print_stats(clock, psi, midpoint_idx, rank); });

  if (rank == 0 && midpoint_idx >= 0) {
    const double v = psi.vec()[static_cast<size_t>(midpoint_idx)];
    if (abs(v - 0.5) < 0.01) {
      cout << "Test pass!" << endl;
    } else {
      cerr << "Test failed!" << endl;
    }
  }

  MPI_Finalize();
  return 0;
}
