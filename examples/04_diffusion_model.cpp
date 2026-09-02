// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "diffusion_spectral_helpers.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using namespace std;
using namespace pfc;

/** \example 04_diffusion_model.cpp
 *
 * Spectral implicit-Euler diffusion on a `SpectralCPUStack` (Domain +
 * Decomposition + `IHostFFT` + host field). Physics is a linear operator in
 * k-space; the time loop is `pfc::sim::run`. No `Model` / `World`.
 *
 * Expected output is:
 *
 *      ( initialization messages ... )
 *      n = 0, t = 0.000000000000, psi[133152] = 1.000000000000
 *      n = 1, t = 0.013985739333, psi[133152] = 0.979721090279
 *      n = 2, t = 0.027971478665, psi[133152] = 0.960110027682
 *      n = 3, t = 0.041957217998, psi[133152] = 0.941136780128
 *      n = 4, t = 0.055942957330, psi[133152] = 0.922773010503
 *      ( time stepping continues ...)
 *      n = 40, t = 0.559429573303, psi[133152] = 0.516585236400
 *      n = 41, t = 0.573415312636, psi[133152] = 0.509734461852
 *      n = 42, t = 0.587401051968, psi[133152] = 0.503032957135
 *
 * The live print is global min/max (the midpoint index in the comment is from
 * an older single-rank dump). Rank 0 checks `psi_max` against 0.5.
 */
int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(12);
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  constexpr int L = 64;
  const double pi = 3;
  const double dx = 2.0 * pi / 8.0;
  const double x0 = -0.5 * L * dx;
  Domain domain = domain::create(GridSize{{L, L, L}}, PhysicalOrigin{{x0, x0, x0}},
                                 GridSpacing{{dx, dx, dx}});

  sim::stacks::SpectralCPUStack stack(std::move(domain), rank, nproc);
  auto &fft = stack.fft();
  auto &psi = stack.u();
  if (rank == 0) {
    cout << "Allocate space" << endl;
    cout << "Domain: " << psi.domain() << endl;
    cout << "Create initial condition" << endl;
  }
  constexpr double D = 1.0;
  psi.apply([&](double x, double y, double z) {
    return exp(-(x * x + y * y + z * z) / (4.0 * D));
  });

  const double t0 = 0.0;
  const double t_stop = 0.5874010519681994;
  const double dt = (t_stop - t0) / 42;
  if (rank == 0) cout << "Prepare operators" << endl;
  std::vector<double> opL(fft.size_outbox());
  std::vector<complex<double>> psi_F(fft.size_outbox());
  pfc::fft::kspace::for_each_kpoint(
      get_outbox(fft), psi.domain(),
      [&](std::size_t idx, double ki, double kj, double kk, int, int, int) {
        const double kLap = -(ki * ki + kj * kj + kk * kk);
        opL[idx] = 1.0 / (1.0 - dt * kLap);
      });

  Time time({t0, t_stop, dt}, dt);

  double psi_min = 0.0;
  double psi_max = 1.0;
  auto print_line = [&](const Time &clock) {
    if (rank != 0) return;
    cout << "n = " << time::increment(clock) << ", t = " << time::current(clock)
         << ", min = " << psi_min << ", max = " << psi_max << endl;
  };

  if (rank == 0) cout << "n = 0, t = 0, min = 0.0, max = 1.0" << endl;
  pfc::sim::run(
      time,
      [&](double) {
        diffusion_example::spectral_diffusion_step(fft, psi.vec(), psi_F, opL);
        diffusion_example::reduce_psi_min_max_mpi(psi.vec(), psi_min, psi_max);
      },
      {}, {}, print_line);

  if (rank == 0) {
    if (abs(psi_max - 0.5) < 0.01) {
      cout << "Test pass!" << endl;
    } else {
      cerr << "Test failed!" << endl;
    }
  }

  MPI_Finalize();
  return 0;
}
