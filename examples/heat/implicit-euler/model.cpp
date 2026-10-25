// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    const auto p = heat::Parameters::read(argc, argv);
    pfc::sim::stacks::SpectralCPUStack stack(heat::domain(p), comm.rank(),
                                             comm.size());
    auto &u = stack.u();
    auto &fft = stack.fft();
    u.apply([&](double x, double y, double z) {
      return heat::exact(x, y, z, 0, p.diffusivity);
    });
    // start snippet implicit
    std::vector<std::complex<double>> modes(fft.size_outbox());
    std::vector<double> denominator(modes.size());
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), heat::domain(p),
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          denominator[i] = 1 + p.dt * p.diffusivity * (kx * kx + ky * ky + kz * kz);
        });
    auto diagnostics = heat::open_diagnostics(comm.rank());
    heat::observe(u, p, 0, diagnostics, comm.rank());
    for (int step = 1; step <= p.steps; ++step) {
      fft.forward(u.vec(), modes);
      for (std::size_t i = 0; i < modes.size(); ++i) modes[i] /= denominator[i];
      fft.backward(modes, u.vec());
      heat::observe(u, p, step * p.dt, diagnostics, comm.rank());
    }
    // end snippet implicit
    heat::write_field(u, p, comm.rank());
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
