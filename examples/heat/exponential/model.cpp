// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "../lesson.hpp"
#include <complex>
#include <vector>

#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

// The forward/multiply/backward pattern comes from 04_diffusion_model.cpp.
// Here exp(-D |k|^2 dt) integrates linear heat flow exactly in Fourier space.
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
      return heat::exact(x, y, z, 0.0, p.diffusivity);
    });

    // start snippet propagator
    std::vector<std::complex<double>> spectrum(fft.size_outbox());
    std::vector<double> decay(fft.size_outbox());
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), heat::domain(p),
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          decay[i] = std::exp(-p.diffusivity * (kx * kx + ky * ky + kz * kz) * p.dt);
        });
    // end snippet propagator

    auto diagnostics = heat::open_diagnostics(comm.rank());
    heat::observe(u, p, 0.0, diagnostics, comm.rank());
    // start snippet evolve
    for (int step = 1; step <= p.steps; ++step) {
      fft.forward(u.vec(), spectrum);
      for (std::size_t i = 0; i < spectrum.size(); ++i) spectrum[i] *= decay[i];
      fft.backward(spectrum, u.vec());
      heat::observe(u, p, step * p.dt, diagnostics, comm.rank());
    }
    // end snippet evolve
    heat::write_field(u, p, comm.rank());
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
