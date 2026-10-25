// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "../lesson.hpp"
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

// Refactored from 20_explicit_stepper_spectral.cpp. The same physical RHS
// now receives Fourier derivatives; the Euler time integrator is unchanged.
int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    const auto p = heat::Parameters::read(argc, argv);
    const double h = p.spacing();
    if (p.dt * p.diffusivity >
        2.0 * h * h / (3.0 * std::numbers::pi * std::numbers::pi)) {
      throw std::invalid_argument(
          "Spectral Euler requires dt * diffusivity <= 2h^2/(3pi^2)");
    }

    // start snippet initialize
    pfc::sim::stacks::SpectralCPUStack stack(heat::domain(p), comm.rank(),
                                             comm.size());
    auto &u = stack.u();
    u.apply([&](double x, double y, double z) {
      return heat::exact(x, y, z, 0.0, p.diffusivity);
    });
    auto du = stack.du<heat::Derivatives>();
    // end snippet initialize

    auto diagnostics = heat::open_diagnostics(comm.rank());
    heat::observe(u, p, 0.0, diagnostics, comm.rank());
    // start snippet evolve
    for (int step = 1; step <= p.steps; ++step) {
      du.apply([&](const heat::Derivatives &g) {
        return p.diffusivity * (g.xx + g.yy + g.zz);
      });
      u += p.dt * du;
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
