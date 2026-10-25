// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "../lesson.hpp"
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/scaled_field.hpp>

using pfc::field::operator*;
using pfc::field::operator+=;

// Refactored from 19_explicit_stepper_fd.cpp. The Euler update is visible;
// The padded field, periodic halo exchange and central differences are explicit.
int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    const auto p = heat::Parameters::read(argc, argv);
    const double h = p.spacing();
    if (p.dt * p.diffusivity > h * h / 6.0) {
      throw std::invalid_argument(
          "Forward Euler requires dt * diffusivity <= h^2 / 6");
    }

    // start snippet initialize
    const auto decomp = pfc::decomposition::create(heat::domain(p), comm.size());
    auto u = pfc::data::field_from_subdomain<double>(decomp, comm.rank(), 1);
    pfc::comm::HaloExchange<pfc::HostSpace, double> halo(u, decomp, comm.rank(),
                                                         comm);
    u.apply([&](double x, double y, double z) {
      return heat::exact(x, y, z, 0.0, p.diffusivity);
    });
    auto gradient = pfc::field::create<heat::Derivatives>(u, 2);
    auto du = pfc::data::field_from_subdomain<double>(decomp, comm.rank(), 1);
    // end snippet initialize

    auto diagnostics = heat::open_diagnostics(comm.rank());
    heat::observe(u, p, 0.0, diagnostics, comm.rank());
    // start snippet evolve
    for (int step = 1; step <= p.steps; ++step) {
      halo.exchange();
      u.for_each_owned([&](int i, int j, int k) {
        const auto g = gradient(i, j, k);
        du(i, j, k) = p.diffusivity * (g.xx + g.yy + g.zz);
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
