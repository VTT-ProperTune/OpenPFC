// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <openpfc/kernel/decomposition/comm_sparse_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    const auto p = heat::Parameters::read(argc, argv);
    const double inverse_h2 = 1 / std::pow(p.spacing(), 2);
    if (p.dt * p.diffusivity * inverse_h2 > 1.0 / 6)
      throw std::invalid_argument("Require dt*diffusivity/h^2 <= 1/6");
    const auto partition = pfc::decomposition::create(heat::domain(p), comm.size());
    // start snippet storage
    auto u = pfc::data::field_from_subdomain<double>(partition, comm.rank(), 0);
    auto laplacian =
        pfc::data::field_from_subdomain<double>(partition, comm.rank(), 0);
    u.apply([&](double x, double y, double z) {
      return heat::exact(x, y, z, 0, p.diffusivity);
    });
    constexpr int width = 1;
    pfc::comm::SparseExchange<pfc::HostSpace, double> exchange(
        u.vec(), partition, comm.rank(), comm, width);
    auto faces =
        pfc::halo::allocate_face_halos<double>(partition, comm.rank(), width);
    std::array<const double *, 6> face_pointers;
    for (int face = 0; face < 6; ++face) face_pointers[face] = faces[face].data();
    const auto size = u.box().size;
    // end snippet storage
    auto diagnostics = heat::open_diagnostics(comm.rank());
    heat::observe(u, p, 0, diagnostics, comm.rank());
    // start snippet exchange
    for (int step = 1; step <= p.steps; ++step) {
      exchange.exchange();
      pfc::halo::copy_to_face_layout(exchange.halos(), faces);
      pfc::field::fd::laplacian_periodic_separated<2>(
          u.vec().data(), face_pointers, laplacian.vec().data(), size[0], size[1],
          size[2], inverse_h2, inverse_h2, inverse_h2, width);
      for (std::size_t i = 0; i < u.size(); ++i)
        u.vec()[i] += p.dt * p.diffusivity * laplacian.vec()[i];
      heat::observe(u, p, step * p.dt, diagnostics, comm.rank());
    }
    // end snippet exchange
    heat::write_field(u, p, comm.rank());
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
