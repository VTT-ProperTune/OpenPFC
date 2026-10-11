// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <chrono>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>
#include <openpfc/solvers/multigrid.hpp>
namespace pfc::solvers {
/** Halo-based scalar periodic FD action on owned vectors, for the unit cube.
 * Applies -Laplacian times cell volume, matching poisson_auxiliary.
 * Reuses OpenPFC padded fields, halo exchange and finite-difference kernel.
 * Decomposition must outlive this action; each rank has one nonempty brick.
 */
class PeriodicPoissonAction {
  data::Field<double, HostSpace> field_, laplacian_;
  comm::HaloExchange<HostSpace, double> halo_;
  StructuredGrid grid_;

public:
  double communication_seconds = 0;
  PeriodicPoissonAction(const decomposition::Decomposition &decomp, int rank,
                        MPI_Comm communicator)
      : field_(data::field_from_subdomain<double>(decomp, rank, 1)),
        laplacian_(data::field_from_subdomain<double>(decomp, rank, 1)),
        halo_(field_, decomp, rank, communicator),
        grid_{domain::get_size(decomposition::domain(decomp)), 1,
              Boundary::Periodic} {
    grid_.validate();
    // This adapter's spacing is fixed by its unit-cube contract.
  }
  std::vector<int> owned_indices() const {
    const auto box = field_.box();
    auto n = field_.size3();
    std::vector<int> indices;
    for (int z = 0; z < n[2]; ++z)
      for (int y = 0; y < n[1]; ++y)
        for (int x = 0; x < n[0]; ++x)
          indices.push_back(
              grid_.node(x + box.low[0], y + box.low[1], z + box.low[2]));
    return indices;
  }
  void operator()(const std::vector<double> &x, std::vector<double> &y) {
    const auto n = field_.size3();
    if (x.size() != static_cast<std::size_t>(n[0] * n[1] * n[2]))
      throw std::invalid_argument("Poisson owned vector mismatch");
    std::size_t index = 0;
    for (int z = 0; z < n[2]; ++z)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) field_(i, j, z) = x[index++];
    auto t = std::chrono::steady_clock::now();
    halo_.exchange();
    communication_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    field::fd::laplacian_interior<2>(field_.data(), laplacian_.data(), n[0] + 2,
                                     n[1] + 2, n[2] + 2,
                                     double(grid_.shape[0] * grid_.shape[0]),
                                     double(grid_.shape[1] * grid_.shape[1]),
                                     double(grid_.shape[2] * grid_.shape[2]), 1);
    double volume = 1.0 / grid_.nodes();
    y.resize(x.size());
    index = 0;
    for (int z = 0; z < n[2]; ++z)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) y[index++] = -volume * laplacian_(i, j, z);
  }
};
} // namespace pfc::solvers
