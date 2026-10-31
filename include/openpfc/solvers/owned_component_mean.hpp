// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/solvers/vector_operations.hpp>
#include <stdexcept>
#include <vector>
namespace pfc::solvers {
/// Subtract the communicator-wide mean of each interleaved component.
/// `owned[i]` is the global degree of freedom stored in `values[i]`. The
/// mean divides by `nodes`, the global node count, not by the local entry
/// count and not by a ghost count. `StructuredGrid::project` and the device
/// `VectorOperations::project` stay local. One `dots` reduction covers every
/// component. `MPI_COMM_NULL` performs the sum on this rank only.
inline void remove_owned_component_mean(std::vector<double> &values,
                                        const std::vector<int> &owned,
                                        int components, int nodes, MPI_Comm comm) {
  if (components < 1 || nodes < 1)
    throw std::invalid_argument("owned component mean needs components and nodes");
  if (values.size() != owned.size())
    throw std::invalid_argument("owned component mean size mismatch");
  for (int index : owned)
    if (index < 0)
      throw std::invalid_argument("owned component mean index is negative");
  VectorOperations ops(comm);
  std::vector<std::vector<double>> masks;
  masks.reserve(static_cast<std::size_t>(components));
  for (int component = 0; component < components; ++component) {
    auto mask = ops.make(values.size());
    for (std::size_t i = 0; i < owned.size(); ++i)
      mask[i] = owned[i] % components == component ? 1 : 0;
    masks.push_back(std::move(mask));
  }
  std::vector<double> sums(static_cast<std::size_t>(components));
  ops.dots(masks.data(), components, values, sums.data());
  for (int component = 0; component < components; ++component)
    ops.axpy(values, -sums[static_cast<std::size_t>(component)] / nodes,
             masks[static_cast<std::size_t>(component)]);
}
} // namespace pfc::solvers
