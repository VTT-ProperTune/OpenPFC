// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file nodal_dirichlet_nested_geometry.hpp
 * @brief Shared validation of nested free-node Q1 transfer reach on OpenPFC boxes.
 *
 * This validates geometry already partitioned by OpenPFC; it never creates,
 * redistributes or owns a decomposition. Both Host and CUDA/HIP Dirichlet
 * levels use precisely this admission gate for two-cell halo transfers.
 */
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc::solvers::nmg::dirichlet {

inline constexpr int kNestedHalo = 2;

inline void validate_nested_local_width(const decomposition::Decomposition &dec,
                                        int width = kNestedHalo) {
  if (width < 1)
    throw std::invalid_argument("Dirichlet Q1: invalid nested halo width");
  for (int rank = 0; rank < decomposition::get_num_domains(dec); ++rank) {
    const Box3i box = decomposition::local_box(dec, rank);
    for (int d = 0; d < 3; ++d) {
      if (box.size[d] < width ||
          box.size[d] > std::numeric_limits<int>::max() - 2 * width)
        throw std::invalid_argument(
            "Dirichlet Q1: OpenPFC rank-owned brick too thin/large for halo");
    }
  }
}

/// Validate the geometry of P, P^T from an already-created pair of OpenPFC
/// free-node decompositions. The finest grid has 2*nc+1 free nodes per axis.
/// The pre/post-Dirichlet transfer reads only the OpenPFC halo-2 fields.
inline void
validate_nested_transfer_reach(const decomposition::Decomposition &fine,
                               const decomposition::Decomposition &coarse,
                               int halo = kNestedHalo) {
  const int nranks = decomposition::get_num_domains(fine);
  if (nranks < 1 || decomposition::get_num_domains(coarse) != nranks)
    throw std::invalid_argument(
        "Dirichlet Q1: transfer changes MPI decomposition rank count");
  validate_nested_local_width(fine, halo);
  validate_nested_local_width(coarse, halo);

  const auto &fd = decomposition::domain(fine);
  const auto &cd = decomposition::domain(coarse);
  const Int3 nf = domain::get_size(fd), nc = domain::get_size(cd);
  const Real3 hf = domain::get_spacing(fd), hc = domain::get_spacing(cd);
  const auto fp = domain::get_periodic(fd), cp = domain::get_periodic(cd);

  for (int d = 0; d < 3; ++d) {
    if (fp[d] || cp[d] ||
        static_cast<long long>(nf[d]) != 2LL * static_cast<long long>(nc[d]) + 1 ||
        !(hf[d] > 0.0) || !std::isfinite(hf[d]) || !(hc[d] > 0.0) ||
        !std::isfinite(hc[d]) ||
        std::abs(hc[d] - 2.0 * hf[d]) >
            16.0 * std::numeric_limits<double>::epsilon() *
                std::max({std::abs(hc[d]), std::abs(2.0 * hf[d]), 1.0}))
      throw std::invalid_argument(
          "Dirichlet Q1: expected nested nonperiodic free nodes and 2x spacing");
  }

  for (int rank = 0; rank < nranks; ++rank) {
    const Box3i f = decomposition::local_box(fine, rank);
    const Box3i c = decomposition::local_box(coarse, rank);
    for (int d = 0; d < 3; ++d) {
      // P^T samples fine 2*c +/- one free node; P reads the two coarse
      // nodes bracketing each fine node. Mask the true physical exterior.
      const int rlo = std::max(0, 2 * c.low[d]);
      const int rhi = std::min(nf[d] - 1, 2 * c.high[d] + 2);
      const int plo = std::max(0, (f.low[d] + 1) / 2 - 1);
      const int phi = std::min(nc[d] - 1, (f.high[d] + 1) / 2);
      if (rlo < f.low[d] - halo || rhi > f.high[d] + halo || plo < c.low[d] - halo ||
          phi > c.high[d] + halo)
        throw std::invalid_argument(
            "Dirichlet Q1: OpenPFC nested P/P^T exceeds two-cell owned halo");
    }
  }
}

/// Construct an arbitrary-depth hierarchy with *OpenPFC's own* Cartesian
/// decomposition factory. This code chooses geometric grid RESOLUTION only,
/// not MPI ownership; the original process grid is reused at each level.
/// The physical origin of coarse free nodes shifts by one fine spacing.
[[nodiscard]] inline std::vector<decomposition::Decomposition>
make_nested_decompositions(const decomposition::Decomposition &fine) {
  const auto &d = decomposition::domain(fine);
  const auto n = domain::get_size(d);
  const auto p = domain::get_periodic(d);
  const auto is_pow2 = [](long long value) {
    return value > 0 && (value & (value - 1)) == 0;
  };
  if (n[0] < 15 || n[1] != n[0] ||
      !(n[2] == n[0] || static_cast<long long>(n[2]) == 2LL * n[0] + 1) ||
      !is_pow2(static_cast<long long>(n[0]) + 1) ||
      !is_pow2(static_cast<long long>(n[2]) + 1) || p[0] || p[1] || p[2])
    throw std::invalid_argument(
        "Dirichlet Q1 hierarchy: expected nonperiodic cubic/2:1 2^k-1 free nodes");
  for (int d_axis = 0; d_axis < 3; ++d_axis)
    if (!(d.spacing[d_axis] > 0.0) || !std::isfinite(d.spacing[d_axis]) ||
        !std::isfinite(d.origin[d_axis]))
      throw std::invalid_argument(
          "Dirichlet Q1 hierarchy: nonfinite spacing or origin");

  std::vector<decomposition::Decomposition> levels{fine};
  auto current = d;
  const auto grid = decomposition::get_grid(fine);
  while (current.size[0] > 7) {
    Int3 next_size{};
    Real3 spacing{}, origin{};
    for (int axis = 0; axis < 3; ++axis) {
      next_size[axis] = (current.size[axis] - 1) / 2;
      spacing[axis] = 2.0 * current.spacing[axis];
      origin[axis] = current.origin[axis] + current.spacing[axis];
      // Inadequately sized owned subdomains cannot support full width-2
      // halos. Reject before decomposition/communicator construction.
      if (next_size[axis] < 2 * grid[axis] || !std::isfinite(spacing[axis]) ||
          !std::isfinite(origin[axis]))
        throw std::invalid_argument(
            "Dirichlet Q1 hierarchy: insufficient coarse MPI owned width");
    }
    const auto next =
        domain::create(GridSize(next_size), PhysicalOrigin(origin),
                       GridSpacing(spacing), Bool3{false, false, false});
    auto next_dec = decomposition::create(next, grid);
    validate_nested_transfer_reach(levels.back(), next_dec);
    levels.push_back(std::move(next_dec));
    current = next;
  }
  if (levels.size() < 2)
    throw std::invalid_argument("Dirichlet Q1 hierarchy: need two or more levels");
  return levels;
}

} // namespace pfc::solvers::nmg::dirichlet
