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

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc::solvers::nmg::dirichlet {

inline constexpr int kNestedHalo = 2;

inline void validate_nested_local_width(
    const decomposition::Decomposition &dec, int width = kNestedHalo) {
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
inline void validate_nested_transfer_reach(
    const decomposition::Decomposition &fine,
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
        static_cast<long long>(nf[d]) !=
            2LL * static_cast<long long>(nc[d]) + 1 ||
        !(hf[d] > 0.0) || !std::isfinite(hf[d]) ||
        !(hc[d] > 0.0) || !std::isfinite(hc[d]) ||
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
      if (rlo < f.low[d] - halo || rhi > f.high[d] + halo ||
          plo < c.low[d] - halo || phi > c.high[d] + halo)
        throw std::invalid_argument(
            "Dirichlet Q1: OpenPFC nested P/P^T exceeds two-cell owned halo");
    }
  }
}

} // namespace pfc::solvers::nmg::dirichlet
