// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file interface_diffusion.hpp
 * @brief Matrix-free Cartesian diffusion with transmission at an interface.
 *
 * The equation is -div(κ ∇u) = f. κ is κ_minus where φ < 0 and κ_plus
 * where φ > 0. The jump is the outside value minus the inside value,
 * with n = ∇φ/|∇φ| pointing toward φ > 0:
 *
 *   [u] = u+ - u-
 *   [κ ∂n u] = κ+ ∂n u+ - κ- ∂n u-
 *
 * Both are zero for the operator built here. Edges the interface does not
 * cross keep the standard two-point flux. A crossed edge uses the series
 * resistance ∫ ds/κ. Nodes that touch such an edge can instead use a
 * calibrated 7-point row that is exact for a planar interface and every
 * piecewise-linear field satisfying the two jump conditions. That row is
 * not a harmonic average, and it is not required to be symmetric.
 *
 * Regular nodes are not stored. Only interface rows are packed.
 */

#include <openpfc/kernel/geometry/implicit.hpp>

#include <array>
#include <cmath>
#include <vector>

namespace pfc::solvers {

enum class TransmissionClosure { Nodal, Series, Calibrated };

struct TransmissionRequest {
  int cells = 16;
  double kappa_minus = 1;
  double kappa_plus = 10;
  TransmissionClosure closure = TransmissionClosure::Calibrated;
  /// Segments shorter than this fraction of h are merged into the node.
  double theta_floor = 0;
};

struct TransmissionRow {
  int node = 0;
  int n = 0;
  int column[7]{};
  double value[7]{};
};

struct TransmissionOperator {
  int cells = 0;
  double spacing = 0;
  int nodes_per_axis = 0;
  int node_count = 0;
  double kappa_minus = 1;
  double kappa_plus = 1;
  TransmissionClosure closure = TransmissionClosure::Calibrated;
  geometry::Implicit shape{};
  std::vector<TransmissionRow> interface_row;
  std::vector<int> interface_at;
  int cut_edges = 0;
  int calibration_fallbacks = 0;

  int index(int i, int j, int k) const {
    return i + nodes_per_axis * (j + nodes_per_axis * k);
  }
  bool boundary(int i, int j, int k) const {
    return i == 0 || j == 0 || k == 0 || i == cells || j == cells || k == cells;
  }
  geometry::Vec3 point(int i, int j, int k) const {
    return {i * spacing, j * spacing, k * spacing};
  }
  void apply(const std::vector<double> &u, std::vector<double> &y) const;
  double diagonal(int node) const;
};

TransmissionOperator build_transmission(const geometry::Implicit &shape,
                                        const TransmissionRequest &request);

} // namespace pfc::solvers

#include <openpfc/solvers/interface_diffusion.inl>
