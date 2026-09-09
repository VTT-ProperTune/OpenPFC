// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file order_parameter.hpp
 * @brief Real-space bond-orientational order metric for PFC crystal
 *        classification (`#118`).
 *
 * @details
 * `#118`'s complaint is that the existing report infers "square" or
 * "triangular" from which reciprocal-space ring carries power. A ring is a
 * necessary but not sufficient witness: two independent plane-wave families
 * at \f$|k|=1\f$ can beat together into several different real-space
 * patterns, only one of which is actually square. This header adds the
 * missing real-space check.
 *
 * The metric is the standard bond-orientational order parameter used to tell
 * 2D crystal symmetries apart (Nelson & Halperin, Phys. Rev. B 19, 2457
 * (1979), for the melting-transition context the parameter was introduced
 * in): treat each density peak as a "particle", find its near neighbours, and
 * average \f$e^{in\theta}\f$ over the bond angles \f$\theta\f$ to those
 * neighbours,
 *
 * \f[
 *   \psi_n(j) = \frac{1}{N_b(j)}\sum_{k \in \text{neighbours}(j)} e^{in\theta_{jk}} .
 * \f]
 *
 * A perfect square lattice with its 4 nearest neighbours gives
 * \f$|\psi_4|=1,\ \psi_6=0\f$ exactly (bond angles \f$0,\tfrac\pi2,\pi,
 * \tfrac{3\pi}2\f$: \f$4\theta\f$ is a multiple of \f$2\pi\f$ every time,
 * \f$6\theta\f$ alternates sign in pairs and cancels). A perfect triangular
 * lattice with its 6 nearest neighbours gives the reverse, \f$\psi_4=0,\
 * |\psi_6|=1\f$ exactly (bond angles \f$60^\circ\f$ apart). That clean
 * separation, verified analytically in the test suite, is what makes
 * \f$(\psi_4,\psi_6)\f$ a real structural classifier rather than another
 * reciprocal-space proxy.
 *
 * Two averages are reported for each order:
 *
 * - **global**: the magnitude of the *complex* average of \f$\psi_n(j)\f$
 *   over all peaks. Grains at different orientations have \f$\psi_n(j)\f$
 *   pointing in different directions, so their contributions partially
 *   cancel; a noise-seeded polycrystal gives a suppressed global value even
 *   when every grain is locally well ordered. This is the number that
 *   answers "is this one crystal or many misoriented grains".
 * - **local**: the average of \f$|\psi_n(j)|\f$, orientation blind. This
 *   answers "is each neighbourhood locally ordered at all", independent of
 *   whether the grains agree with each other.
 *
 * Neighbours are found within a cutoff derived from the point set's own
 * nearest-neighbour distance (`neighbour_cutoff_factor` times the median
 * nearest-neighbour distance), not a value the caller has to know the
 * lattice spacing to supply. `1.3` sits strictly between the square lattice's
 * first shell (distance \f$a\f$) and its diagonal second shell
 * (\f$a\sqrt2\approx1.41a\f$), and strictly below the triangular lattice's
 * second shell (\f$a\sqrt3\approx1.73a\f$), so the default admits exactly the
 * first neighbour shell for both lattices without extra tuning.
 *
 * Peak detection on a simulated field is a periodic 8-neighbour local-maximum
 * test at grid resolution (no sub-cell refinement); see `detect_peaks`. That
 * is coarser than a crystallographer would want for a lattice-constant
 * measurement, which is why this module reports *symmetry* (a normalised
 * ratio), not absolute peak positions -- the reciprocal-space report in
 * `free_energy.hpp` already gives sub-bin-averaged wavenumbers for that.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>

namespace higher_order_pfc {

/// A detected density peak, in physical (not grid-index) coordinates.
struct Peak {
  double x{0.0};
  double y{0.0};
};

/// Bond-orientational order for one harmonic \f$n\f$, both averages.
struct BondOrder {
  double global{0.0};  ///< |mean_j psi_n(j)|      -- single-crystal check
  double local{0.0};   ///< mean_j |psi_n(j)|       -- local-order check
};

/// \f$\psi_4\f$ and \f$\psi_6\f$ together, plus how many peaks/bonds went in.
struct BondOrientationalOrder {
  BondOrder psi4{};
  BondOrder psi6{};
  std::size_t n_points{0};
  double mean_neighbours{0.0};
};

namespace detail {
[[nodiscard]] inline double wrap(double d, double L) {
  if (L <= 0.0) return d;
  return d - L * std::round(d / L);
}
} // namespace detail

/**
 * @brief Bond-orientational order parameters for a periodic 2D point set.
 *
 * @param points  detected peak / lattice-site positions
 * @param Lx, Ly  periodic box size (minimum-image convention)
 * @param neighbour_cutoff_factor  cutoff = this * median nearest-neighbour
 *        distance; see the file comment for why 1.3 separates both lattices'
 *        first shell cleanly
 *
 * \f$O(N^2)\f$ in the number of points; fine for the peak counts a PFC
 * benchmark grid produces (hundreds, not millions).
 */
[[nodiscard]] inline BondOrientationalOrder
bond_orientational_order(const std::vector<Peak> &points, double Lx, double Ly,
                         double neighbour_cutoff_factor = 1.3) {
  BondOrientationalOrder out;
  const std::size_t N = points.size();
  out.n_points = N;
  if (N < 2) return out;

  // Median nearest-neighbour distance sets the cutoff, so the caller never
  // has to know the lattice spacing in advance.
  std::vector<double> nn(N, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < N; ++j) {
      if (i == j) continue;
      const double dx = detail::wrap(points[j].x - points[i].x, Lx);
      const double dy = detail::wrap(points[j].y - points[i].y, Ly);
      best = std::min(best, dx * dx + dy * dy);
    }
    nn[i] = std::sqrt(best);
  }
  std::vector<double> sorted_nn = nn;
  std::sort(sorted_nn.begin(), sorted_nn.end());
  const double median_nn = sorted_nn[sorted_nn.size() / 2];
  const double r_cut = neighbour_cutoff_factor * median_nn;
  const double r_cut2 = r_cut * r_cut;

  std::complex<double> sum4{0.0, 0.0}, sum6{0.0, 0.0};
  double local4 = 0.0, local6 = 0.0;
  double total_neighbours = 0.0;
  std::size_t counted = 0;
  for (std::size_t i = 0; i < N; ++i) {
    std::complex<double> p4{0.0, 0.0}, p6{0.0, 0.0};
    std::size_t nb = 0;
    for (std::size_t j = 0; j < N; ++j) {
      if (i == j) continue;
      const double dx = detail::wrap(points[j].x - points[i].x, Lx);
      const double dy = detail::wrap(points[j].y - points[i].y, Ly);
      if (dx * dx + dy * dy > r_cut2) continue;
      const double theta = std::atan2(dy, dx);
      p4 += std::complex<double>(std::cos(4.0 * theta), std::sin(4.0 * theta));
      p6 += std::complex<double>(std::cos(6.0 * theta), std::sin(6.0 * theta));
      ++nb;
    }
    if (nb == 0) continue;
    p4 /= double(nb);
    p6 /= double(nb);
    sum4 += p4;
    sum6 += p6;
    local4 += std::abs(p4);
    local6 += std::abs(p6);
    total_neighbours += double(nb);
    ++counted;
  }
  if (counted == 0) return out;
  out.psi4.global = std::abs(sum4) / double(counted);
  out.psi6.global = std::abs(sum6) / double(counted);
  out.psi4.local = local4 / double(counted);
  out.psi6.local = local6 / double(counted);
  out.mean_neighbours = total_neighbours / double(counted);
  return out;
}

/**
 * @brief Ideal square lattice point set on a periodic \f$L\times L\f$ box.
 *
 * `n_cells` lattice constants per side, so `Lx = Ly = n_cells * a` is exactly
 * periodic. Used only by the analytical unit tests -- the whole point is to
 * check `bond_orientational_order` against a lattice built independently of
 * any peak detector.
 */
[[nodiscard]] inline std::vector<Peak> ideal_square_lattice(double a, int n_cells) {
  std::vector<Peak> pts;
  pts.reserve(std::size_t(n_cells) * std::size_t(n_cells));
  for (int j = 0; j < n_cells; ++j)
    for (int i = 0; i < n_cells; ++i) pts.push_back({i * a, j * a});
  return pts;
}

/**
 * @brief Ideal triangular lattice point set on a periodic box.
 *
 * Primitive vectors \f$a_1=(a,0)\f$, \f$a_2=(a/2,a\sqrt3/2)\f$, so the
 * commensurate rectangular box is \f$L_x=n_x a\f$,
 * \f$L_y=n_y a\sqrt3\f$ with \f$n_y\f$ rows of two offset sub-rows each.
 */
[[nodiscard]] inline std::vector<Peak> ideal_triangular_lattice(double a, int n_x,
                                                                int n_y) {
  std::vector<Peak> pts;
  const double row_h = a * std::numbers::sqrt3 / 2.0;
  for (int j = 0; j < 2 * n_y; ++j) {
    const double y = j * row_h;
    const double x_offset = (j % 2 == 0) ? 0.0 : 0.5 * a;
    for (int i = 0; i < n_x; ++i) pts.push_back({i * a + x_offset, y});
  }
  return pts;
}

/**
 * @brief Local density maxima on a periodic 2D real-space field.
 *
 * A grid point is a peak if it is `>=` all 8 periodic neighbours and strictly
 * above the field mean (drops flat/interstitial regions in a disordered or
 * still-relaxing field). Grid resolution only, no sub-cell refinement -- see
 * the file comment for why that is acceptable here.
 *
 * @param psi   whole-domain field (single rank; box must cover the domain)
 * @param nx,ny grid size
 * @param dx,dy grid spacing
 */
[[nodiscard]] inline std::vector<Peak>
detect_peaks(const pfc::data::Field<double> &psi, int nx, int ny, double dx,
            double dy) {
  double mean = 0.0;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) mean += psi(i, j, 0);
  mean /= double(nx) * double(ny);

  std::vector<Peak> peaks;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double v = psi(i, j, 0);
      if (v <= mean) continue;
      bool is_peak = true;
      for (int dj = -1; dj <= 1 && is_peak; ++dj) {
        for (int di = -1; di <= 1; ++di) {
          if (di == 0 && dj == 0) continue;
          const int ii = ((i + di) % nx + nx) % nx;
          const int jj = ((j + dj) % ny + ny) % ny;
          if (psi(ii, jj, 0) > v) {
            is_peak = false;
            break;
          }
        }
      }
      if (is_peak) peaks.push_back({i * dx, j * dy});
    }
  }
  return peaks;
}

} // namespace higher_order_pfc
