// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_flux.hpp
 * @brief Conservative face-flux finite-difference stepper for the thin film
 *        equation, distributed over MPI via a separated halo exchange.
 *
 * @details
 * Solves the same equation as the spectral solver in `nonlinear.hpp`,
 *
 * \f[
 *   \partial_t h = \nabla\cdot\bigl[M(h)\nabla p\bigr],\qquad
 *   p = -\gamma\nabla^2 h - \Pi(h),\qquad
 *   M(h) = M_0(h/h_0)^3,
 * \f]
 *
 * but where the spectral solver forms `M(h) grad p` pointwise in real space
 * and transforms the whole product (so the flux only exists implicitly, as
 * whatever the FFT of a real-space product happens to be), this stepper
 * forms the flux **at cell faces**:
 *
 * \f[
 *   F_{i+1/2,j} = M_{i+1/2,j}\,\frac{p_{i+1,j}-p_{i,j}}{\Delta x},
 *   \qquad
 *   \frac{dh_{i,j}}{dt} = \frac{F_{i+1/2,j}-F_{i-1/2,j}}{\Delta x}
 *                        + \frac{F_{i,j+1/2}-F_{i,j-1/2}}{\Delta y}.
 * \f]
 *
 * Two structural properties fall out of this that the spectral scheme does
 * not have:
 *
 * 1. **Exact mass conservation.** Every interior face flux is computed
 *    identically by the two ranks (or two cells) that share it -- same two
 *    `h`/`p` values, same symmetric averaging formula, hence the same
 *    floating-point result -- so it cancels in the telescoping sum
 *    `sum_cells dh/dt` to machine round-off, for *any* timestep. The
 *    spectral scheme's flux only telescopes in the continuum; discretely it
 *    conserves volume because a Fourier mode with `k=0` is untouched, not
 *    because of face-by-face cancellation.
 * 2. **Positivity under a degenerate mobility.** `M(h) -> 0` as `h -> 0`.
 *    With the *harmonic* mean of the two neighbouring mobilities at a face,
 *    `M_face = 2 M_L M_R / (M_L + M_R)`, the face mobility itself vanishes
 *    whenever either neighbour is dry, so the flux out of (or into) a
 *    vanishing cell vanishes with it: an empty cell cannot be driven
 *    negative by drainage. The arithmetic mean `M_face = (M_L+M_R)/2` does
 *    not have this property -- it only halves the flux, so a dry cell can
 *    still be pulled below zero by a wet neighbour. Both are provided
 *    (`FaceMobility::Harmonic`, `FaceMobility::Arithmetic`); see
 *    `apps/thin_film/README.md` for the measured difference.
 *
 * The curvature term `-gamma*lap(h)` inside `p` is formed with the shared
 * order-2 or order-4 central Laplacian
 * (`pfc::field::fd::laplacian2d_xy_periodic_separated`); the face-flux
 * difference itself is always the natural 2-point (second-order) form,
 * since that is what "flux at a face" means for a finite-volume update --
 * only the curvature operator's order is a free choice here.
 *
 * MPI parallelism reuses the same building blocks as `apps/allen_cahn`
 * (`pfc::decomposition`, `pfc::comm::SparseExchange`,
 * `pfc::halo::allocate_face_halos` / `copy_to_face_layout`): a "separated"
 * halo layout where the owned cells are a plain, unpadded `nx*ny` buffer and
 * the neighbour layers live in a side array, indexed exactly as
 * `pfc::field::fd::laplacian2d_xy_periodic_separated` expects.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_sparse_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

#include <thin_film/nonlinear.hpp>
#include <thin_film/thin_film_pointwise.hpp>

namespace thin_film {

/// How the two cell-centred mobilities either side of a face are combined.
enum class FaceMobility { Arithmetic, Harmonic };

/**
 * @brief Combine the mobility on either side of a face into one face value.
 *
 * `Harmonic` is the positivity-preserving choice for a degenerate mobility:
 * it is zero whenever either side is zero, regardless of the other side.
 * `Arithmetic` only halves the flux in that situation, which is not enough
 * to stop a dry cell from being pulled negative -- see the "Face mobility
 * choice" measurement in the README.
 */
[[nodiscard]] inline double face_mobility(FaceMobility kind, double m_left,
                                          double m_right) {
  switch (kind) {
  case FaceMobility::Harmonic: {
    const double s = m_left + m_right;
    return (s > 0.0) ? (2.0 * m_left * m_right / s) : 0.0;
  }
  case FaceMobility::Arithmetic:
  default:
    return 0.5 * (m_left + m_right);
  }
}

namespace detail {

/// Value at `(ix+1, iy)`, reading the owned buffer or the +X halo slab.
[[nodiscard]] inline double at_xp(const double *core, const double *hpx, int ix,
                                  int iy, int nx, int hw) {
  return (ix + 1 < nx) ? core[(ix + 1) + iy * nx] : hpx[iy * hw + 0];
}
/// Value at `(ix-1, iy)`, reading the owned buffer or the -X halo slab.
[[nodiscard]] inline double at_xm(const double *core, const double *hnx, int ix,
                                  int iy, int nx, int hw) {
  return (ix - 1 >= 0) ? core[(ix - 1) + iy * nx] : hnx[iy * hw + (hw - 1)];
}
/// Value at `(ix, iy+1)`, reading the owned buffer or the +Y halo slab.
[[nodiscard]] inline double at_yp(const double *core, const double *hpy, int ix,
                                  int iy, int nx, int ny, int /*hw*/) {
  return (iy + 1 < ny) ? core[ix + (iy + 1) * nx] : hpy[ix];
}
/// Value at `(ix, iy-1)`, reading the owned buffer or the -Y halo slab.
[[nodiscard]] inline double at_ym(const double *core, const double *hny, int ix,
                                  int iy, int nx, int hw) {
  return (iy - 1 >= 0) ? core[ix + (iy - 1) * nx] : hny[(hw - 1) * nx + ix];
}

/// Runtime dispatch over the two supported curvature-operator orders.
inline void laplacian2d_dispatch(int order, const double *core,
                                 const std::array<const double *, 6> &face_halos,
                                 double *lap, int nx, int ny, double inv_dx2,
                                 double inv_dy2, int hw) {
  switch (order) {
  case 2:
    pfc::field::fd::laplacian2d_xy_periodic_separated<2>(
        core, face_halos, lap, nx, ny, 1, inv_dx2, inv_dy2, hw);
    return;
  case 4:
    pfc::field::fd::laplacian2d_xy_periodic_separated<4>(
        core, face_halos, lap, nx, ny, 1, inv_dx2, inv_dy2, hw);
    return;
  default:
    throw std::invalid_argument(
        "FDFluxSolver: order must be 2 or 4 (curvature-operator order)");
  }
}

} // namespace detail

/**
 * @brief Conservative face-flux thin-film stepper, one MPI rank's share.
 *
 * Owns the scratch buffers (`p`, `lap_h`, `dh/dt`) and the two halo
 * exchangers (one for `h`, at the curvature operator's halo width; one for
 * `p`, always width 1, since the face flux is always the natural 2-point
 * difference). The caller owns the state vector `h` itself and passes the
 * same buffer to every call -- `FDFluxSolver` binds its `h` exchanger to
 * `h.data()` at construction, so `h` must not be resized afterwards.
 */
class FDFluxSolver {
public:
  /**
   * @param domain Global grid geometry (2-D: `Lz` / grid size along z must
   *               be 1).
   * @param decomp Domain decomposition (`pfc::decomposition::create`).
   * @param rank   This MPI rank.
   * @param comm   Communicator matching @p decomp.
   * @param h      Owned-cell state buffer, already sized to this rank's
   *               local `nx*ny`; kept bound for the lifetime of the solver.
   * @param order  Curvature-operator (`lap(h)`) order: 2 or 4.
   */
  FDFluxSolver(const pfc::Domain &domain,
               const pfc::decomposition::Decomposition &decomp, int rank,
               MPI_Comm comm, std::vector<double> &h, int order = 2)
      : domain_(domain), decomp_(decomp), rank_(rank), order_(order),
        hw_(order / 2),
        nx_(pfc::decomposition::local_box(decomp_, rank_).size[0]),
        ny_(pfc::decomposition::local_box(decomp_, rank_).size[1]),
        dx_(pfc::domain::get_spacing(domain_)[0]),
        dy_(pfc::domain::get_spacing(domain_)[1]),
        p_(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
        lap_h_(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
        dhdt_(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
        // Analytic (not decomposition-validated) counts: this is a strictly
        // 2-D solver (`nz == 1`), so the Z faces are never exchanged, and
        // `pfc::halo::allocate_face_halos(decomp, rank, hw)` would otherwise
        // reject `hw > 1` outright -- it sizes the Z slabs from the *global*
        // Z extent, which is 1 here, regardless of the curvature operator's
        // in-plane halo width.
        h_halos_(pfc::halo::allocate_face_halos<double>(
            pfc::halo::face_halo_counts_analytic(nx_, ny_, 1, hw_))),
        p_halos_(pfc::halo::allocate_face_halos<double>(
            pfc::halo::face_halo_counts_analytic(nx_, ny_, 1, 1))),
        exch_h_(h.data(), h.size(), decomp_, rank_, comm, hw_,
               {.dirs = pfc::halo::presets::Axes2D()}),
        exch_p_(p_.data(), p_.size(), decomp_, rank_, comm, 1,
               {.dirs = pfc::halo::presets::Axes2D()}) {
    if (order_ != 2 && order_ != 4) {
      throw std::invalid_argument("FDFluxSolver: order must be 2 or 4");
    }
    if (pfc::decomposition::local_box(decomp_, rank_).size[2] != 1) {
      throw std::invalid_argument("FDFluxSolver: domain must be 2-D (Lz == 1)");
    }
    if (h.size() != p_.size()) {
      throw std::invalid_argument(
          "FDFluxSolver: h must be sized to this rank's local nx*ny");
    }
  }

  [[nodiscard]] int nx() const noexcept { return nx_; }
  [[nodiscard]] int ny() const noexcept { return ny_; }
  [[nodiscard]] double dx() const noexcept { return dx_; }
  [[nodiscard]] double dy() const noexcept { return dy_; }
  [[nodiscard]] int order() const noexcept { return order_; }
  [[nodiscard]] const pfc::decomposition::Decomposition &decomposition() const {
    return decomp_;
  }
  /// Pressure field from the most recent `compute_rhs` / `step` call.
  [[nodiscard]] const std::vector<double> &pressure() const { return p_; }

  /**
   * @brief Evaluate `dh/dt` for the current state, without advancing it.
   *
   * Exposed separately from `step()` so tests can assert exact mass
   * conservation (`sum(dh/dt) == 0` to round-off, for any `h`, before any
   * timestep even enters the picture) and so a caller could build a
   * higher-order explicit integrator (RK2/RK3) on top.
   */
  /**
   * @tparam Mobility Callable `double operator()(double h) const`. Usually
   *         `thin_film::CubicMobility`, but any functor works -- a lambda
   *         returning a constant is what lets the linear `k^4`-decay test
   *         reproduce the analytical constant-mobility limit without going
   *         through the nonlinear `h^3` law at all.
   */
  template <class Mobility>
  void compute_rhs(std::vector<double> &h, double gamma, const ThinFilmPointwise &pw,
                   const Mobility &mobility, FaceMobility kind,
                   std::vector<double> &dhdt_out) {
    // 1. h halo exchange, then the curvature operator lap(h) on the full
    //    owned domain.
    exch_h_.exchange();
    pfc::halo::copy_to_face_layout(exch_h_.halos(), h_halos_);
    std::array<const double *, 6> h_face_ptrs{
        h_halos_[0].data(), h_halos_[1].data(), h_halos_[2].data(),
        h_halos_[3].data(), h_halos_[4].data(), h_halos_[5].data()};
    const double inv_dx2 = 1.0 / (dx_ * dx_);
    const double inv_dy2 = 1.0 / (dy_ * dy_);
    detail::laplacian2d_dispatch(order_, h.data(), h_face_ptrs, lap_h_.data(), nx_,
                                 ny_, inv_dx2, inv_dy2, hw_);

    // 2. p = -gamma*lap(h) - Pi(h), owned cells only -- Pi is pointwise.
    for (std::size_t c = 0; c < p_.size(); ++c) {
      p_[c] = -gamma * lap_h_[c] - pw.Pi(h[c]);
    }

    // 3. p halo exchange (width 1: the flux only ever needs the nearest
    //    neighbour, whatever order the curvature operator used).
    exch_p_.exchange();
    pfc::halo::copy_to_face_layout(exch_p_.halos(), p_halos_);

    // 4. Face fluxes and their divergence.
    for (int iy = 0; iy < ny_; ++iy) {
      for (int ix = 0; ix < nx_; ++ix) {
        const std::size_t c =
            static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * nx_;
        const double hc = h[c];
        const double pc = p_[c];

        const double h_xp = detail::at_xp(h.data(), h_halos_[0].data(), ix, iy,
                                          nx_, hw_);
        const double h_xm = detail::at_xm(h.data(), h_halos_[1].data(), ix, iy,
                                          nx_, hw_);
        const double h_yp = detail::at_yp(h.data(), h_halos_[2].data(), ix, iy,
                                          nx_, ny_, hw_);
        const double h_ym = detail::at_ym(h.data(), h_halos_[3].data(), ix, iy,
                                          nx_, hw_);

        const double p_xp = detail::at_xp(p_.data(), p_halos_[0].data(), ix, iy,
                                          nx_, 1);
        const double p_xm = detail::at_xm(p_.data(), p_halos_[1].data(), ix, iy,
                                          nx_, 1);
        const double p_yp = detail::at_yp(p_.data(), p_halos_[2].data(), ix, iy,
                                          nx_, ny_, 1);
        const double p_ym = detail::at_ym(p_.data(), p_halos_[3].data(), ix, iy,
                                          nx_, 1);

        const double Mc = mobility(hc);
        const double Mxp = face_mobility(kind, Mc, mobility(h_xp));
        const double Mxm = face_mobility(kind, mobility(h_xm), Mc);
        const double Myp = face_mobility(kind, Mc, mobility(h_yp));
        const double Mym = face_mobility(kind, mobility(h_ym), Mc);

        const double Fxp = Mxp * (p_xp - pc) / dx_;
        const double Fxm = Mxm * (pc - p_xm) / dx_;
        const double Fyp = Myp * (p_yp - pc) / dy_;
        const double Fym = Mym * (pc - p_ym) / dy_;

        dhdt_out[c] = (Fxp - Fxm) / dx_ + (Fyp - Fym) / dy_;
      }
    }
  }

  /// One explicit-Euler step of size `dt`.
  template <class Mobility>
  void step(std::vector<double> &h, double dt, double gamma,
            const ThinFilmPointwise &pw, const Mobility &mobility,
            FaceMobility kind) {
    compute_rhs(h, gamma, pw, mobility, kind, dhdt_);
    for (std::size_t c = 0; c < h.size(); ++c) h[c] += dt * dhdt_[c];
  }

private:
  pfc::Domain domain_;
  pfc::decomposition::Decomposition decomp_;
  int rank_;
  int order_;
  int hw_;
  int nx_, ny_;
  double dx_, dy_;
  std::vector<double> p_, lap_h_, dhdt_;
  std::array<std::vector<double>, 6> h_halos_, p_halos_;
  pfc::comm::SparseExchange<pfc::HostSpace, double> exch_h_;
  pfc::comm::SparseExchange<pfc::HostSpace, double> exch_p_;
};

/// Same observables as `thin_film::sample_film`, for a raw FD state vector.
[[nodiscard]] inline FilmSample
sample_film_fd(const std::vector<double> &h, const pfc::Domain &domain, double h0,
               double rupture_frac, MPI_Comm comm) {
  double lo = std::numeric_limits<double>::infinity(), hi = -lo;
  double local_sum = 0.0, local_holes = 0.0, local_cells = 0.0;
  for (double v : h) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
    local_sum += v;
    if (v < 0.5 * h0) local_holes += 1.0;
    local_cells += 1.0;
  }
  double g_lo = 0, g_hi = 0, g[3]{}, l[3]{local_sum, local_holes, local_cells};
  MPI_Allreduce(&lo, &g_lo, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&hi, &g_hi, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(l, g, 3, MPI_DOUBLE, MPI_SUM, comm);

  const auto dx = pfc::domain::get_spacing(domain);
  const double cell = dx[0] * dx[1] * dx[2];
  FilmSample s;
  s.min_h = g_lo;
  s.max_h = g_hi;
  s.mean_h = (g[2] > 0.0) ? g[0] / g[2] : 0.0;
  s.volume = g[0] * cell;
  s.hole_area_fraction = (g[2] > 0.0) ? g[1] / g[2] : 0.0;
  s.ruptured = g_lo < rupture_frac * h0;
  return s;
}

/**
 * @brief Number of 4-connected dry regions (`h < threshold`) on a global,
 *        rank-0-gathered copy of the field.
 *
 * A diagnostic, not a distributed algorithm: flood-fills the whole global
 * grid on rank 0 with a simple union-find, treating the domain as doubly
 * periodic (wraps at both edges) since that is the actual boundary
 * condition. `global_xy` must already be assembled in `[gx + gy*nx_glob]`
 * order, e.g. via `pfc::apps::gather_global_xy_rank0`. Cheap enough for the
 * diagnostic cadence this application uses it at (every `saveat`, not every
 * step) on grids up to a few hundred thousand cells; call only on rank 0.
 */
[[nodiscard]] inline int count_dry_regions_rank0(const std::vector<double> &global_xy,
                                                 int nx_glob, int ny_glob,
                                                 double threshold) {
  const std::size_t n =
      static_cast<std::size_t>(nx_glob) * static_cast<std::size_t>(ny_glob);
  std::vector<int> parent(n);
  for (std::size_t i = 0; i < n; ++i) parent[i] = static_cast<int>(i);
  std::vector<int> rank_of(n, 0);

  auto find = [&](int x) {
    while (parent[static_cast<std::size_t>(x)] != x) {
      const int p = parent[static_cast<std::size_t>(x)];
      parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(p)];
      x = p;
    }
    return x;
  };
  auto unite = [&](int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    auto &ra = rank_of[static_cast<std::size_t>(a)];
    auto &rb = rank_of[static_cast<std::size_t>(b)];
    if (ra < rb) std::swap(a, b);
    parent[static_cast<std::size_t>(b)] = a;
    if (ra == rb) ++ra;
  };

  auto is_dry = [&](int gx, int gy) {
    return global_xy[static_cast<std::size_t>(gx) +
                     static_cast<std::size_t>(gy) *
                         static_cast<std::size_t>(nx_glob)] < threshold;
  };

  for (int gy = 0; gy < ny_glob; ++gy) {
    for (int gx = 0; gx < nx_glob; ++gx) {
      if (!is_dry(gx, gy)) continue;
      const int c = gx + gy * nx_glob;
      const int xp = (gx + 1) % nx_glob;
      const int yp = (gy + 1) % ny_glob;
      if (is_dry(xp, gy)) unite(c, xp + gy * nx_glob);
      if (is_dry(gx, yp)) unite(c, gx + yp * nx_glob);
    }
  }

  int count = 0;
  for (int gy = 0; gy < ny_glob; ++gy) {
    for (int gx = 0; gx < nx_glob; ++gx) {
      if (!is_dry(gx, gy)) continue;
      if (find(gx + gy * nx_glob) == gx + gy * nx_glob) ++count;
    }
  }
  return count;
}

} // namespace thin_film
