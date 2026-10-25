// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file finite_strain_grid.hpp
 * @brief Small 2-D periodic finite-strain homogenizer (research #484).
 *
 * Cell-centred fluctuation displacement, central differences, plane strain.
 * Macroscopic \f$F_{11}\f$ is prescribed; \f$F_{22}\f$ is relaxed so
 * \f$\langle P_{22}\rangle=0\f$. Uses the shipped neo-Hookean / StVK
 * `first_pk` from finite_strain_forward.hpp. Not small-strain \f$C_H\f$.
 *
 * Dense Newton is intended for 8²–16² qualification cells.
 */

#include "finite_strain_forward.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pfc::apps::inverse::fs {

struct GridHomogenization {
  double F11{1.0};
  double F22{1.0};
  double P11{0.0};
  double P22{0.0};
  double residual{0.0};
  int newton_iters{0};
  bool converged{false};
  bool stable{false};
};

inline int wrap(int i, int n) {
  int r = i % n;
  return r < 0 ? r + n : r;
}

inline bool solve_dense(std::vector<double> &A, std::vector<double> &b, int n) {
  const double eps = 1e-14;
  for (int k = 0; k < n; ++k) {
    int piv = k;
    double best = std::abs(A[k * n + k]);
    for (int i = k + 1; i < n; ++i) {
      const double v = std::abs(A[i * n + k]);
      if (v > best) {
        best = v;
        piv = i;
      }
    }
    if (best < eps) return false;
    if (piv != k) {
      for (int j = k; j < n; ++j) std::swap(A[k * n + j], A[piv * n + j]);
      std::swap(b[k], b[piv]);
    }
    const double akk = A[k * n + k];
    for (int i = k + 1; i < n; ++i) {
      const double f = A[i * n + k] / akk;
      b[i] -= f * b[k];
      for (int j = k; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    double s = b[i];
    for (int j = i + 1; j < n; ++j) s -= A[i * n + j] * b[j];
    const double aii = A[i * n + i];
    if (std::abs(aii) < eps) return false;
    b[i] = s / aii;
  }
  return true;
}

struct PixelCell {
  int nx{8};
  int ny{8};
  Model model{Model::NeoHookean};
  std::vector<Lame> lame;
  std::vector<double> ux;
  std::vector<double> uy;
};

inline int idx(const PixelCell &c, int i, int j) {
  return wrap(j, c.ny) * c.nx + wrap(i, c.nx);
}

inline void init_cell(PixelCell &c, int nx, int ny, Model model) {
  c.nx = nx;
  c.ny = ny;
  c.model = model;
  const int N = nx * ny;
  c.lame.assign(static_cast<std::size_t>(N), Lame{});
  c.ux.assign(static_cast<std::size_t>(N), 0.0);
  c.uy.assign(static_cast<std::size_t>(N), 0.0);
}

inline Mat2 F_at(const PixelCell &c, int i, int j, double F11, double F22) {
  const int ip = wrap(i + 1, c.nx), im = wrap(i - 1, c.nx);
  const int jp = wrap(j + 1, c.ny), jm = wrap(j - 1, c.ny);
  const double dx = 1.0 / static_cast<double>(c.nx);
  const double dy = 1.0 / static_cast<double>(c.ny);
  const auto at = [&](int ii, int jj) { return idx(c, ii, jj); };
  const double dux_dx = (c.ux[at(ip, j)] - c.ux[at(im, j)]) / (2.0 * dx);
  const double dux_dy = (c.ux[at(i, jp)] - c.ux[at(i, jm)]) / (2.0 * dy);
  const double duy_dx = (c.uy[at(ip, j)] - c.uy[at(im, j)]) / (2.0 * dx);
  const double duy_dy = (c.uy[at(i, jp)] - c.uy[at(i, jm)]) / (2.0 * dy);
  return Mat2{F11 + dux_dx, dux_dy, duy_dx, F22 + duy_dy};
}

inline void residual(const PixelCell &c, double F11, double F22,
                     std::vector<double> &rx, std::vector<double> &ry, double &P11,
                     double &P22, bool &valid) {
  const int N = c.nx * c.ny;
  std::vector<double> P11c(N), P12c(N), P21c(N), P22c(N);
  P11 = 0.0;
  P22 = 0.0;
  valid = true;
  for (int j = 0; j < c.ny; ++j) {
    for (int i = 0; i < c.nx; ++i) {
      const int p = idx(c, i, j);
      FirstPK pk{};
      try {
        pk = first_pk(c.model, c.lame[static_cast<std::size_t>(p)],
                      F_at(c, i, j, F11, F22));
      } catch (const std::runtime_error &) {
        valid = false;
        pk.P = Mat2{1e3, 0.0, 0.0, 1e3};
        pk.J = 0.0;
      }
      if (!(pk.J > 1e-12)) valid = false;
      P11c[p] = pk.P.a11;
      P12c[p] = pk.P.a12;
      P21c[p] = pk.P.a21;
      P22c[p] = pk.P.a22;
      P11 += pk.P.a11;
      P22 += pk.P.a22;
    }
  }
  P11 /= static_cast<double>(N);
  P22 /= static_cast<double>(N);
  const double dx = 1.0 / static_cast<double>(c.nx);
  const double dy = 1.0 / static_cast<double>(c.ny);
  rx.assign(N, 0.0);
  ry.assign(N, 0.0);
  for (int j = 0; j < c.ny; ++j) {
    for (int i = 0; i < c.nx; ++i) {
      const int p = idx(c, i, j);
      const int ip = idx(c, i + 1, j), im = idx(c, i - 1, j);
      const int jp = idx(c, i, j + 1), jm = idx(c, i, j - 1);
      rx[p] =
          (P11c[ip] - P11c[im]) / (2.0 * dx) + (P12c[jp] - P12c[jm]) / (2.0 * dy);
      ry[p] =
          (P21c[ip] - P21c[im]) / (2.0 * dx) + (P22c[jp] - P22c[jm]) / (2.0 * dy);
    }
  }
}

// A centered periodic derivative annihilates constants on each parity class
// when the corresponding extent is even. Fix one displacement value in each
// class, not only the global translation. Subtracting these constants changes
// neither F_at nor the constitutive response.
inline int gauge_anchor(const PixelCell &c, int p) {
  const int gx = c.nx % 2 == 0 ? 2 : 1;
  const int gy = c.ny % 2 == 0 ? 2 : 1;
  return (p % c.nx) % gx + ((p / c.nx) % gy) * c.nx;
}

inline int n_unknowns(const PixelCell &c) {
  const int anchors = (c.nx % 2 == 0 ? 2 : 1) * (c.ny % 2 == 0 ? 2 : 1);
  return 2 * (c.nx * c.ny - anchors) + 1;
}

inline void pack_unknowns(const PixelCell &c, double F22, std::vector<double> &x) {
  const int N = c.nx * c.ny;
  x.resize(static_cast<std::size_t>(n_unknowns(c)));
  int k = 0;
  for (const auto *u : {&c.ux, &c.uy})
    for (int p = 0; p < N; ++p) {
      const int anchor = gauge_anchor(c, p);
      if (p != anchor) x[k++] = (*u)[p] - (*u)[anchor];
    }
  x[k] = F22;
}

inline void unpack_unknowns(PixelCell &c, const std::vector<double> &x,
                            double &F22) {
  const int N = c.nx * c.ny;
  int k = 0;
  for (auto *u : {&c.ux, &c.uy})
    for (int p = 0; p < N; ++p) (*u)[p] = p == gauge_anchor(c, p) ? 0.0 : x[k++];
  F22 = x[k];
}

inline void pack_residual(const PixelCell &c, double F11, double F22,
                          std::vector<double> &r, double &P11, double &P22,
                          bool &valid) {
  const int N = c.nx * c.ny;
  std::vector<double> rx, ry;
  residual(c, F11, F22, rx, ry, P11, P22, valid);
  r.resize(static_cast<std::size_t>(n_unknowns(c)));
  int k = 0;
  for (const auto *component : {&rx, &ry})
    for (int p = 0; p < N; ++p)
      if (p != gauge_anchor(c, p)) r[k++] = (*component)[p];
  r[k] = P22;
  if (!valid) {
    for (double &v : r) v = 1e3;
  }
}

[[nodiscard]] inline GridHomogenization homogenize_periodic_2d(PixelCell &c,
                                                               double F11,
                                                               int max_iter = 25,
                                                               double tol = 1e-8) {
  if (c.nx < 2 || c.ny < 2) {
    throw std::invalid_argument("homogenize_periodic_2d: grid too small");
  }
  GridHomogenization out;
  out.F11 = F11;
  double F22 = 1.0 / std::sqrt(F11);
  std::vector<double> x;
  pack_unknowns(c, F22, x);
  const int n = static_cast<int>(x.size());
  const double fd = 1e-7;
  std::vector<double> r, rp, col(static_cast<std::size_t>(n));
  double P11 = 0.0, P22 = 0.0;
  bool valid = true;
  for (int it = 0; it < max_iter; ++it) {
    unpack_unknowns(c, x, F22);
    if (!(F22 > 1e-6)) F22 = 1e-6;
    pack_residual(c, F11, F22, r, P11, P22, valid);
    double nrm = 0.0;
    for (double v : r) nrm = std::max(nrm, std::abs(v));
    out.newton_iters = it + 1;
    out.residual = nrm;
    out.P11 = P11;
    out.P22 = P22;
    out.F22 = F22;
    if (valid && nrm < tol) {
      out.converged = true;
      out.stable = F22 > 1e-6 && std::isfinite(P11);
      return out;
    }
    std::vector<double> J(static_cast<std::size_t>(n * n), 0.0);
    for (int j = 0; j < n; ++j) {
      const double xj = x[static_cast<std::size_t>(j)];
      x[static_cast<std::size_t>(j)] = xj + fd;
      unpack_unknowns(c, x, F22);
      bool vfd = true;
      pack_residual(c, F11, F22, rp, P11, P22, vfd);
      x[static_cast<std::size_t>(j)] = xj;
      for (int i = 0; i < n; ++i) {
        J[i * n + j] =
            (rp[static_cast<std::size_t>(i)] - r[static_cast<std::size_t>(i)]) / fd;
      }
    }
    std::vector<double> rhs = r;
    for (double &v : rhs) v = -v;
    if (!solve_dense(J, rhs, n)) break;
    std::vector<double> xtry = x;
    double alpha = 1.0;
    bool accepted = false;
    for (int ls = 0; ls < 8; ++ls) {
      for (int i = 0; i < n; ++i)
        xtry[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] +
                                            alpha * rhs[static_cast<std::size_t>(i)];
      unpack_unknowns(c, xtry, F22);
      bool vls = true;
      pack_residual(c, F11, F22, rp, P11, P22, vls);
      double n2 = 0.0;
      for (double v : rp) n2 = std::max(n2, std::abs(v));
      if (vls && n2 < nrm * (1.0 - 1e-4 * alpha)) {
        x = xtry;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
  }
  unpack_unknowns(c, x, F22);
  pack_residual(c, F11, F22, r, P11, P22, valid);
  out.F22 = F22;
  out.P11 = P11;
  out.P22 = P22;
  out.residual = 0.0;
  for (double v : r) out.residual = std::max(out.residual, std::abs(v));
  out.converged = out.residual < tol;
  out.stable = F22 > 1e-6 && std::isfinite(P11);
  return out;
}

inline bool in_rotated_square_local(double px, double py, double cx, double cy,
                                    double half, double angle) {
  auto wrapd = [](double a, double b) {
    double d = a - b;
    d -= std::round(d);
    return d;
  };
  const double dx = wrapd(px, cx);
  const double dy = wrapd(py, cy);
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const double lx = c * dx + s * dy;
  const double ly = -s * dx + c * dy;
  return std::abs(lx) <= half && std::abs(ly) <= half;
}

/// Occupancy in [0,1]; 1 is solid. Does not use OpenPFC Field.
inline void fill_rotating_squares_vec(std::vector<double> &h, int nx, int ny,
                                      double half, double angle) {
  h.assign(static_cast<std::size_t>(nx * ny), 0.0);
  const double cxs[4] = {0.25, 0.75, 0.75, 0.25};
  const double cys[4] = {0.25, 0.25, 0.75, 0.75};
  const double angs[4] = {angle, -angle, angle, -angle};
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double x = (static_cast<double>(i) + 0.5) / static_cast<double>(nx);
      const double y = (static_cast<double>(j) + 0.5) / static_cast<double>(ny);
      bool solid = false;
      for (int s = 0; s < 4; ++s) {
        if (in_rotated_square_local(x, y, cxs[s], cys[s], half, angs[s])) {
          solid = true;
          break;
        }
      }
      h[static_cast<std::size_t>(j * nx + i)] = solid ? 1.0 : 0.0;
    }
  }
}

inline void assign_lame_two_phase(PixelCell &c, const std::vector<double> &h,
                                  const Lame &solid, const Lame &voided) {
  const int N = c.nx * c.ny;
  c.lame.resize(static_cast<std::size_t>(N));
  for (int p = 0; p < N; ++p) {
    const double a = std::clamp(h[static_cast<std::size_t>(p)], 0.0, 1.0);
    c.lame[static_cast<std::size_t>(p)] =
        Lame{a * solid.lambda + (1.0 - a) * voided.lambda,
             a * solid.mu + (1.0 - a) * voided.mu};
  }
}

inline void assign_lame_three_phase(PixelCell &c, const std::vector<int> &phase,
                                    const Lame &stiff, const Lame &compliant,
                                    const Lame &voided) {
  const int N = c.nx * c.ny;
  c.lame.resize(static_cast<std::size_t>(N));
  for (int p = 0; p < N; ++p) {
    const int ph = phase[static_cast<std::size_t>(p)];
    if (ph == 2)
      c.lame[static_cast<std::size_t>(p)] = stiff;
    else if (ph == 1)
      c.lame[static_cast<std::size_t>(p)] = compliant;
    else
      c.lame[static_cast<std::size_t>(p)] = voided;
  }
}

/// Solid cells with a void neighbour become compliant (phase 1).
inline void paint_hinge_phase(const std::vector<double> &h, int nx, int ny,
                              std::vector<int> &phase) {
  phase.assign(static_cast<std::size_t>(nx * ny), 0);
  auto at = [&](int i, int j) { return wrap(j, ny) * nx + wrap(i, nx); };
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int p = at(i, j);
      if (h[static_cast<std::size_t>(p)] < 0.5) {
        phase[static_cast<std::size_t>(p)] = 0;
        continue;
      }
      bool hinge = false;
      const int nbs[4] = {at(i + 1, j), at(i - 1, j), at(i, j + 1), at(i, j - 1)};
      for (int q : nbs) {
        if (h[static_cast<std::size_t>(q)] < 0.5) hinge = true;
      }
      phase[static_cast<std::size_t>(p)] = hinge ? 1 : 2;
    }
  }
}

} // namespace pfc::apps::inverse::fs
