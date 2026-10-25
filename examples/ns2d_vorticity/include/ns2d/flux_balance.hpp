// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file flux_balance.hpp
 * @brief Closed-box flux of the Fourier velocity (#226).
 *
 * The integral is the trigonometric interpolant of the r2c coefficients:
 * each stored mode plus the unstored negative-kx conjugate. It is the
 * represented spectral field, not a finite-volume face flux. A grid
 * trapezoid is provided only as a low-order sample of the same nodal
 * values.
 */

#include <ns2d/spectral.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ns2d {

struct AxisBox {
  double x0{0.0};
  double x1{0.0};
  double y0{0.0};
  double y1{0.0};
};

struct FaceFlux {
  double net{0.0};
  double imag{0.0};
  double abs_faces{0.0};
  double relative{0.0};
};

namespace detail {

[[nodiscard]] inline SpectralPlane::Complex integrate_exp(double k, double a,
                                                          double b) {
  if (k == 0.0) return SpectralPlane::Complex(b - a, 0.0);
  const SpectralPlane::Complex ik(0.0, k);
  return (std::exp(ik * b) - std::exp(ik * a)) / ik;
}

struct FaceAccum {
  SpectralPlane::Complex right{};
  SpectralPlane::Complex left{};
  SpectralPlane::Complex top{};
  SpectralPlane::Complex bottom{};
};

inline void add_mode(FaceAccum &acc, SpectralPlane::Complex u_hat,
                     SpectralPlane::Complex v_hat, double kx, double ky,
                     const AxisBox &box) {
  const auto iy = integrate_exp(ky, box.y0, box.y1);
  const auto ix = integrate_exp(kx, box.x0, box.x1);
  const auto ex0 = std::exp(SpectralPlane::Complex(0.0, kx * box.x0));
  const auto ex1 = std::exp(SpectralPlane::Complex(0.0, kx * box.x1));
  const auto ey0 = std::exp(SpectralPlane::Complex(0.0, ky * box.y0));
  const auto ey1 = std::exp(SpectralPlane::Complex(0.0, ky * box.y1));
  acc.right += u_hat * ex1 * iy;
  acc.left += u_hat * ex0 * iy;
  acc.top += v_hat * ey1 * ix;
  acc.bottom += v_hat * ey0 * ix;
}

} // namespace detail

/// Net outward flux of the r2c interpolant through an axis-aligned box.
[[nodiscard]] inline FaceFlux
face_flux(const SpectralPlane &plane,
          const std::vector<SpectralPlane::Complex> &u_hat,
          const std::vector<SpectralPlane::Complex> &v_hat, AxisBox box) {
  if (u_hat.size() != plane.out_n() || v_hat.size() != plane.out_n()) {
    throw std::invalid_argument("ns2d: velocity hat size does not match the plane");
  }
  if (!(box.x1 >= box.x0 && box.y1 >= box.y0)) {
    throw std::invalid_argument("ns2d: control-volume box is inverted");
  }
  const auto gs = plane.gsize();
  const double ntot = static_cast<double>(gs[0]) * static_cast<double>(gs[1]) *
                      static_cast<double>(gs[2]);
  const double inv = 1.0 / ntot;
  detail::FaceAccum acc;
  pfc::fft::kspace::for_each_kpoint(
      plane.outbox(), gs, plane.spacing(),
      [&](std::size_t idx, double kx, double ky, double, int i, int, int) {
        const auto u = u_hat[idx] * inv;
        const auto v = v_hat[idx] * inv;
        detail::add_mode(acc, u, v, kx, ky, box);
        const bool partner_stored = (i == 0 || i == gs[0] / 2);
        if (!partner_stored) {
          detail::add_mode(acc, std::conj(u), std::conj(v), -kx, -ky, box);
        }
      });
  const auto out = acc.right - acc.left + acc.top - acc.bottom;
  FaceFlux flux;
  flux.net = out.real();
  flux.imag = out.imag();
  const double right = acc.right.real();
  const double left = acc.left.real();
  const double top = acc.top.real();
  const double bottom = acc.bottom.real();
  flux.abs_faces =
      std::abs(right) + std::abs(left) + std::abs(top) + std::abs(bottom);
  flux.relative = (flux.abs_faces > 0.0) ? flux.net / flux.abs_faces : 0.0;
  return flux;
}

/// Largest modal amplitude of \f$i k_x \hat u + i k_y \hat v\f$.
///
/// Wavenumbers are the odd multipliers used by `spectral_div`. The zero
/// mode of that divergence is zero for every periodic field.
[[nodiscard]] inline double
modal_div_amplitude(const SpectralPlane &plane,
                    const std::vector<SpectralPlane::Complex> &u_hat,
                    const std::vector<SpectralPlane::Complex> &v_hat) {
  if (u_hat.size() != plane.out_n() || v_hat.size() != plane.out_n()) {
    throw std::invalid_argument("ns2d: velocity hat size does not match the plane");
  }
  const auto gs = plane.gsize();
  const double ntot = static_cast<double>(gs[0]) * static_cast<double>(gs[1]) *
                      static_cast<double>(gs[2]);
  const auto &kx = plane.kx_odd();
  const auto &ky = plane.ky_odd();
  double max_amp = 0.0;
  for (std::size_t i = 0; i < u_hat.size(); ++i) {
    const auto div = u_hat[i] * SpectralPlane::Complex(0.0, kx[i]) +
                     v_hat[i] * SpectralPlane::Complex(0.0, ky[i]);
    max_amp = std::max(max_amp, std::abs(div) / ntot);
  }
  return max_amp;
}

/// Area integral of the spectral divergence. This is the zero mode, so it
/// is zero for every periodic Fourier field, divergence-free or not.
[[nodiscard]] inline double
domain_divergence(const SpectralPlane &plane,
                  const std::vector<SpectralPlane::Complex> &u_hat,
                  const std::vector<SpectralPlane::Complex> &v_hat) {
  if (u_hat.size() != plane.out_n() || v_hat.size() != plane.out_n()) {
    throw std::invalid_argument("ns2d: velocity hat size does not match the plane");
  }
  const auto gs = plane.gsize();
  const auto sp = plane.spacing();
  const double ntot = static_cast<double>(gs[0]) * static_cast<double>(gs[1]) *
                      static_cast<double>(gs[2]);
  const double area =
      (static_cast<double>(gs[0]) * sp[0]) * (static_cast<double>(gs[1]) * sp[1]);
  const auto &kx = plane.kx_odd();
  const auto &ky = plane.ky_odd();
  double value = 0.0;
  pfc::fft::kspace::for_each_kpoint(
      plane.outbox(), gs, sp,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        if (i != 0 || j != 0 || k != 0) return;
        const auto div = u_hat[idx] * SpectralPlane::Complex(0.0, kx[idx]) +
                         v_hat[idx] * SpectralPlane::Complex(0.0, ky[idx]);
        value = div.real() / ntot * area;
      });
  return value;
}

/// Composite trapezoid on grid nodes. Empty when a corner is off-node.
/// Index `N` is the periodic image of index 0.
[[nodiscard]] inline std::optional<double>
trapezoid_flux(const SpectralPlane &plane, const std::vector<double> &u,
               const std::vector<double> &v, AxisBox box) {
  const auto gs = plane.gsize();
  const int n = gs[0];
  if (gs[1] != n) return std::nullopt;
  const auto sp = plane.spacing();
  const double dx = sp[0];
  const double dy = sp[1];
  auto node = [](double x, double h, int ngrid) -> int {
    const double q = x / h;
    const double nearest = std::round(q);
    if (std::abs(q - nearest) > 1.0e-8) return -1;
    const int i = static_cast<int>(nearest);
    if (i < 0 || i > ngrid) return -1;
    return i;
  };
  const int i0 = node(box.x0, dx, n);
  const int i1 = node(box.x1, dx, n);
  const int j0 = node(box.y0, dy, n);
  const int j1 = node(box.y1, dy, n);
  if (i0 < 0 || i1 < 0 || j0 < 0 || j1 < 0 || i1 < i0 || j1 < j0) {
    return std::nullopt;
  }
  auto sample = [&](const std::vector<double> &f, int i, int j) {
    if (i == n) i = 0;
    if (j == n) j = 0;
    return f[plane.real_idx(i, j, 0)];
  };
  auto trap = [&](const std::vector<double> &f, bool vertical, int fixed, int a,
                  int b, double h) {
    double sum = 0.5 * sample(f, vertical ? fixed : a, vertical ? a : fixed);
    sum += 0.5 * sample(f, vertical ? fixed : b, vertical ? b : fixed);
    for (int t = a + 1; t < b; ++t) {
      sum += sample(f, vertical ? fixed : t, vertical ? t : fixed);
    }
    return sum * h;
  };
  const double right = trap(u, true, i1, j0, j1, dy);
  const double left = trap(u, true, i0, j0, j1, dy);
  const double top = trap(v, false, j1, i0, i1, dx);
  const double bottom = trap(v, false, j0, i0, i1, dx);
  return (right - left) + (top - bottom);
}

} // namespace ns2d
