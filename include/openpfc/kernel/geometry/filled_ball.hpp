// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file filled_ball.hpp
 * @brief Filled spherical-volume reference domain and its scalar calculus.
 *
 * This is the volume `0 <= r <= R`, not a sphere-surface manifold. A surface
 * domain would be a different type. Coordinates are `(r, θ, φ)` with `θ` the
 * polar angle from `+z` and `φ` the azimuth. `r = 0` is the origin, not a
 * boundary. The only physical boundary is `r = R`.
 *
 * A mode is `coefficient * S_l^m(x, y, z) * (x² + y² + z²)^k`. `S_l^m` is
 * the regular solid harmonic frozen below for `0 <= l <= 3` and `|m| <= l`.
 * It is a polynomial, so the origin is never replaced by `1/(r+ε)`. The
 * complex direction is `x + iy` for `m > 0` and `x - iy` for `m < 0`. The
 * frozen polynomials, with `ζ = x ± iy` and `ρ = x² + y²`, are:
 *
 * - `l = 0`: `1`
 * - `l = 1`: `z`, `ζ`
 * - `l = 2`: `2z² - ρ`, `z ζ`, `ζ²`
 * - `l = 3`: `z(2z² - 3ρ)`, `ζ(4z² - ρ)`, `z ζ²`, `ζ³`
 *
 * A general spherical-harmonic transform and a production radial basis are
 * not selected. `(r²)^k` is the verification factor. For `k = 0`, `ΔS = 0`
 * and `x ∂x + y ∂y + z ∂z` equals `l S`, which is the axis-safe form of the
 * spherical eigenvalue `-l(l+1)`.
 *
 * `AffineMap` sends the embedding to an ellipsoid. The volume factor is
 * `|det A|`. The physical Laplacian is `G : Hess` with `G = A^{-1} A^{-T}`.
 * An identity Jacobian uses the trace and does not form `G`.
 */

#include <openpfc/kernel/geometry/affine_map.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace pfc::geometry {

enum class BallBoundary { RadialWall };

struct FilledBall {
  double radius = 1;

  double volume() const noexcept {
    return (4.0 / 3.0) * std::numbers::pi * radius * radius * radius;
  }
};

struct BallHarmonic {
  int degree = 0;
  int order = 0;
  int radial = 0;
  std::complex<double> coefficient{};
};

struct BallSeries {
  FilledBall domain{};
  std::vector<BallHarmonic> modes{};
};

[[nodiscard]] inline bool ball_has_boundary(const FilledBall &domain,
                                            BallBoundary piece) noexcept {
  (void)domain;
  return piece == BallBoundary::RadialWall;
}

[[nodiscard]] inline FilledBall filled_ball(double radius) {
  if (!std::isfinite(radius) || !(radius > 0.0)) {
    throw std::invalid_argument("filled ball needs a positive radius");
  }
  return FilledBall{radius};
}

[[nodiscard]] inline bool ball_contains(const FilledBall &domain, double r,
                                        double theta, double phi) noexcept {
  return std::isfinite(r) && std::isfinite(theta) && std::isfinite(phi) &&
         r >= 0.0 && r <= domain.radius && theta >= 0.0 && theta <= std::numbers::pi;
}

[[nodiscard]] inline Vec3 ball_embedding(double r, double theta, double phi) {
  if (!std::isfinite(r) || !std::isfinite(theta) || !std::isfinite(phi) || r < 0.0) {
    throw std::invalid_argument("ball radius must be finite and non-negative");
  }
  const double radial = r * std::sin(theta);
  return {radial * std::cos(phi), radial * std::sin(phi), r * std::cos(theta)};
}

[[nodiscard]] inline bool ball_harmonic_is_admissible(const BallHarmonic &mode) {
  // Validate the small degree first: negating an unchecked INT_MIN order is UB.
  if (mode.degree < 0 || mode.degree > 3) return false;
  if (mode.order < -mode.degree || mode.order > mode.degree) return false;
  if (mode.radial < 0 || mode.radial > 12) return false;
  if (!std::isfinite(mode.coefficient.real())) return false;
  if (!std::isfinite(mode.coefficient.imag())) return false;
  return true;
}

inline void add_ball_harmonic(BallSeries &series, BallHarmonic mode) {
  if (!ball_harmonic_is_admissible(mode)) {
    throw std::invalid_argument(
        "ball harmonic degree or coefficient is not in the frozen set");
  }
  series.modes.push_back(mode);
}

namespace ball_detail {

struct Jet {
  std::complex<double> v{};
  std::complex<double> x{};
  std::complex<double> y{};
  std::complex<double> z{};
  std::complex<double> xx{};
  std::complex<double> yy{};
  std::complex<double> zz{};
  std::complex<double> xy{};
  std::complex<double> xz{};
  std::complex<double> yz{};
};

inline Jet operator+(Jet a, Jet b) {
  a.v += b.v;
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.xx += b.xx;
  a.yy += b.yy;
  a.zz += b.zz;
  a.xy += b.xy;
  a.xz += b.xz;
  a.yz += b.yz;
  return a;
}

inline Jet operator-(Jet a, Jet b) {
  a.v -= b.v;
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.xx -= b.xx;
  a.yy -= b.yy;
  a.zz -= b.zz;
  a.xy -= b.xy;
  a.xz -= b.xz;
  a.yz -= b.yz;
  return a;
}

inline Jet operator*(Jet a, Jet b) {
  Jet c;
  c.v = a.v * b.v;
  c.x = a.x * b.v + a.v * b.x;
  c.y = a.y * b.v + a.v * b.y;
  c.z = a.z * b.v + a.v * b.z;
  c.xx = a.xx * b.v + 2.0 * a.x * b.x + a.v * b.xx;
  c.yy = a.yy * b.v + 2.0 * a.y * b.y + a.v * b.yy;
  c.zz = a.zz * b.v + 2.0 * a.z * b.z + a.v * b.zz;
  c.xy = a.xy * b.v + a.x * b.y + a.y * b.x + a.v * b.xy;
  c.xz = a.xz * b.v + a.x * b.z + a.z * b.x + a.v * b.xz;
  c.yz = a.yz * b.v + a.y * b.z + a.z * b.y + a.v * b.yz;
  return c;
}

inline Jet jet_one() {
  Jet jet;
  jet.v = 1;
  return jet;
}

inline Jet jet_scale(Jet jet, std::complex<double> scale) {
  jet.v *= scale;
  jet.x *= scale;
  jet.y *= scale;
  jet.z *= scale;
  jet.xx *= scale;
  jet.yy *= scale;
  jet.zz *= scale;
  jet.xy *= scale;
  jet.xz *= scale;
  jet.yz *= scale;
  return jet;
}

inline Jet jet_pow(Jet base, int exponent) {
  Jet result = jet_one();
  for (int i = 0; i < exponent; ++i) result = result * base;
  return result;
}

inline Jet axis_x(double x) {
  Jet jet;
  jet.v = x;
  jet.x = 1;
  return jet;
}

inline Jet axis_y(double y) {
  Jet jet;
  jet.v = y;
  jet.y = 1;
  return jet;
}

inline Jet axis_z(double z) {
  Jet jet;
  jet.v = z;
  jet.z = 1;
  return jet;
}

inline Jet zeta_jet(int order, double x, double y) {
  const auto sign =
      order >= 0 ? std::complex<double>(0.0, 1.0) : std::complex<double>(0.0, -1.0);
  return axis_x(x) + jet_scale(axis_y(y), sign);
}

inline Jet solid_jet(int degree, int order, double x, double y, double z) {
  // A public BallSeries can also be modified after add_ball_harmonic.
  if (degree < 0 || degree > 3 || order < -degree || order > degree)
    throw std::invalid_argument("ball solid harmonic degree/order is invalid");
  const int abs_order = order < 0 ? -order : order;
  const Jet rho = axis_x(x) * axis_x(x) + axis_y(y) * axis_y(y);
  const Jet vertical = axis_z(z);
  if (degree == 0) return jet_one();
  if (degree == 1 && abs_order == 0) return vertical;
  const Jet zeta = zeta_jet(order, x, y);
  if (degree == 1) return zeta;
  if (degree == 2 && abs_order == 0)
    return jet_scale(vertical * vertical, 2.0) - rho;
  if (degree == 2 && abs_order == 1) return vertical * zeta;
  if (degree == 2) return zeta * zeta;
  if (degree == 3 && abs_order == 0) {
    return vertical * (jet_scale(vertical * vertical, 2.0) - jet_scale(rho, 3.0));
  }
  if (degree == 3 && abs_order == 1) {
    return zeta * (jet_scale(vertical * vertical, 4.0) - rho);
  }
  if (degree == 3 && abs_order == 2) return vertical * zeta * zeta;
  if (degree == 3) return zeta * zeta * zeta;
  throw std::invalid_argument("ball solid harmonic degree is not in the frozen set");
}

inline Jet harmonic_jet(const BallHarmonic &mode, double x, double y, double z) {
  if (!ball_harmonic_is_admissible(mode))
    throw std::invalid_argument("ball harmonic was mutated to an invalid mode");
  const Jet radius2 =
      axis_x(x) * axis_x(x) + axis_y(y) * axis_y(y) + axis_z(z) * axis_z(z);
  return jet_scale(solid_jet(mode.degree, mode.order, x, y, z) *
                       jet_pow(radius2, mode.radial),
                   mode.coefficient);
}

inline void require_radius(const FilledBall &domain) {
  if (!std::isfinite(domain.radius) || !(domain.radius > 0.0)) {
    throw std::invalid_argument("filled ball needs a positive radius");
  }
}

inline Jet series_jet(const BallSeries &series, double r, double theta, double phi) {
  require_radius(series.domain);
  const Vec3 point = ball_embedding(r, theta, phi);
  Jet sum;
  for (const BallHarmonic &mode : series.modes) {
    sum = sum + harmonic_jet(mode, point.x, point.y, point.z);
  }
  return sum;
}

inline std::complex<double> real_power(double base, int exponent) {
  std::complex<double> value = 1;
  for (int i = 0; i < exponent; ++i) value *= base;
  return value;
}

} // namespace ball_detail

[[nodiscard]] inline Vec3 ball_physical_point(double r, double theta, double phi,
                                              const AffineMap &map) {
  return map.to_physical(ball_embedding(r, theta, phi));
}

[[nodiscard]] inline double ball_physical_volume(const FilledBall &domain,
                                                 const AffineMap &map) {
  ball_detail::require_radius(domain);
  return physical_volume(map, domain.volume());
}

[[nodiscard]] inline std::complex<double>
evaluate_ball(const BallSeries &series, double r, double theta, double phi) {
  return ball_detail::series_jet(series, r, theta, phi).v;
}

[[nodiscard]] inline std::array<std::complex<double>, 3>
ball_gradient(const BallSeries &series, double r, double theta, double phi) {
  const auto jet = ball_detail::series_jet(series, r, theta, phi);
  return {jet.x, jet.y, jet.z};
}

/// `x ∂x + y ∂y + z ∂z`. Defined at the origin.
[[nodiscard]] inline std::complex<double>
ball_euler(const BallSeries &series, double r, double theta, double phi) {
  const Vec3 point = ball_embedding(r, theta, phi);
  const auto jet = ball_detail::series_jet(series, r, theta, phi);
  return point.x * jet.x + point.y * jet.y + point.z * jet.z;
}

/// Reference sphere r=R: local outward normal derivative d(u)/dr.
/// The reference radial normal is not the physical normal on a mapped ellipsoid.
[[nodiscard]] inline std::complex<double>
ball_wall_normal_derivative(const BallSeries &series, double theta, double phi) {
  ball_detail::require_radius(series.domain);
  const double radius = series.domain.radius;
  return ball_euler(series, radius, theta, phi) / radius;
}

[[nodiscard]] inline std::complex<double>
ball_laplacian(const BallSeries &series, double r, double theta, double phi) {
  const auto jet = ball_detail::series_jet(series, r, theta, phi);
  return jet.xx + jet.yy + jet.zz;
}

[[nodiscard]] inline std::complex<double>
ball_physical_laplacian(const BallSeries &series, const AffineMap &map, double r,
                        double theta, double phi) {
  const auto jet = ball_detail::series_jet(series, r, theta, phi);
  if (map.uses_cartesian_laplacian()) return jet.xx + jet.yy + jet.zz;
  double metric[9];
  map.inverse_metric(metric);
  const auto g = [&](int row, int col) { return metric[col * 3 + row]; };
  return g(0, 0) * jet.xx + g(1, 1) * jet.yy + g(2, 2) * jet.zz +
         (g(0, 1) + g(1, 0)) * jet.xy + (g(0, 2) + g(2, 0)) * jet.xz +
         (g(1, 2) + g(2, 1)) * jet.yz;
}

/// Integral with the volume weight. Only `l = 0` contributes.
[[nodiscard]] inline std::complex<double> integrate_ball(const BallSeries &series) {
  ball_detail::require_radius(series.domain);
  std::complex<double> sum;
  const double radius = series.domain.radius;
  for (const BallHarmonic &mode : series.modes) {
    if (mode.degree != 0) continue;
    const int power = 2 * mode.radial + 3;
    sum += mode.coefficient * 4.0 * std::numbers::pi *
           ball_detail::real_power(radius, power) / static_cast<double>(power);
  }
  return sum;
}

/// Outward flux of `grad u` through `r = R`. The origin is omitted.
[[nodiscard]] inline std::complex<double>
ball_boundary_flux(const BallSeries &series) {
  ball_detail::require_radius(series.domain);
  std::complex<double> flux;
  const double radius = series.domain.radius;
  for (const BallHarmonic &mode : series.modes) {
    if (mode.degree != 0 || mode.radial == 0) continue;
    const int radial = mode.radial;
    flux += mode.coefficient * (2.0 * static_cast<double>(radial)) * 4.0 *
            std::numbers::pi * ball_detail::real_power(radius, 2 * radial + 1);
  }
  return flux;
}

} // namespace pfc::geometry
