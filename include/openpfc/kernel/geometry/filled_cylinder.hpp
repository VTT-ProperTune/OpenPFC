// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file filled_cylinder.hpp
 * @brief Filled cylindrical reference domain and its scalar calculus.
 *
 * Coordinates are `(r, θ, z)` with `0 <= r <= R` and `θ` `2π`-periodic.
 * `r = 0` is a coordinate axis, not a boundary. The radial wall is `r = R`.
 * Caps `z = z_lower` and `z = z_upper` are physical boundaries only when the
 * axial policy is finite. A periodic axis has no caps.
 *
 * A scalar mode is `coefficient * (x ± iy)^{|m|} * (x² + y²)^k * Z_n(z)`
 * in the Cartesian embedding `x = r cos θ`, `y = r sin θ`. That factor is
 * `r^{|m|} (r²)^k e^{imθ}` and is smooth at the axis, so the Laplacian never
 * replaces `1/r` by `1/(r+ε)`. `θ` is the Fourier index `m`. Periodic `z`
 * uses `e^{i 2π n (z-z_lower)/L}`. Finite `z` uses the monomial
 * `(z-z_lower)^n`, which is the verification space only: the production
 * axial basis (Fourier is already the periodic choice; finite production is
 * FD or Chebyshev) and the production radial basis (Chebyshev, Jacobi, or
 * ultraspherical) are not selected here.
 *
 * `AffineMap` sends the embedding to physical space. The physical Laplacian
 * is `G : Hess u` with `G = A^{-1} A^{-T}`. An identity Jacobian uses the
 * embedding trace and does not form `G`.
 */

#include <openpfc/kernel/geometry/affine_map.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace pfc::geometry {

enum class CylinderAxialPolicy { Periodic, Finite };

enum class CylinderBoundary { RadialWall, LowerCap, UpperCap };

struct FilledCylinder {
  double radius = 1;
  double z_lower = 0;
  double z_upper = 1;
  CylinderAxialPolicy axial = CylinderAxialPolicy::Periodic;

  double length() const noexcept { return z_upper - z_lower; }

  double volume() const noexcept {
    return std::numbers::pi * radius * radius * length();
  }
};

struct CylinderHarmonic {
  int angular = 0;
  int radial = 0;
  int axial = 0;
  std::complex<double> coefficient{};
};

struct CylinderSeries {
  FilledCylinder domain{};
  std::vector<CylinderHarmonic> modes{};
};

[[nodiscard]] inline bool cylinder_has_boundary(const FilledCylinder &domain,
                                                CylinderBoundary piece) noexcept {
  if (piece == CylinderBoundary::RadialWall) return true;
  return domain.axial == CylinderAxialPolicy::Finite;
}

[[nodiscard]] inline FilledCylinder filled_cylinder(double radius, double z_lower,
                                                    double z_upper,
                                                    CylinderAxialPolicy axial) {
  if (!std::isfinite(radius) || !std::isfinite(z_lower) || !std::isfinite(z_upper) ||
      !(radius > 0.0) || !(z_upper > z_lower)) {
    throw std::invalid_argument(
        "filled cylinder needs a positive radius and a positive axial length");
  }
  return FilledCylinder{radius, z_lower, z_upper, axial};
}

[[nodiscard]] inline bool cylinder_contains(const FilledCylinder &domain, double r,
                                            double theta, double z) noexcept {
  return std::isfinite(r) && std::isfinite(theta) && std::isfinite(z) && r >= 0.0 &&
         r <= domain.radius && z >= domain.z_lower && z <= domain.z_upper;
}

/// Embedding of a cylindrical point. The axis angle is unused at `r = 0`.
[[nodiscard]] inline Vec3 cylinder_embedding(double r, double theta, double z) {
  if (!std::isfinite(r) || !std::isfinite(theta) || !std::isfinite(z) || r < 0.0) {
    throw std::invalid_argument(
        "cylindrical radius must be finite and non-negative");
  }
  return {r * std::cos(theta), r * std::sin(theta), z};
}

[[nodiscard]] inline Vec3 cylinder_physical_point(double r, double theta, double z,
                                                  const AffineMap &map) {
  return map.to_physical(cylinder_embedding(r, theta, z));
}

[[nodiscard]] inline double cylinder_physical_volume(const FilledCylinder &domain,
                                                     const AffineMap &map) noexcept {
  return physical_volume(map, domain.volume());
}

inline void add_cylinder_harmonic(CylinderSeries &series, CylinderHarmonic mode) {
  if (mode.radial < 0 || mode.radial > 12 || mode.angular < -24 ||
      mode.angular > 24 || mode.axial < -24 || mode.axial > 24 ||
      !std::isfinite(mode.coefficient.real()) ||
      !std::isfinite(mode.coefficient.imag())) {
    throw std::invalid_argument(
        "cylinder harmonic degree or coefficient is not valid");
  }
  // A periodic z-Fourier series allows +/- axial indices; a real sine or
  // cosine needs the two conjugate coefficients at both signs of k_z.
  if (series.domain.axial == CylinderAxialPolicy::Finite && mode.axial < 0) {
    throw std::invalid_argument(
        "a finite cylinder uses a non-negative axial monomial");
  }
  series.modes.push_back(mode);
}

namespace cylinder_detail {

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

inline std::complex<double> real_power(double base, int exponent) {
  std::complex<double> value = 1;
  for (int i = 0; i < exponent; ++i) value *= base;
  return value;
}

inline Jet axial_jet(const FilledCylinder &domain, int axial, double z) {
  Jet jet = jet_one();
  if (domain.axial == CylinderAxialPolicy::Periodic) {
    const double kappa =
        2.0 * std::numbers::pi * static_cast<double>(axial) / domain.length();
    const auto Z = std::exp(std::complex<double>(0.0, kappa * (z - domain.z_lower)));
    jet.v = Z;
    jet.z = std::complex<double>(0.0, kappa) * Z;
    jet.zz = std::complex<double>(-kappa * kappa, 0.0) * Z;
    return jet;
  }
  const double t = z - domain.z_lower;
  jet.v = real_power(t, axial);
  if (axial >= 1) jet.z = static_cast<double>(axial) * real_power(t, axial - 1);
  if (axial >= 2) {
    jet.zz = static_cast<double>(axial * (axial - 1)) * real_power(t, axial - 2);
  }
  return jet;
}

inline Jet harmonic_jet(const FilledCylinder &domain, const CylinderHarmonic &mode,
                        double x, double y, double z) {
  const int order = mode.angular < 0 ? -mode.angular : mode.angular;
  Jet radial = jet_one();
  if (order > 0) {
    Jet zeta;
    zeta.v =
        mode.angular >= 0 ? std::complex<double>(x, y) : std::complex<double>(x, -y);
    zeta.x = 1;
    zeta.y = mode.angular >= 0 ? std::complex<double>(0.0, 1.0)
                               : std::complex<double>(0.0, -1.0);
    radial = jet_pow(zeta, order);
  }
  if (mode.radial > 0) {
    Jet rho;
    rho.v = x * x + y * y;
    rho.x = 2.0 * x;
    rho.y = 2.0 * y;
    rho.xx = 2;
    rho.yy = 2;
    radial = radial * jet_pow(rho, mode.radial);
  }
  return jet_scale(radial * axial_jet(domain, mode.axial, z), mode.coefficient);
}

inline Jet series_jet(const CylinderSeries &series, double r, double theta,
                      double z) {
  const Vec3 point = cylinder_embedding(r, theta, z);
  Jet sum;
  for (const CylinderHarmonic &mode : series.modes) {
    const Jet term = harmonic_jet(series.domain, mode, point.x, point.y, point.z);
    sum.v += term.v;
    sum.x += term.x;
    sum.y += term.y;
    sum.z += term.z;
    sum.xx += term.xx;
    sum.yy += term.yy;
    sum.zz += term.zz;
    sum.xy += term.xy;
    sum.xz += term.xz;
    sum.yz += term.yz;
  }
  return sum;
}

inline std::complex<double> integrate_axial(const FilledCylinder &domain,
                                            int axial) {
  const double length = domain.length();
  if (domain.axial == CylinderAxialPolicy::Periodic) {
    return axial == 0 ? std::complex<double>(length, 0.0) : std::complex<double>{};
  }
  return real_power(length, axial + 1) / static_cast<double>(axial + 1);
}

inline std::complex<double> disk_moment(double radius, int radial_power) {
  const int p = 2 * radial_power;
  return 2.0 * std::numbers::pi * real_power(radius, p + 2) /
         static_cast<double>(p + 2);
}

} // namespace cylinder_detail

[[nodiscard]] inline std::complex<double>
evaluate_cylinder(const CylinderSeries &series, double r, double theta, double z) {
  return cylinder_detail::series_jet(series, r, theta, z).v;
}

/// Gradient in the Cartesian embedding. Defined at `r = 0`; `θ` is not a basis
/// there.
[[nodiscard]] inline std::array<std::complex<double>, 3>
cylinder_gradient(const CylinderSeries &series, double r, double theta, double z) {
  const auto jet = cylinder_detail::series_jet(series, r, theta, z);
  return {jet.x, jet.y, jet.z};
}

[[nodiscard]] inline std::complex<double>
cylinder_laplacian(const CylinderSeries &series, double r, double theta, double z) {
  const auto jet = cylinder_detail::series_jet(series, r, theta, z);
  return jet.xx + jet.yy + jet.zz;
}

[[nodiscard]] inline std::complex<double>
cylinder_physical_laplacian(const CylinderSeries &series, const AffineMap &map,
                            double r, double theta, double z) {
  const auto jet = cylinder_detail::series_jet(series, r, theta, z);
  if (map.uses_cartesian_laplacian()) return jet.xx + jet.yy + jet.zz;
  double metric[9];
  map.inverse_metric(metric);
  const auto g = [&](int row, int col) { return metric[col * 3 + row]; };
  return g(0, 0) * jet.xx + g(1, 1) * jet.yy + g(2, 2) * jet.zz +
         (g(0, 1) + g(1, 0)) * jet.xy + (g(0, 2) + g(2, 0)) * jet.xz +
         (g(1, 2) + g(2, 1)) * jet.yz;
}

/// Integral of the series with the cylindrical weight `r`. Only `m = 0` contributes.
[[nodiscard]] inline std::complex<double>
integrate_cylinder(const CylinderSeries &series) {
  std::complex<double> sum;
  for (const CylinderHarmonic &mode : series.modes) {
    if (mode.angular != 0) continue;
    sum += mode.coefficient *
           cylinder_detail::disk_moment(series.domain.radius, mode.radial) *
           cylinder_detail::integrate_axial(series.domain, mode.axial);
  }
  return sum;
}

/**
 * @brief Outward flux of `grad u` through the physical boundary.
 *
 * Periodic cylinders integrate the radial wall only. Finite cylinders add
 * both caps. The axis is omitted. The divergence theorem says this equals
 * `integrate_cylinder` of the Laplacian for every series in this space.
 */
[[nodiscard]] inline std::complex<double>
cylinder_boundary_flux(const CylinderSeries &series) {
  std::complex<double> flux;
  const double radius = series.domain.radius;
  for (const CylinderHarmonic &mode : series.modes) {
    if (mode.angular != 0) continue;
    const int p = 2 * mode.radial;
    const auto axial = cylinder_detail::integrate_axial(series.domain, mode.axial);
    if (p > 0) {
      flux += mode.coefficient * static_cast<double>(p) *
              cylinder_detail::real_power(radius, p) * (2.0 * std::numbers::pi) *
              axial;
    }
    if (series.domain.axial != CylinderAxialPolicy::Finite || mode.axial == 0)
      continue;
    const auto area = cylinder_detail::disk_moment(radius, mode.radial);
    const double height = series.domain.length();
    const auto upper = mode.coefficient * static_cast<double>(mode.axial) *
                       cylinder_detail::real_power(height, mode.axial - 1) * area;
    flux += upper;
    if (mode.axial == 1) flux -= mode.coefficient * area;
  }
  return flux;
}

/**
 * @brief Exact diffusion `∂t u = D Δu` for a z-independent axisymmetric series.
 *
 * Angular and axial modes are rejected: a wall boundary condition belongs to
 * the boundary roadmap, and the finite-z production basis is not selected.
 * The monomials in `r²` stay closed and the update is the exact ODE.
 */
[[nodiscard]] inline CylinderSeries
diffuse_axisymmetric_cylinder(const CylinderSeries &series, double diffusivity,
                              double dt) {
  if (!std::isfinite(diffusivity) || !std::isfinite(dt)) {
    throw std::invalid_argument("cylinder diffusion needs finite D and dt");
  }
  int max_radial = 0;
  for (const CylinderHarmonic &mode : series.modes) {
    if (mode.angular != 0 || mode.axial != 0) {
      throw std::invalid_argument(
          "axisymmetric diffusion advances m = 0 and a flat axial mode only");
    }
    if (mode.radial > max_radial) max_radial = mode.radial;
  }
  std::vector<std::vector<std::complex<double>>> polynomial(
      static_cast<std::size_t>(max_radial + 1), {0});
  for (const CylinderHarmonic &mode : series.modes) {
    polynomial[static_cast<std::size_t>(mode.radial)][0] += mode.coefficient;
  }
  for (int radial = max_radial - 1; radial >= 0; --radial) {
    const double couple = 4.0 * diffusivity * static_cast<double>(radial + 1) *
                          static_cast<double>(radial + 1);
    const auto &higher = polynomial[static_cast<std::size_t>(radial + 1)];
    auto &current = polynomial[static_cast<std::size_t>(radial)];
    if (current.size() < higher.size() + 1) current.resize(higher.size() + 1);
    for (std::size_t power = 0; power < higher.size(); ++power) {
      current[power + 1] += couple * higher[power] / static_cast<double>(power + 1);
    }
  }
  CylinderSeries advanced;
  advanced.domain = series.domain;
  for (int radial = 0; radial <= max_radial; ++radial) {
    std::complex<double> value;
    std::complex<double> time_power = 1;
    for (const auto coefficient : polynomial[static_cast<std::size_t>(radial)]) {
      value += coefficient * time_power;
      time_power *= dt;
    }
    advanced.modes.push_back(CylinderHarmonic{0, radial, 0, value});
  }
  return advanced;
}

} // namespace pfc::geometry
