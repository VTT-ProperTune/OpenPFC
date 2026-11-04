// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file filled_disk.hpp
 * @brief Filled polar reference domain and its scalar calculus.
 *
 * Coordinates are `(r, θ)` with `0 <= r <= R` and `θ` `2π`-periodic.
 * `r = 0` is the origin, not a boundary. The only physical boundary is the
 * wall `r = R` (`DiskBoundary::RadialWall`), which is the site a boundary
 * condition from the boundary roadmap can name. This type does not store one.
 *
 * A scalar mode reuses the cylinder's in-plane factor
 * `coefficient * (x ± iy)^{|m|} * (x² + y²)^k`. That is
 * `r^{|m|} (r²)^k e^{imθ}` and is smooth at the origin, so the Laplacian
 * never replaces `1/r` by `1/(r+ε)`. The mode is evaluated as a flat axial
 * mode of `FilledCylinder`. Chebyshev, Jacobi, and Zernike radial bases are
 * not selected; this polynomial is the verification space.
 *
 * `AffineMap` sends the embedding `(r cos θ, r sin θ, 0)` to a flat ellipse.
 * The area factor is `|column_0 × column_1|`. The physical Laplacian is the
 * Laplace–Beltrami operator of that induced metric, not the 3-D volume
 * Laplacian. An identity Jacobian uses the in-plane trace.
 */

#include <openpfc/kernel/geometry/filled_cylinder.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace pfc::geometry {

enum class DiskBoundary { RadialWall };

struct FilledDisk {
  double radius = 1;

  double area() const noexcept { return std::numbers::pi * radius * radius; }
};

struct DiskHarmonic {
  int angular = 0;
  int radial = 0;
  std::complex<double> coefficient{};
};

struct DiskSeries {
  FilledDisk domain{};
  std::vector<DiskHarmonic> modes{};
};

[[nodiscard]] inline bool disk_has_boundary(const FilledDisk &domain,
                                            DiskBoundary piece) noexcept {
  (void)domain;
  return piece == DiskBoundary::RadialWall;
}

[[nodiscard]] inline FilledDisk filled_disk(double radius) {
  if (!std::isfinite(radius) || !(radius > 0.0)) {
    throw std::invalid_argument("filled disk needs a positive radius");
  }
  return FilledDisk{radius};
}

[[nodiscard]] inline bool disk_contains(const FilledDisk &domain, double r,
                                        double theta) noexcept {
  return std::isfinite(r) && std::isfinite(theta) && r >= 0.0 && r <= domain.radius;
}

inline void add_disk_harmonic(DiskSeries &series, DiskHarmonic mode) {
  if (mode.radial < 0 || mode.radial > 12 || mode.angular < -24 ||
      mode.angular > 24 || !std::isfinite(mode.coefficient.real()) ||
      !std::isfinite(mode.coefficient.imag())) {
    throw std::invalid_argument("disk harmonic degree or coefficient is not valid");
  }
  series.modes.push_back(mode);
}

namespace disk_detail {

inline void require_radius(const FilledDisk &domain) {
  if (!std::isfinite(domain.radius) || !(domain.radius > 0.0)) {
    throw std::invalid_argument("filled disk needs a positive radius");
  }
}

inline CylinderSeries as_flat_cylinder(const DiskSeries &disk) {
  require_radius(disk.domain);
  CylinderSeries series;
  series.domain =
      filled_cylinder(disk.domain.radius, 0.0, 1.0, CylinderAxialPolicy::Periodic);
  series.modes.reserve(disk.modes.size());
  for (const DiskHarmonic &mode : disk.modes) {
    series.modes.push_back(
        CylinderHarmonic{mode.angular, mode.radial, 0, mode.coefficient});
  }
  return series;
}

inline cylinder_detail::Jet in_plane_jet(const DiskSeries &disk, double r,
                                         double theta) {
  return cylinder_detail::series_jet(as_flat_cylinder(disk), r, theta, 0.0);
}

} // namespace disk_detail

[[nodiscard]] inline Vec3 disk_embedding(double r, double theta) {
  return cylinder_embedding(r, theta, 0.0);
}

[[nodiscard]] inline Vec3 disk_physical_point(double r, double theta,
                                              const AffineMap &map) {
  return map.to_physical(disk_embedding(r, theta));
}

/// Area of the image. The factor is the parallelogram of the first two columns.
[[nodiscard]] inline double disk_physical_area(const FilledDisk &domain,
                                               const AffineMap &map) {
  disk_detail::require_radius(domain);
  return norm(cross(map.column(0), map.column(1))) * domain.area();
}

[[nodiscard]] inline std::complex<double> evaluate_disk(const DiskSeries &series,
                                                        double r, double theta) {
  return disk_detail::in_plane_jet(series, r, theta).v;
}

/// Gradient in the Cartesian embedding. Defined at `r = 0`.
[[nodiscard]] inline std::array<std::complex<double>, 2>
disk_gradient(const DiskSeries &series, double r, double theta) {
  const auto jet = disk_detail::in_plane_jet(series, r, theta);
  return {jet.x, jet.y};
}

[[nodiscard]] inline std::complex<double> disk_laplacian(const DiskSeries &series,
                                                         double r, double theta) {
  const auto jet = disk_detail::in_plane_jet(series, r, theta);
  return jet.xx + jet.yy;
}

/**
 * @brief Laplace–Beltrami operator of the affinely mapped disk.
 *
 * The metric is the Gram matrix of the first two columns. An identity
 * Jacobian returns the in-plane trace and does not form that matrix.
 */
[[nodiscard]] inline std::complex<double>
disk_physical_laplacian(const DiskSeries &series, const AffineMap &map, double r,
                        double theta) {
  const auto jet = disk_detail::in_plane_jet(series, r, theta);
  if (map.uses_cartesian_laplacian()) return jet.xx + jet.yy;
  const Vec3 a = map.column(0);
  const Vec3 b = map.column(1);
  const double g00 = dot(a, a);
  const double g01 = dot(a, b);
  const double g11 = dot(b, b);
  const double gram = g00 * g11 - g01 * g01;
  if (!(gram > 0.0) || !std::isfinite(gram)) {
    throw std::invalid_argument("the affine image of the disk has no area");
  }
  const double ginv00 = g11 / gram;
  const double ginv01 = -g01 / gram;
  const double ginv11 = g00 / gram;
  return ginv00 * jet.xx + ginv11 * jet.yy + 2.0 * ginv01 * jet.xy;
}

/// Integral with the polar weight `r`. Only `m = 0` contributes.
[[nodiscard]] inline std::complex<double> integrate_disk(const DiskSeries &series) {
  return integrate_cylinder(disk_detail::as_flat_cylinder(series));
}

/// Outward flux of `grad u` through `r = R`. The origin is omitted.
[[nodiscard]] inline std::complex<double>
disk_boundary_flux(const DiskSeries &series) {
  return cylinder_boundary_flux(disk_detail::as_flat_cylinder(series));
}

} // namespace pfc::geometry
