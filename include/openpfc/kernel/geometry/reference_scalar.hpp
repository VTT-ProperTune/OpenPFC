// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file reference_scalar.hpp
 * @brief Operator boundary for a reference scalar and a physical map.
 *
 * The reference domain is the series type: a periodic Cartesian cell stays
 * on `AffineMap` and `laplacian_symbol`, and a filled cylinder, disk, or
 * ball stays on its own series. The physical map is always `AffineMap`.
 * There is no domain-shape enum. A caller passes the series it already
 * holds. Overload resolution selects the chart.
 *
 * `sample_reference` reads the physical Laplacian at one station, the
 * reference integral, the Jacobian-scaled physical integral, and physical
 * measure. Both integrals have explicit frame names so users cannot compute
 * an invalid physical mean by mixing coordinate systems. Identity uses
 * the embedding Laplacian. Any other constant map uses the inverse metric
 * already implemented for that chart. This header does not add a boundary
 * condition, a material model, or a second Cartesian fast path.
 */

#include <openpfc/kernel/geometry/filled_ball.hpp>
#include <openpfc/kernel/geometry/filled_cylinder.hpp>
#include <openpfc/kernel/geometry/filled_disk.hpp>

#include <complex>

namespace pfc::geometry {

struct CylinderStation {
  double r = 0;
  double theta = 0;
  double z = 0;
};

struct DiskStation {
  double r = 0;
  double theta = 0;
};

struct BallStation {
  double r = 0;
  double theta = 0;
  double phi = 0;
};

struct ReferenceReading {
  /// Physical-space scalar Laplacian at the requested station.
  std::complex<double> laplacian{};
  /// Integral with the native reference chart's volume/area element.
  std::complex<double> reference_integral{};
  /// Integral with the constant map's physical volume/area element.
  std::complex<double> physical_integral{};
  /// Physical volume for 3D charts, physical area for the 2D disk.
  double physical_measure = 0;
};

[[nodiscard]] inline std::complex<double>
reference_laplacian(const CylinderSeries &series, const AffineMap &map,
                    CylinderStation station) {
  return cylinder_physical_laplacian(series, map, station.r, station.theta,
                                     station.z);
}

[[nodiscard]] inline std::complex<double>
reference_laplacian(const DiskSeries &series, const AffineMap &map,
                    DiskStation station) {
  return disk_physical_laplacian(series, map, station.r, station.theta);
}

[[nodiscard]] inline std::complex<double>
reference_laplacian(const BallSeries &series, const AffineMap &map,
                    BallStation station) {
  return ball_physical_laplacian(series, map, station.r, station.theta, station.phi);
}

[[nodiscard]] inline std::complex<double>
reference_integral(const CylinderSeries &series) {
  return integrate_cylinder(series);
}

[[nodiscard]] inline std::complex<double>
reference_integral(const DiskSeries &series) {
  return integrate_disk(series);
}

[[nodiscard]] inline std::complex<double>
reference_integral(const BallSeries &series) {
  return integrate_ball(series);
}

[[nodiscard]] inline double reference_measure(const FilledCylinder &domain,
                                              const AffineMap &map) {
  return cylinder_physical_volume(domain, map);
}

[[nodiscard]] inline double reference_measure(const FilledDisk &domain,
                                              const AffineMap &map) {
  return disk_physical_area(domain, map);
}

[[nodiscard]] inline double reference_measure(const FilledBall &domain,
                                              const AffineMap &map) {
  return ball_physical_volume(domain, map);
}

/// Unmapped size of each chart, before the constant physical Jacobian.
[[nodiscard]] inline double reference_domain_measure(const FilledCylinder &domain) {
  return domain.volume();
}
[[nodiscard]] inline double reference_domain_measure(const FilledDisk &domain) {
  return domain.area();
}
[[nodiscard]] inline double reference_domain_measure(const FilledBall &domain) {
  return domain.volume();
}

/// One reading for whichever reference series the caller already holds.
/// The combined integral/measure values use compatible coordinate frames.
template <class Series, class Station>
[[nodiscard]] inline ReferenceReading
sample_reference(const Series &series, const AffineMap &map, Station station) {
  ReferenceReading reading;
  reading.laplacian = reference_laplacian(series, map, station);
  reading.reference_integral = reference_integral(series);
  reading.physical_measure = reference_measure(series.domain, map);
  const double reference_size = reference_domain_measure(series.domain);
  reading.physical_integral =
      reading.reference_integral * (reading.physical_measure / reference_size);
  return reading;
}

} // namespace pfc::geometry
