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
 * reference integral, and the physical measure. An identity Jacobian uses
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
  std::complex<double> laplacian{};
  std::complex<double> integral{};
  double measure = 0;
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

/// One reading for whichever reference series the caller already holds.
template <class Series, class Station>
[[nodiscard]] inline ReferenceReading
sample_reference(const Series &series, const AffineMap &map, Station station) {
  ReferenceReading reading;
  reading.laplacian = reference_laplacian(series, map, station);
  reading.integral = reference_integral(series);
  reading.measure = reference_measure(series.domain, map);
  return reading;
}

} // namespace pfc::geometry
