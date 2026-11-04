// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file pipe.hpp
 * @brief Steady laminar Hagen–Poiseuille balance on a filled cylinder.
 *
 * The pipe is the reference domain `0 <= r <= R`, with `θ` and `z`
 * periodic. `r = R` is the radial wall. `r = 0` is an axis, not a
 * boundary. A periodic axis has no caps. The axial velocity is a
 * scalar series on that domain. Its Laplacian, gradient, and integral
 * are the installed cylinder operators, including the identity-map
 * physical Laplacian. The basis is smooth at the axis, so the balance
 * does not replace `1/r` by `1/(r+ε)`.
 *
 * No-slip is the value of that series at `r = R`. The modes used here
 * are built to vanish there. This is not a ghost scheme and it does
 * not close the boundary-condition roadmap. A constant axial body
 * force is the pressure gradient. The steady statement is
 * `ν Δu + G = 0`. The velocity is divergence-free, so that residual
 * needs no further pressure correction.
 *
 * Radial degree 0 has only the zero field that meets the wall, and the
 * momentum residual stays at `|G|`. Degree 1 contains the parabola.
 * Higher radial degrees, one angular mode, and one axial mode stay at
 * coefficient zero when they are admitted. The periodic Fourier pipe
 * and the Chebyshev channel are unchanged. This is not a turbulent
 * pipe and not a Cartesian mask.
 */

#include <openpfc/kernel/geometry/filled_cylinder.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace flow {

struct PipeRequest {
  double radius{1.0};
  double length{1.0};
  double viscosity{0.2};
  double force{0.8};
  int radial_degree{1};
  bool angular{false};
  bool axial{false};
};

struct PipeBalance {
  bool radial_wall{false};
  bool has_caps{false};
  double wall_error{0.0};
  double axis_value{0.0};
  double axis_gradient{0.0};
  double flow_rate{0.0};
  double flow_imag{0.0};
  double wall_shear{0.0};
  double shear_spread{0.0};
  double momentum{0.0};
  double divergence{0.0};
  double metric_gap{0.0};
  double leading_coefficient{0.0};
  double higher_coefficient{0.0};
  double angular_coefficient{0.0};
  double axial_coefficient{0.0};
};

namespace pipe_detail {

struct Sample {
  double r;
  double theta;
  double z;
};

inline double whole_power(double base, int exponent) {
  double value = 1.0;
  for (int i = 0; i < exponent; ++i) value *= base;
  return value;
}

inline pfc::geometry::CylinderSeries
radial_mode(const pfc::geometry::FilledCylinder &domain, int radial) {
  pfc::geometry::CylinderSeries series;
  series.domain = domain;
  const double wall = whole_power(domain.radius * domain.radius, radial);
  pfc::geometry::add_cylinder_harmonic(
      series, pfc::geometry::CylinderHarmonic{0, 0, 0, wall});
  pfc::geometry::add_cylinder_harmonic(
      series, pfc::geometry::CylinderHarmonic{0, radial, 0, -1.0});
  return series;
}

inline pfc::geometry::CylinderSeries
angular_mode(const pfc::geometry::FilledCylinder &domain) {
  pfc::geometry::CylinderSeries series;
  series.domain = domain;
  const double wall = domain.radius * domain.radius;
  pfc::geometry::add_cylinder_harmonic(
      series, pfc::geometry::CylinderHarmonic{1, 0, 0, wall});
  pfc::geometry::add_cylinder_harmonic(
      series, pfc::geometry::CylinderHarmonic{1, 1, 0, -1.0});
  return series;
}

inline pfc::geometry::CylinderSeries
axial_mode(const pfc::geometry::FilledCylinder &domain) {
  pfc::geometry::CylinderSeries series;
  series.domain = domain;
  const double wall = domain.radius * domain.radius;
  pfc::geometry::add_cylinder_harmonic(
      series, pfc::geometry::CylinderHarmonic{0, 0, 1, wall});
  pfc::geometry::add_cylinder_harmonic(
      series, pfc::geometry::CylinderHarmonic{0, 1, 1, -1.0});
  return series;
}

inline void accumulate(pfc::geometry::CylinderSeries &sum,
                       const pfc::geometry::CylinderSeries &mode, double scale) {
  if (scale == 0.0) return;
  for (pfc::geometry::CylinderHarmonic harmonic : mode.modes) {
    harmonic.coefficient *= scale;
    pfc::geometry::add_cylinder_harmonic(sum, harmonic);
  }
}

inline std::complex<double>
physical_laplacian(const pfc::geometry::CylinderSeries &series,
                   const pfc::geometry::AffineMap &map, Sample sample) {
  return pfc::geometry::cylinder_physical_laplacian(series, map, sample.r,
                                                    sample.theta, sample.z);
}

inline std::vector<Sample>
interior_samples(const pfc::geometry::FilledCylinder &domain, bool fit) {
  const double radius = domain.radius;
  const double length = domain.length();
  const double radii[] = {0.0, 0.2 * radius, 0.5 * radius, 0.8 * radius};
  const double extra[] = {0.15 * radius, 0.45 * radius, 0.7 * radius, 0.95 * radius};
  const double thetas[] = {0.2, 1.7, 3.3};
  const double heights[] = {0.1, 0.5, 0.9};
  std::vector<Sample> samples;
  const double *radial = fit ? radii : extra;
  for (double r :
       std::initializer_list<double>{radial[0], radial[1], radial[2], radial[3]}) {
    for (double theta : thetas) {
      for (double height : heights) {
        samples.push_back(Sample{r, theta, domain.z_lower + height * length});
      }
    }
  }
  return samples;
}

inline bool solve_columns(std::vector<double> &matrix, std::vector<double> &rhs) {
  const int n = static_cast<int>(rhs.size());
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double best = std::abs(matrix[static_cast<std::size_t>(col * n + col)]);
    for (int row = col + 1; row < n; ++row) {
      const double value = std::abs(matrix[static_cast<std::size_t>(row * n + col)]);
      if (value > best) {
        best = value;
        pivot = row;
      }
    }
    if (!(best > 1.0e-14)) return false;
    if (pivot != col) {
      for (int j = 0; j < n; ++j) {
        std::swap(matrix[static_cast<std::size_t>(col * n + j)],
                  matrix[static_cast<std::size_t>(pivot * n + j)]);
      }
      std::swap(rhs[static_cast<std::size_t>(col)],
                rhs[static_cast<std::size_t>(pivot)]);
    }
    const double diagonal = matrix[static_cast<std::size_t>(col * n + col)];
    for (int row = col + 1; row < n; ++row) {
      const double factor =
          matrix[static_cast<std::size_t>(row * n + col)] / diagonal;
      for (int j = col; j < n; ++j) {
        matrix[static_cast<std::size_t>(row * n + j)] -=
            factor * matrix[static_cast<std::size_t>(col * n + j)];
      }
      rhs[static_cast<std::size_t>(row)] -=
          factor * rhs[static_cast<std::size_t>(col)];
    }
  }
  for (int row = n - 1; row >= 0; --row) {
    double sum = rhs[static_cast<std::size_t>(row)];
    for (int col = row + 1; col < n; ++col) {
      sum -= matrix[static_cast<std::size_t>(row * n + col)] *
             rhs[static_cast<std::size_t>(col)];
    }
    rhs[static_cast<std::size_t>(row)] =
        sum / matrix[static_cast<std::size_t>(row * n + row)];
  }
  return true;
}

inline double abs_complex(std::complex<double> value) {
  return std::hypot(value.real(), value.imag());
}

} // namespace pipe_detail

[[nodiscard]] inline PipeBalance hagen_poiseuille(const PipeRequest &request) {
  if (!std::isfinite(request.radius) || !(request.radius > 0.0) ||
      !std::isfinite(request.length) || !(request.length > 0.0) ||
      !std::isfinite(request.viscosity) || !(request.viscosity > 0.0) ||
      !std::isfinite(request.force) || request.radial_degree < 0 ||
      request.radial_degree > 4) {
    throw std::invalid_argument(
        "Hagen-Poiseuille needs a positive radius, length, and viscosity");
  }
  const auto domain =
      pfc::geometry::filled_cylinder(request.radius, 0.0, request.length,
                                     pfc::geometry::CylinderAxialPolicy::Periodic);
  const auto map = pfc::geometry::identity_affine_map();
  const auto fit = pipe_detail::interior_samples(domain, true);

  std::vector<pfc::geometry::CylinderSeries> modes;
  for (int radial = 1; radial <= request.radial_degree; ++radial) {
    modes.push_back(pipe_detail::radial_mode(domain, radial));
  }
  const int angular_index = request.angular ? static_cast<int>(modes.size()) : -1;
  if (request.angular) modes.push_back(pipe_detail::angular_mode(domain));
  const int axial_index = request.axial ? static_cast<int>(modes.size()) : -1;
  if (request.axial) modes.push_back(pipe_detail::axial_mode(domain));

  std::vector<double> coefficients(modes.size(), 0.0);
  if (!modes.empty()) {
    const int unknowns = static_cast<int>(modes.size());
    const int rows = static_cast<int>(fit.size()) * 2;
    std::vector<double> columns(static_cast<std::size_t>(rows * unknowns), 0.0);
    std::vector<double> target(static_cast<std::size_t>(rows), 0.0);
    for (int sample = 0; sample < static_cast<int>(fit.size()); ++sample) {
      target[static_cast<std::size_t>(2 * sample)] = -request.force;
      for (int mode = 0; mode < unknowns; ++mode) {
        const auto lap = pipe_detail::physical_laplacian(
            modes[static_cast<std::size_t>(mode)], map,
            fit[static_cast<std::size_t>(sample)]);
        const double scale = request.viscosity;
        columns[static_cast<std::size_t>(2 * sample * unknowns + mode)] =
            scale * lap.real();
        columns[static_cast<std::size_t>((2 * sample + 1) * unknowns + mode)] =
            scale * lap.imag();
      }
    }
    std::vector<double> normal(static_cast<std::size_t>(unknowns * unknowns), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(unknowns), 0.0);
    for (int row = 0; row < rows; ++row) {
      for (int i = 0; i < unknowns; ++i) {
        const double left = columns[static_cast<std::size_t>(row * unknowns + i)];
        rhs[static_cast<std::size_t>(i)] +=
            left * target[static_cast<std::size_t>(row)];
        for (int j = 0; j < unknowns; ++j) {
          normal[static_cast<std::size_t>(i * unknowns + j)] +=
              left * columns[static_cast<std::size_t>(row * unknowns + j)];
        }
      }
    }
    // Normal-equation columns scale with viscosity², so an absolute
    // pivot threshold would incorrectly reject valid low-viscosity pipes.
    // Normalize columns to unit diagonal before the dimensionless solve.
    std::vector<double> scale(static_cast<std::size_t>(unknowns));
    for (int i = 0; i < unknowns; ++i) {
      const double diagonal = normal[static_cast<std::size_t>(i * unknowns + i)];
      if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
        throw std::runtime_error("Hagen-Poiseuille basis column is singular");
      }
      scale[static_cast<std::size_t>(i)] = std::sqrt(diagonal);
    }
    for (int i = 0; i < unknowns; ++i) {
      const double si = scale[static_cast<std::size_t>(i)];
      rhs[static_cast<std::size_t>(i)] /= si;
      for (int j = 0; j < unknowns; ++j) {
        const double sj = scale[static_cast<std::size_t>(j)];
        const auto index = static_cast<std::size_t>(i * unknowns + j);
        normal[index] = (normal[index] / si) / sj;
      }
    }
    if (!pipe_detail::solve_columns(normal, rhs)) {
      throw std::runtime_error("Hagen-Poiseuille radial system is singular");
    }
    for (int i = 0; i < unknowns; ++i) {
      rhs[static_cast<std::size_t>(i)] /= scale[static_cast<std::size_t>(i)];
    }
    coefficients = std::move(rhs);
  }

  pfc::geometry::CylinderSeries velocity;
  velocity.domain = domain;
  for (std::size_t i = 0; i < modes.size(); ++i) {
    pipe_detail::accumulate(velocity, modes[i], coefficients[i]);
  }

  PipeBalance report;
  report.radial_wall = pfc::geometry::cylinder_has_boundary(
      domain, pfc::geometry::CylinderBoundary::RadialWall);
  report.has_caps = pfc::geometry::cylinder_has_boundary(
      domain, pfc::geometry::CylinderBoundary::LowerCap);
  if (!modes.empty()) report.leading_coefficient = coefficients[0];
  for (int radial = 2; radial <= request.radial_degree; ++radial) {
    report.higher_coefficient =
        std::max(report.higher_coefficient,
                 std::abs(coefficients[static_cast<std::size_t>(radial - 1)]));
  }
  if (angular_index >= 0) {
    report.angular_coefficient =
        coefficients[static_cast<std::size_t>(angular_index)];
  }
  if (axial_index >= 0) {
    report.axial_coefficient = coefficients[static_cast<std::size_t>(axial_index)];
  }

  const auto probe = pipe_detail::interior_samples(domain, false);
  for (const pipe_detail::Sample &sample : probe) {
    const auto lap = pipe_detail::physical_laplacian(velocity, map, sample);
    const auto reference = pfc::geometry::cylinder_laplacian(velocity, sample.r,
                                                             sample.theta, sample.z);
    report.metric_gap =
        std::max(report.metric_gap, pipe_detail::abs_complex(lap - reference));
    const double residual_real = request.viscosity * lap.real() + request.force;
    report.momentum = std::max(
        report.momentum, std::hypot(residual_real, request.viscosity * lap.imag()));
    const auto gradient =
        pfc::geometry::cylinder_gradient(velocity, sample.r, sample.theta, sample.z);
    report.divergence =
        std::max(report.divergence, pipe_detail::abs_complex(gradient[2]));
    const pfc::geometry::Vec3 reference_gradient{
        gradient[0].real(), gradient[1].real(), gradient[2].real()};
    const pfc::geometry::Vec3 physical = map.physical_gradient(reference_gradient);
    report.metric_gap =
        std::max(report.metric_gap,
                 std::hypot(physical.x - reference_gradient.x,
                            std::hypot(physical.y - reference_gradient.y,
                                       physical.z - reference_gradient.z)));
  }

  const pipe_detail::Sample axis{0.0, 0.0, 0.5 * request.length};
  const auto axis_value =
      pfc::geometry::evaluate_cylinder(velocity, axis.r, axis.theta, axis.z);
  report.axis_value = axis_value.real();
  const auto axis_gradient =
      pfc::geometry::cylinder_gradient(velocity, axis.r, axis.theta, axis.z);
  report.axis_gradient =
      std::max(pipe_detail::abs_complex(axis_gradient[0]),
               std::max(pipe_detail::abs_complex(axis_gradient[1]),
                        pipe_detail::abs_complex(axis_gradient[2])));

  const double thetas[] = {0.0, 1.3, 2.5, 4.2};
  const double heights[] = {0.0, 0.5, 1.0};
  bool have_shear = false;
  for (double theta : thetas) {
    for (double height : heights) {
      const double z = height * request.length;
      const auto value =
          pfc::geometry::evaluate_cylinder(velocity, request.radius, theta, z);
      report.wall_error =
          std::max(report.wall_error, pipe_detail::abs_complex(value));
      const auto gradient =
          pfc::geometry::cylinder_gradient(velocity, request.radius, theta, z);
      const pfc::geometry::Vec3 reference_gradient{
          gradient[0].real(), gradient[1].real(), gradient[2].real()};
      const pfc::geometry::Vec3 physical = map.physical_gradient(reference_gradient);
      const double outward =
          physical.x * std::cos(theta) + physical.y * std::sin(theta);
      const double shear = request.viscosity * outward;
      if (!have_shear) {
        report.wall_shear = shear;
        have_shear = true;
      }
      report.shear_spread =
          std::max(report.shear_spread, std::abs(shear - report.wall_shear));
    }
  }

  const auto inventory = pfc::geometry::integrate_cylinder(velocity);
  report.flow_rate = inventory.real() / request.length;
  report.flow_imag = inventory.imag() / request.length;
  return report;
}

} // namespace flow
