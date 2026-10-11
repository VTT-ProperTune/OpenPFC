// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file chebyshev.hpp
 * @brief FFT-backed Chebyshev transform on a Lobatto grid.
 *
 * Values on x_j = cos(pi j / n) are an even extension of a cosine series.
 * The transform and the first derivative use one ordinary complex FFT of
 * length 2n. The mixed Laplacian differentiates one periodic Fourier
 * direction and this Chebyshev axis on a single rank. Dirichlet
 * Poisson matches both endpoint values. Neumann Poisson matches both
 * endpoint slopes and sets the integral to zero. Dirichlet Helmholtz
 * solves u'' - lambda u = f with those endpoint values. Neumann
 * Helmholtz matches both endpoint slopes. Robin Poisson mixes the
 * value and the slope at each end. Robin Helmholtz uses the same
 * mix for u'' - lambda u = f. A zero lambda reuses the Robin
 * Poisson integral. Dirichlet values on a Fourier × Chebyshev
 * rank reuse the Helmholtz solve with λ = k². Neumann slopes on
 * that rank do the same, and the mean mode keeps the integral
 * gauge. Robin data on that rank mixes the value and the slope
 * with weights that do not vary along x. Helmholtz on that rank
 * adds λ to k² and reuses those weights. A zero λ reuses the
 * Robin Poisson solve. A second periodic direction adds k_y²
 * on that rank. Dirichlet values there reuse the Helmholtz
 * solve with λ = k_x² + k_y². Neumann slopes there do the
 * same, and the mean mode keeps the integral gauge. Robin
 * data there mixes the value and the slope with weights that
 * do not vary on the periodic plane. Helmholtz there adds λ
 * to k_x² + k_y² and reuses those weights. A zero λ reuses
 * the Robin Poisson solve. Partial derivatives on that rank
 * use the same wavenumbers. The Nyquist sample of a periodic
 * first derivative is zero.
 *
 * The FFT plans are created on each call. Do not call these functions
 * concurrently.
 */

#include <span>
#include <vector>

namespace pfc::fft {

/// Lobatto nodes from +1 down to -1. Length is `degree + 1`.
[[nodiscard]] std::vector<double> chebyshev_lobatto(int degree);

/// Coefficients a_0..a_n of the interpolant through `values`.
[[nodiscard]] std::vector<double>
chebyshev_coefficients(std::span<const double> values);

/// Values on the Lobatto grid of coefficients a_0..a_n.
[[nodiscard]] std::vector<double>
chebyshev_values(std::span<const double> coefficients);

/// First derivative on the same Lobatto grid, including both endpoints.
[[nodiscard]] std::vector<double>
chebyshev_derivative(std::span<const double> values);

/// Laplacian d²/dz² + d²/dx² on one rank.
/// `nx` uniform nodes span a periodic interval of length `period`.
/// The other axis is Chebyshev–Lobatto, stored from +1 down to -1.
/// `values` is row-major with the periodic index contiguous.
[[nodiscard]] std::vector<double>
fourier_chebyshev_laplacian(std::span<const double> values, int nx, double period);

/// Laplacian d²/dz² + d²/dy² + d²/dx² on one rank.
/// `nx` and `ny` uniform nodes span periodic intervals `period_x`
/// and `period_y`. The bounded axis is Chebyshev–Lobatto, stored
/// from +1 down to -1. `values` is row-major with x contiguous,
/// then y, then z. A unit count in either periodic direction
/// reuses the one-direction Laplacian.
[[nodiscard]] std::vector<double>
fourier_chebyshev_laplacian(std::span<const double> values, int nx, int ny,
                            double period_x, double period_y);

/// Solve (d²/dz² + d²/dx²) u = forcing, periodic in x.
/// Both Chebyshev ends are Dirichlet. `value_at_plus` and
/// `value_at_minus` each have length `nx`.
/// `forcing` is row-major with the periodic index contiguous.
[[nodiscard]] std::vector<double> fourier_chebyshev_dirichlet_poisson(
    std::span<const double> forcing, int nx, double period,
    std::span<const double> value_at_plus, std::span<const double> value_at_minus);

/// Solve (d²/dz² + d²/dy² + d²/dx²) u = forcing on one rank.
/// Both Chebyshev ends are Dirichlet. `value_at_plus` and
/// `value_at_minus` each have length `nx * ny`, row-major with x
/// contiguous. Each mode reuses the Dirichlet Helmholtz solve
/// with λ = k_x² + k_y². A unit count in either periodic
/// direction reuses the one-direction solve.
[[nodiscard]] std::vector<double>
fourier_chebyshev_dirichlet_poisson(std::span<const double> forcing, int nx, int ny,
                                    double period_x, double period_y,
                                    std::span<const double> value_at_plus,
                                    std::span<const double> value_at_minus);

/// Solve (d²/dz² + d²/dx²) u = forcing, periodic in x.
/// Both Chebyshev ends prescribe the slope. `slope_at_plus` and
/// `slope_at_minus` each have length `nx`. The mean mode keeps the
/// zero-integral gauge and rejects an incompatible slope jump.
/// `forcing` is row-major with the periodic index contiguous.
[[nodiscard]] std::vector<double> fourier_chebyshev_neumann_poisson(
    std::span<const double> forcing, int nx, double period,
    std::span<const double> slope_at_plus, std::span<const double> slope_at_minus);

/// Solve (d²/dz² + d²/dy² + d²/dx²) u = forcing on one rank.
/// Both Chebyshev ends prescribe the slope. `slope_at_plus` and
/// `slope_at_minus` each have length `nx * ny`, row-major with x
/// contiguous. Each mode reuses the Neumann Helmholtz solve with
/// λ = k_x² + k_y². The mean mode keeps the zero-integral gauge.
/// A unit count in either periodic direction reuses the
/// one-direction solve.
[[nodiscard]] std::vector<double>
fourier_chebyshev_neumann_poisson(std::span<const double> forcing, int nx, int ny,
                                  double period_x, double period_y,
                                  std::span<const double> slope_at_plus,
                                  std::span<const double> slope_at_minus);

/// Solve (d²/dz² + d²/dx²) u = forcing, periodic in x.
/// Each end imposes value_weight * u + slope_weight * du/dz = data.
/// `data_at_plus` and `data_at_minus` each have length `nx`.
/// The weights do not vary along x. A zero wavenumber reuses the
/// Robin Poisson solve, including its integral gauge when both
/// ends prescribe only the slope.
/// `forcing` is row-major with the periodic index contiguous.
[[nodiscard]] std::vector<double> fourier_chebyshev_robin_poisson(
    std::span<const double> forcing, int nx, double period, double value_weight_plus,
    double slope_weight_plus, std::span<const double> data_at_plus,
    double value_weight_minus, double slope_weight_minus,
    std::span<const double> data_at_minus);

/// Solve (d²/dz² + d²/dy² + d²/dx²) u = forcing on one rank.
/// Each end imposes value_weight * u + slope_weight * du/dz = data.
/// `data_at_plus` and `data_at_minus` each have length `nx * ny`,
/// row-major with x contiguous. The weights do not vary on the
/// periodic plane. The slope weight multiplies du/dz, so an
/// outward unit weight is +1 at z = +1 and -1 at z = -1. Each
/// mode reuses the Robin Helmholtz solve with λ = k_x² + k_y².
/// A zero pair of wavenumbers reuses the Robin Poisson solve.
/// A unit count in either periodic direction reuses the
/// one-direction solve.
[[nodiscard]] std::vector<double> fourier_chebyshev_robin_poisson(
    std::span<const double> forcing, int nx, int ny, double period_x,
    double period_y, double value_weight_plus, double slope_weight_plus,
    std::span<const double> data_at_plus, double value_weight_minus,
    double slope_weight_minus, std::span<const double> data_at_minus);

/// Solve (d²/dz² + d²/dx² - λ) u = forcing, periodic in x.
/// Each end imposes value_weight * u + slope_weight * du/dz = data.
/// `data_at_plus` and `data_at_minus` each have length `nx`.
/// The weights do not vary along x. A zero λ reuses the Robin
/// Poisson solve. Each Fourier mode solves u'' - (k² + λ) u = f.
/// `forcing` is row-major with the periodic index contiguous.
[[nodiscard]] std::vector<double> fourier_chebyshev_robin_helmholtz(
    std::span<const double> forcing, int nx, double period, double lambda,
    double value_weight_plus, double slope_weight_plus,
    std::span<const double> data_at_plus, double value_weight_minus,
    double slope_weight_minus, std::span<const double> data_at_minus);

/// Solve (d²/dz² + d²/dy² + d²/dx² - λ) u = forcing on one rank.
/// Each end imposes value_weight * u + slope_weight * du/dz = data.
/// `data_at_plus` and `data_at_minus` each have length `nx * ny`,
/// row-major with x contiguous. The weights do not vary on the
/// periodic plane. The slope weight multiplies du/dz, so an
/// outward unit weight is +1 at z = +1 and -1 at z = -1. Each
/// mode solves u'' - (k_x² + k_y² + λ) u = f. A zero λ reuses
/// the Robin Poisson solve. A unit count in either periodic
/// direction reuses the one-direction solve.
[[nodiscard]] std::vector<double> fourier_chebyshev_robin_helmholtz(
    std::span<const double> forcing, int nx, int ny, double period_x,
    double period_y, double lambda, double value_weight_plus,
    double slope_weight_plus, std::span<const double> data_at_plus,
    double value_weight_minus, double slope_weight_minus,
    std::span<const double> data_at_minus);

struct PartialDerivatives {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
};

/// Partial derivatives of one scalar on the mixed grid.
/// x and y are periodic. z is Chebyshev–Lobatto from +1 down to
/// -1. A unit count leaves that periodic derivative at zero and
/// does not build a plan. The Nyquist sample of a periodic first
/// derivative is zero.
[[nodiscard]] PartialDerivatives
fourier_chebyshev_gradient(std::span<const double> values, int nx, int ny,
                           double period_x, double period_y);

/// Solve u'' = forcing on [-1, 1], with both endpoints prescribed.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid.
[[nodiscard]] std::vector<double>
chebyshev_dirichlet_poisson(std::span<const double> forcing, double value_at_plus,
                            double value_at_minus);

/// Solve u'' = forcing on [-1, 1], with both endpoint slopes prescribed.
/// The additive constant is chosen so the integral over the interval is zero.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid. Incompatible slopes are rejected.
[[nodiscard]] std::vector<double>
chebyshev_neumann_poisson(std::span<const double> forcing, double slope_at_plus,
                          double slope_at_minus);

/// Solve u'' - lambda u = forcing on [-1, 1], with both endpoints
/// prescribed. A zero lambda reuses the Dirichlet Poisson solve.
/// Nonzero lambda uses twice-integrated coefficient equations and boundary
/// rows with a row-scaled, pivoted solve (quadratic storage, cubic worst-case
/// work). It avoids downward-recurrence cancellation; no modes are filtered.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid.
[[nodiscard]] std::vector<double>
chebyshev_dirichlet_helmholtz(std::span<const double> forcing, double lambda,
                              double value_at_plus, double value_at_minus);

/// Solve u'' - lambda u = forcing on [-1, 1], with both endpoint
/// slopes prescribed. A zero lambda reuses the Neumann Poisson solve.
/// Nonzero lambda uses the same integrated coefficient solve as Dirichlet.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid.
[[nodiscard]] std::vector<double>
chebyshev_neumann_helmholtz(std::span<const double> forcing, double lambda,
                            double slope_at_plus, double slope_at_minus);

/// Solve u'' = forcing on [-1, 1].
/// Each end imposes weight_value * u + weight_slope * u' = data.
/// Pure value weights reuse the Dirichlet solve. Pure slope weights
/// reuse the Neumann solve and its zero-integral gauge.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid.
[[nodiscard]] std::vector<double>
chebyshev_robin_poisson(std::span<const double> forcing, double value_weight_plus,
                        double slope_weight_plus, double data_plus,
                        double value_weight_minus, double slope_weight_minus,
                        double data_minus);

/// Solve u'' - lambda u = forcing on [-1, 1].
/// Each end imposes weight_value * u + weight_slope * u' = data.
/// A zero lambda reuses the Robin Poisson solve. Pure value weights
/// reuse the Dirichlet Helmholtz solve. Pure slope weights reuse the
/// Neumann Helmholtz solve.
/// `forcing` is sampled on the Lobatto grid from +1 down to -1.
/// The result uses that same grid.
[[nodiscard]] std::vector<double>
chebyshev_robin_helmholtz(std::span<const double> forcing, double lambda,
                          double value_weight_plus, double slope_weight_plus,
                          double data_plus, double value_weight_minus,
                          double slope_weight_minus, double data_minus);

} // namespace pfc::fft
