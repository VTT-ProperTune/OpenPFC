// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/** @file real_transform.hpp
 * @brief Planned uniform-grid sine/cosine transforms and spectral operators.
 * These are distinct from Chebyshev collocation. Forward scaling is FFTW raw;
 * backward applies the inverse divisor. See docs/api/boundary_transforms.md.
 */

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace pfc::fft {

/// Forward type; type II is inverted by type III, type I by itself.
enum class RealTransformType { DCT1, DCT2, DST1, DST2 };
enum class UniformGrid { Endpoints, Interior, CellCentered };
/// Both choices represent -d²/dx²; they are different discrete operators.
enum class LaplacianSymbol { Spectral, CenteredDifference };

/// Explicit bounded-axis contract. n counts unknowns, not cells or endpoints.
struct BoundedAxis {
  RealTransformType type;
  UniformGrid grid;
  int n;
  double length;
  double origin = 0.0;
};

/// Reject inconsistent type/grid, invalid count, or nonfinite geometry.
void validate(const BoundedAxis &axis);
[[nodiscard]] bool is_cosine(RealTransformType type);
[[nodiscard]] double spacing(const BoundedAxis &axis);
[[nodiscard]] double coordinate(const BoundedAxis &axis, int sample);
[[nodiscard]] int mirrored_size(const BoundedAxis &axis);
[[nodiscard]] double wavenumber(const BoundedAxis &axis, int mode);
/// Positive eigenvalue of -d²/dx². Modes are zero-based transform indices.
[[nodiscard]] double eigenvalue(const BoundedAxis &axis, int mode,
                                LaplacianSymbol symbol);

/** @brief Reusable rank-local double-precision FFTW plan.
 * Requires OpenPFC_HAS_FFTW3. Owns scratch and plans; preserves both inputs.
 * Planning/destruction follow FFTW's serial planner contract; execution on
 * different instances is independent. One instance is not reentrant.
 * Copying is disabled. An instance may be moved.
 */
class RealTransform {
public:
  explicit RealTransform(BoundedAxis axis);
  ~RealTransform();
  RealTransform(RealTransform &&) noexcept;
  RealTransform &operator=(RealTransform &&) noexcept;
  RealTransform(const RealTransform &) = delete;
  RealTransform &operator=(const RealTransform &) = delete;

  [[nodiscard]] const BoundedAxis &axis() const;
  [[nodiscard]] std::size_t buffer_bytes() const;
  void forward(std::span<const double> values, std::span<double> coefficients);
  void backward(std::span<const double> coefficients, std::span<double> values);
  /// Apply Delta with the selected symbol.
  void laplacian(std::span<const double> values, std::span<double> result,
                 LaplacianSymbol symbol = LaplacianSymbol::Spectral);
  /// Solve -Delta u=f, reject incompatible cosine forcing; zero weighted mean.
  void poisson(std::span<const double> forcing, std::span<double> solution,
               LaplacianSymbol symbol = LaplacianSymbol::Spectral,
               double compatibility_tolerance = 1e-12);
  /// Evaluate the interpolant or its derivative at any physical coordinate.
  /// O(n) per point, for boundary traces and arbitrary-point sampling.
  [[nodiscard]] double evaluate(std::span<const double> coefficients, double x,
                                int derivative_order = 0) const;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace pfc::fft
