// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file matrix_phi_2x2.hpp
 * @brief Exponential and φ-functions of a real 2 × 2 matrix.
 *
 * @details
 * For a real matrix `M` this evaluates
 *
 *   φ0(M) = exp(M),  φ1(M) = Σ M^n/(n+1)!,  φ2(M) = Σ M^n/(n+2)!,
 *
 * the matrix functions behind second-order exponential time differencing
 * on a field pair whose Fourier modes each carry a 2 × 2 linear symbol.
 *
 * Every function of a 2 × 2 matrix is affine in the matrix. With
 * `μ = tr(M)/2`, `E = M − μI` and `q = δ² = ((m00 − m11)/2)² + m01 m10`
 * (Cayley–Hamilton gives `E² = q I`),
 *
 *   f(M) = c0 I + c1 E,
 *   c0 = (f(μ+δ) + f(μ−δ)) / 2,   c1 = (f(μ+δ) − f(μ−δ)) / (2δ).
 *
 * `c0` and `c1` are even in `δ`, hence real for real `M`; a complex pair
 * (`q < 0`) uses one complex evaluation. This closed form costs one
 * exponential per mode.
 *
 * The divided difference `c1` cancels when the eigenvalues nearly coincide.
 * Its relative error is about `ε |μ| / |δ|` for stiff `μ`, so when
 * `|δ| ≤ 10⁻² max(1, |μ|)` the matrix is evaluated instead by scaling and
 * squaring: a Taylor series of `M / 2^s` with `‖M / 2^s‖₁ ≤ 1/2`, then `s`
 * doublings
 *
 *   φ0(2Z) = φ0(Z)²,
 *   φ1(2Z) = (φ0(Z) φ1(Z) + φ1(Z)) / 2,
 *   φ2(2Z) = (φ0(Z) φ2(Z) + φ1(Z) + φ2(Z)) / 4.
 *
 * That path has no eigenvalue division and covers the zero matrix,
 * defective (Jordan) matrices and repeated eigenvalues.
 *
 * Host-only scalar code. Inputs with a non-finite entry return non-finite
 * results; callers check finiteness where it matters.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>

namespace pfc::integrator {

/// Real 2 × 2 matrix, row-major: `[[m00, m01], [m10, m11]]`.
struct Matrix2x2 {
  double m00{0.0};
  double m01{0.0};
  double m10{0.0};
  double m11{0.0};
};

[[nodiscard]] constexpr Matrix2x2 operator*(const Matrix2x2 &a,
                                            const Matrix2x2 &b) noexcept {
  return {a.m00 * b.m00 + a.m01 * b.m10, a.m00 * b.m01 + a.m01 * b.m11,
          a.m10 * b.m00 + a.m11 * b.m10, a.m10 * b.m01 + a.m11 * b.m11};
}

[[nodiscard]] constexpr Matrix2x2 operator+(const Matrix2x2 &a,
                                            const Matrix2x2 &b) noexcept {
  return {a.m00 + b.m00, a.m01 + b.m01, a.m10 + b.m10, a.m11 + b.m11};
}

[[nodiscard]] constexpr Matrix2x2 operator*(double s, const Matrix2x2 &a) noexcept {
  return {s * a.m00, s * a.m01, s * a.m10, s * a.m11};
}

/// φ0, φ1 and φ2 of one matrix.
struct MatrixPhi2x2 {
  Matrix2x2 phi0; ///< exp(M)
  Matrix2x2 phi1; ///< φ1(M)
  Matrix2x2 phi2; ///< φ2(M)
};

namespace detail {

/// Scalar φ0, φ1, φ2 at real or complex `z`. Taylor for `|z| < 1/2`.
template <class T> struct ScalarPhi {
  T phi0;
  T phi1;
  T phi2;
};

template <class T> [[nodiscard]] inline ScalarPhi<T> scalar_phi(T z) {
  ScalarPhi<T> out{};
  out.phi0 = std::exp(z);
  if (std::abs(z) < 0.5) {
    // φ1 = Σ z^n/(n+1)!, φ2 = Σ z^n/(n+2)!; 0.5^18/18! is far below 1e-16.
    T term1 = T(1.0); // z^n/(n+1)!
    T term2 = T(0.5); // z^n/(n+2)!
    T s1 = term1;
    T s2 = term2;
    for (int n = 1; n <= 18; ++n) {
      term1 *= z / static_cast<double>(n + 1);
      term2 *= z / static_cast<double>(n + 2);
      s1 += term1;
      s2 += term2;
    }
    out.phi1 = s1;
    out.phi2 = s2;
    return out;
  }
  if constexpr (std::is_floating_point_v<T>) {
    out.phi1 = std::expm1(z) / z;
  } else {
    out.phi1 = (out.phi0 - T(1.0)) / z;
  }
  out.phi2 = (out.phi1 - T(1.0)) / z;
  return out;
}

[[nodiscard]] inline double norm1(const Matrix2x2 &m) noexcept {
  return std::max(std::abs(m.m00) + std::abs(m.m10),
                  std::abs(m.m01) + std::abs(m.m11));
}

[[nodiscard]] inline Matrix2x2 affine(double c0, double c1,
                                      const Matrix2x2 &e) noexcept {
  return {c0 + c1 * e.m00, c1 * e.m01, c1 * e.m10, c0 + c1 * e.m11};
}

} // namespace detail

/**
 * @brief φ0, φ1, φ2 by scaling and squaring; no eigenvalue division.
 *
 * Exact in exact arithmetic for every real 2 × 2 matrix, including the
 * zero matrix and defective ones. About ten times the cost of
 * @ref matrix_phi for stiff modes.
 */
[[nodiscard]] inline MatrixPhi2x2 matrix_phi_by_squaring(const Matrix2x2 &m) {
  const double nrm = detail::norm1(m);
  int s = 0;
  if (nrm > 0.5) {
    s = static_cast<int>(std::ceil(std::log2(nrm / 0.5)));
  }
  const Matrix2x2 x = std::ldexp(1.0, -s) * m;
  const Matrix2x2 id{1.0, 0.0, 0.0, 1.0};

  // Truncated Taylor sums of φ0, φ1, φ2 at x, ‖x‖₁ ≤ 1/2.
  Matrix2x2 power = id;
  Matrix2x2 p0 = id;
  Matrix2x2 p1 = id;
  Matrix2x2 p2 = 0.5 * id;
  double f0 = 1.0; // 1/n!
  double f1 = 1.0; // 1/(n+1)!
  double f2 = 0.5; // 1/(n+2)!
  for (int n = 1; n <= 20; ++n) {
    power = power * x;
    f0 /= static_cast<double>(n);
    f1 /= static_cast<double>(n + 1);
    f2 /= static_cast<double>(n + 2);
    p0 = p0 + f0 * power;
    p1 = p1 + f1 * power;
    p2 = p2 + f2 * power;
  }
  for (int i = 0; i < s; ++i) {
    const Matrix2x2 n2 = 0.25 * (p0 * p2 + p1 + p2);
    const Matrix2x2 n1 = 0.5 * (p0 * p1 + p1);
    p0 = p0 * p0;
    p1 = n1;
    p2 = n2;
  }
  return {p0, p1, p2};
}

/**
 * @brief φ0, φ1, φ2 of a real 2 × 2 matrix.
 *
 * Closed form through the eigenvalues when they are well separated,
 * @ref matrix_phi_by_squaring when `|δ| ≤ 10⁻² max(1, |μ|)`.
 */
[[nodiscard]] inline MatrixPhi2x2 matrix_phi(const Matrix2x2 &m) {
  const double mu = 0.5 * (m.m00 + m.m11);
  const double half_gap = 0.5 * (m.m00 - m.m11);
  const double q = half_gap * half_gap + m.m01 * m.m10;
  const double delta = std::sqrt(std::abs(q));
  if (!(delta > 1e-2 * std::max(1.0, std::abs(mu)))) {
    return matrix_phi_by_squaring(m);
  }
  const Matrix2x2 e{m.m00 - mu, m.m01, m.m10, m.m11 - mu};
  if (q > 0.0) {
    const auto plus = detail::scalar_phi(mu + delta);
    const auto minus = detail::scalar_phi(mu - delta);
    const double inv = 0.5 / delta;
    return {detail::affine(0.5 * (plus.phi0 + minus.phi0),
                           inv * (plus.phi0 - minus.phi0), e),
            detail::affine(0.5 * (plus.phi1 + minus.phi1),
                           inv * (plus.phi1 - minus.phi1), e),
            detail::affine(0.5 * (plus.phi2 + minus.phi2),
                           inv * (plus.phi2 - minus.phi2), e)};
  }
  // Complex pair μ ± iδ: f(μ−iδ) = conj f(μ+iδ).
  const auto z = detail::scalar_phi(std::complex<double>(mu, delta));
  const double inv = 1.0 / delta;
  return {detail::affine(z.phi0.real(), inv * z.phi0.imag(), e),
          detail::affine(z.phi1.real(), inv * z.phi1.imag(), e),
          detail::affine(z.phi2.real(), inv * z.phi2.imag(), e)};
}

/// `exp(hL)`, `h φ1(hL)` and `h φ2(hL)` for one ETD2 step of size `h`.
struct ETD2Coefficients2x2 {
  Matrix2x2 exp_hL;
  Matrix2x2 h_phi1;
  Matrix2x2 h_phi2;
};

/**
 * @brief Coefficients of one second-order exponential Runge–Kutta step.
 *
 * `A = exp_hL U + h_phi1 N(U)`, `U⁺ = A + h_phi2 (N(A) − N(U))`.
 */
[[nodiscard]] inline ETD2Coefficients2x2 etd2_coefficients(const Matrix2x2 &L,
                                                           double h) {
  const MatrixPhi2x2 f = matrix_phi(h * L);
  return {f.phi0, h * f.phi1, h * f.phi2};
}

} // namespace pfc::integrator
