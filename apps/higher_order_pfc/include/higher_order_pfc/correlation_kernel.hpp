// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file correlation_kernel.hpp
 * @brief Polynomial-in-\f$k_{\mathrm{lap}}\f$ linear operators for PFC kernels.
 *
 * @details
 * Every PFC correlation kernel used here is a polynomial in the Laplacian
 * symbol \f$u=k_{\mathrm{lap}}=-|k|^2\f$. Writing it that way keeps the
 * eighth-order kernel and the tenth-order conserved operator as ordinary
 * Horner evaluations instead of hand-rolled \f$k^8\f$ / \f$k^{10}\f$ special
 * cases, which is the whole point of the example: **high differential order is
 * not high implementation complexity in a spectral code.**
 *
 * The helper is deliberately small and stays app-local. Promote it to the core
 * only if a second application genuinely needs it.
 */

#include <array>
#include <cstddef>

#include <openpfc/kernel/data/host_device.hpp>

namespace higher_order_pfc {

/**
 * @brief Degree-`N` polynomial in \f$u=k_{\mathrm{lap}}\f$, Horner-evaluated.
 *
 * `coeff[i]` multiplies \f$u^i\f$, so `coeff[0]` is the \f$k=0\f$ value. For
 * conserved dynamics that entry is exactly zero, which is what makes mass
 * conservation exact rather than approximate.
 */
template <std::size_t N> struct PolynomialInKLap {
  static constexpr std::size_t degree = N;

  std::array<double, N + 1> coeff{};

  [[nodiscard]] OPENPFC_HD double operator()(double u) const {
    double acc = coeff[N];
    for (std::size_t i = N; i-- > 0;) acc = acc * u + coeff[i];
    return acc;
  }
};

/// Free-energy kernel \f$\Lambda(u)\f$: quartic in \f$u\f$, so \f$k^8\f$.
using QuadraticKernel = PolynomialInKLap<4>;

/// Conserved evolution symbol \f$L(u)=M\,u\,\Lambda(u)\f$: \f$k^{10}\f$.
using EvolutionSymbol = PolynomialInKLap<5>;

/**
 * @brief Single-mode (classical) PFC kernel
 *        \f$\Lambda_1(u)=-\varepsilon+(1+u)^2\f$.
 *
 * Fourth order in \f$k\f$. One minimum, at \f$|k|=1\f$, depth
 * \f$-\varepsilon\f$. In 2D this selects a triangular lattice.
 */
[[nodiscard]] inline QuadraticKernel single_mode_kernel(double eps) {
  // (1+u)^2 = 1 + 2u + u^2
  return QuadraticKernel{{1.0 - eps, 2.0, 1.0, 0.0, 0.0}};
}

/**
 * @brief Two-mode PFC kernel
 *        \f$\Lambda_2(u)=-\varepsilon+(1+u)^2\bigl[r_1+(q_1^2+u)^2\bigr]\f$.
 *
 * Eighth order in \f$k\f$. Expanding with \f$a=q_1^2\f$, \f$A=r_1+a^2\f$,
 * \f$B=2a\f$, \f$C=1\f$:
 *
 * \f[
 *   \Lambda_2(u) = (A-\varepsilon) + (B+2A)u + (C+2B+A)u^2
 *                  + (2C+B)u^3 + C u^4 .
 * \f]
 *
 * With \f$r_1=0\f$ the two minima at \f$|k|=1\f$ and \f$|k|=q_1\f$ are exactly
 * degenerate at depth \f$-\varepsilon\f$; \f$r_1>0\f$ lifts the second one by
 * \f$(1-q_1^2)^2 r_1\f$. Two unstable bands is what lets the kernel select
 * square (2D, \f$q_1=\sqrt2\f$) or FCC (3D, \f$q_1=2/\sqrt3\f$) ordering
 * instead of the single-mode triangular/BCC result.
 */
[[nodiscard]] inline QuadraticKernel two_mode_kernel(double eps, double q1,
                                                     double r1) {
  const double a = q1 * q1;
  const double A = r1 + a * a;
  const double B = 2.0 * a;
  const double C = 1.0;
  return QuadraticKernel{
      {A - eps, B + 2.0 * A, C + 2.0 * B + A, 2.0 * C + B, C}};
}

/**
 * @brief Conserved PFC evolution symbol \f$L(u)=M\,u\,\Lambda(u)\f$.
 *
 * Multiplying by \f$u\f$ shifts every coefficient up one power, which is why
 * \f$L\f$ is degree five in \f$u\f$ (tenth order in \f$k\f$) and why
 * `coeff[0]` is identically zero.
 */
[[nodiscard]] inline EvolutionSymbol conserved_symbol(const QuadraticKernel &kernel,
                                                      double mobility) {
  EvolutionSymbol L{};
  L.coeff[0] = 0.0;
  for (std::size_t i = 0; i <= QuadraticKernel::degree; ++i)
    L.coeff[i + 1] = mobility * kernel.coeff[i];
  return L;
}

} // namespace higher_order_pfc
