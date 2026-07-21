// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_spectral.hpp
 * @brief Shared k-space physics construction for CPU and GPU tungsten models
 *
 * @details
 * Physics for one Fourier mode is @ref PhysicsMode (@c filterMF, @c opCk).
 * The diagonal linear symbol used by the shared exponential coefficient path is
 * @c L = k_laplacian * opCk (@ref linear_symbol).
 *
 * Historical ETD apply weights (@c opL / @c opN) remain available only via
 * @ref legacy_etd_weights_for_mode for equivalence tests. Production models must
 * build method weights through @c tungsten::etd::TungstenEtdWorkspace /
 * @c pfc::integrator::SpectralExpCoefficientCache.
 */

#ifndef TUNGSTEN_SPECTRAL_HPP
#define TUNGSTEN_SPECTRAL_HPP

#include <cmath>
#include <tungsten/common/tungsten_params.hpp>

namespace tungsten {
namespace spectral {

/**
 * @brief Physics quantities for one Fourier mode (not ETD method weights).
 *
 * @c filterMF is the mean-field filter @c χ(k). @c opCk is the Tungsten linear
 * peak/stabilization symbol such that the true Fourier multiplier is
 * @c L = k_laplacian * opCk.
 */
struct PhysicsMode {
  double filterMF{};
  double opCk{};
};

/**
 * @brief Legacy ETD apply weights for one mode (tests / equivalence only).
 *
 * @c opL = exp(L·dt), @c opN = expm1(L·dt)/opCk with @c L = k_laplacian*opCk.
 * Production models must not call @ref legacy_etd_weights_for_mode.
 */
struct ModeOperators {
  double filterMF{};
  double opL{};
  double opN{};
};

/** Scalar bundle read once per `prepare_operators` (avoid repeated getters in the
 * k-loop). */
struct OperatorParams {
  double alpha2;
  double lambda2;
  double alpha_farTol;
  int alpha_highOrd;
  double Bx;
  double T;
  double T0;
  double stabP;
  double p2_bar;
  double q2_bar;
};

inline OperatorParams make_operator_params(const TungstenParams &params) {
  double alpha = params.get_alpha();
  double lambda = params.get_lambda();
  return OperatorParams{2.0 * alpha * alpha,       2.0 * lambda * lambda,
                        params.get_alpha_farTol(), params.get_alpha_highOrd(),
                        params.get_Bx(),           params.get_T(),
                        params.get_T0(),           params.get_stabP(),
                        params.get_p2_bar(),       params.get_q2_bar()};
}

/**
 * @brief Build mean-field filter and linear peak symbol for one mode.
 *
 * @param k_laplacian Laplacian symbol in k-space (negative squared wavenumber).
 * @param p Precomputed parameter scalars from make_operator_params().
 */
[[nodiscard]] inline PhysicsMode physics_for_mode(double k_laplacian,
                                                  const OperatorParams &p) {
  double fMF = std::exp(k_laplacian / p.lambda2);

  double k_val = std::sqrt(-k_laplacian) - 1.0;
  double k2 = k_val * k_val;
  double rTol = -p.alpha2 * std::log(p.alpha_farTol) - 1.0;

  double g1 = 0.0;
  if (p.alpha_highOrd == 0) {
    g1 = std::exp(-k2 / p.alpha2);
  } else {
    g1 = std::exp(-(k2 + rTol * std::pow(k_val, p.alpha_highOrd)) / p.alpha2);
  }

  double g2 = 1.0 - 1.0 / p.alpha2 * k2;
  double gf = (k_val < 0.0) ? g1 : g2;
  double opPeak = p.Bx * std::exp(-p.T / p.T0) * gf;
  double opCk = p.stabP + p.p2_bar - opPeak + p.q2_bar * fMF;

  return PhysicsMode{.filterMF = fMF, .opCk = opCk};
}

/**
 * @brief Diagonal linear symbol for shared spectral-exp coefficients.
 *
 * @return @c k_laplacian * opCk (the true Fourier multiplier on @c ψ̂).
 */
[[nodiscard]] inline double linear_symbol(double k_laplacian,
                                          double opCk) noexcept {
  return k_laplacian * opCk;
}

/**
 * @brief Historical ETD weights for equivalence tests only.
 *
 * Production models must use @c TungstenEtdWorkspace instead.
 *
 * @param k_laplacian Laplacian symbol in k-space (negative squared wavenumber).
 * @param dt Time step length.
 * @param p Precomputed parameter scalars from make_operator_params().
 */
[[nodiscard]] inline ModeOperators
legacy_etd_weights_for_mode(double k_laplacian, double dt,
                            const OperatorParams &p) {
  const PhysicsMode phys = physics_for_mode(k_laplacian, p);

  ModeOperators out{};
  out.filterMF = phys.filterMF;

  constexpr double opCk_threshold = 1e-12;
  const double arg = k_laplacian * phys.opCk * dt;
  out.opL = std::exp(arg);
  if (std::abs(phys.opCk) < opCk_threshold) {
    // Taylor series: e^x - 1 ≈ x + x²/2 for small x, then divide by opCk
    // arg = k_laplacian * opCk * dt, so arg² = k_laplacian² * opCk² * dt²
    // Dividing (arg + arg²/2) by opCk gives:
    //   k_laplacian * dt + 0.5 * k_laplacian² * dt² * opCk
    out.opN = k_laplacian * dt +
              0.5 * k_laplacian * k_laplacian * dt * dt * phys.opCk;
  } else {
    out.opN = std::expm1(arg) / phys.opCk;
  }
  return out;
}

} // namespace spectral
} // namespace tungsten

#endif
