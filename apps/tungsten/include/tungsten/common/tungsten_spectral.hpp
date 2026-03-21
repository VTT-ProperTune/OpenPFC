// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_spectral.hpp
 * @brief Shared k-space operator construction for CPU and GPU tungsten models
 */

#ifndef TUNGSTEN_SPECTRAL_HPP
#define TUNGSTEN_SPECTRAL_HPP

#include <cmath>
#include <tungsten/common/tungsten_params.hpp>

namespace tungsten {
namespace spectral {

/** Mean-field filter and exponential time-stepping weights for one Fourier mode. */
struct ModeOperators {
  double filterMF;
  double opL;
  double opN;
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
 * @param k_laplacian Laplacian symbol in k-space (negative squared wavenumber).
 * @param dt Time step length.
 * @param p Precomputed parameter scalars from make_operator_params().
 */
inline ModeOperators operators_for_mode(double k_laplacian, double dt,
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

  ModeOperators out{};
  out.filterMF = fMF;
  out.opL = std::exp(k_laplacian * opCk * dt);
  out.opN = (opCk == 0.0) ? k_laplacian * dt : (out.opL - 1.0) / opCk;
  return out;
}

} // namespace spectral
} // namespace tungsten

#endif
