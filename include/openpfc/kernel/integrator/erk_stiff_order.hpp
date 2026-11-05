// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file erk_stiff_order.hpp
 * @brief Audit of ERK43ZB against the Hochbruck–Ostermann stiff-order conditions.
 *
 * @details
 * Zoto & Bowman, arXiv:2303.12139v2 (26 Mar 2023), give the five-stage pair
 * ERK43ZB in Table 10. The coefficients are a preprint. This header evaluates
 * those weights, and Krogstad's ERK4K from their Table 3, as φ-combinations
 * and returns the Frobenius residual of each stiff-order condition in
 * Hochbruck & Ostermann, SIAM J. Numer. Anal. 43 (2005), as restated in
 * Zoto & Bowman Table 4.
 *
 * The pair is not a stepper. Nothing here advances a field, and
 * `SpectralETDSystem` is unchanged. Step doubling (#209) stays the baseline.
 *
 * φ_0(Z) = exp(Z) and φ_{k+1}(Z) = (φ_k(Z) − I/k!) Z^{-1}, with
 * φ_k(0) = I/k!. The argument `Z` is the paper's −hL. Weights are matrix
 * functions of `Z`, so a Jacobian `J` that does not commute with `Z` is
 * visible in conditions 5, 7, 8 and 9. Strong order-four conditions 6–9
 * are stricter than the *weak* fixed-step conditions in Hochbruck–Ostermann
 * Theorem 4.7: ψ4(0)=0, and conditions 7–9 use b_i(0) only on the
 * outermost stage factor while all internal stages remain at Z != 0.
 * Testing every coefficient at Z=0 is NOT this weak-form test.
 * This reference does not prove uniform order for variable step sizes.
 * Condition 8 follows Hochbruck &
 * Ostermann: the inner factor is ψ_{2,j}. Table 4 of the preprint writes
 * ψ_{2,i} on that factor; `condition8_repeated_index` records that typesetting.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <openpfc/kernel/integrator/matrix_phi_2x2.hpp>

namespace pfc::integrator {

/// φ_k of a real 2 × 2 matrix, for k = 0, 1, 2, 3, 4.
///
/// Taylor series on a scaled argument, then the doubling
/// φ_k(2X) = 2^{−k} (φ_0(X) φ_k(X) + ∑_{j=1}^{k} φ_j(X) / (k−j)!).
/// The same doubling for k = 1 and k = 2 is the one used by
/// @ref matrix_phi_by_squaring. Host only.
[[nodiscard]] inline Matrix2x2 phi_function(const Matrix2x2 &argument, int order) {
  if (order < 0 || order > 4) {
    throw std::invalid_argument("phi_function order must be from 0 through 4");
  }
  const double nrm = std::max(std::abs(argument.m00) + std::abs(argument.m10),
                              std::abs(argument.m01) + std::abs(argument.m11));
  if (!std::isfinite(nrm)) {
    throw std::invalid_argument("phi_function argument must have finite norm");
  }
  int scales = 0;
  if (nrm > 0.5) {
    const double required = std::ceil(std::log2(nrm) + 1.0);
    if (required > 1023.0) {
      throw std::out_of_range("phi_function argument requires excessive scaling");
    }
    scales = static_cast<int>(required);
  }
  const Matrix2x2 small = std::ldexp(1.0, -scales) * argument;
  const Matrix2x2 id{1.0, 0.0, 0.0, 1.0};

  Matrix2x2 series[5]{};
  Matrix2x2 power = id;
  double coeff[5]{};
  for (int k = 0; k <= order; ++k) {
    coeff[k] = 1.0;
    for (int i = 1; i <= k; ++i) {
      coeff[k] /= static_cast<double>(i);
    }
    series[k] = coeff[k] * id;
  }
  for (int n = 1; n <= 24; ++n) {
    power = power * small;
    for (int k = 0; k <= order; ++k) {
      coeff[k] /= static_cast<double>(k + n);
      series[k] = series[k] + coeff[k] * power;
    }
  }
  for (int step = 0; step < scales; ++step) {
    Matrix2x2 doubled[5]{};
    for (int k = 0; k <= order; ++k) {
      Matrix2x2 acc = series[0] * series[k];
      for (int j = 1; j <= k; ++j) {
        double inv_fact = 1.0;
        for (int i = 1; i <= k - j; ++i) {
          inv_fact /= static_cast<double>(i);
        }
        acc = acc + inv_fact * series[j];
      }
      doubled[k] = std::ldexp(1.0, -k) * acc;
    }
    for (int k = 0; k <= order; ++k) {
      series[k] = doubled[k];
    }
  }
  return series[order];
}

/// Which transcribed tableau to evaluate. The ERK43ZB rows share internal
/// stages; the order-4 result and the embedded estimate use different output
/// weights. ERK4K is the Krogstad method, which misses stiff conditions 7 and 8.
enum class ExponentialRkMethod {
  erk43zb_order4,
  erk43zb_embedded,
  erk4k,
};

/// Internal stages and output weights of one method at one matrix argument.
/// `weight[i][k]` is a_{ik} for k < i. `output[k]` is the quadrature weight
/// on stage k. Stage 0 is the node c = 0.
struct ExponentialRkEvaluation {
  int stage_count{0};
  double node[6]{};
  Matrix2x2 weight[6][6]{};
  Matrix2x2 output[6]{};
};

/// Frobenius norm of each Hochbruck–Ostermann condition, numbered 1..9
/// in `condition[0]` .. `condition[8]`. Condition 3 is the maximum over
/// internal stages. A zero residual is a numerical zero, not a stored flag.
struct StiffOrderResiduals {
  double condition[9]{};
  double condition8_repeated_index{0.0};
};

namespace erk_detail {

struct PhiTerm {
  int order;
  double node;
  double coefficient;
};

struct PhiWeight {
  static constexpr int capacity = 40;
  PhiTerm term[capacity]{};
  int count{0};

  PhiWeight &add(int order, double node, double coefficient) {
    if (count >= capacity) {
      throw std::logic_error("ERK phi-weight exceeded its term capacity");
    }
    term[count++] = PhiTerm{order, node, coefficient};
    return *this;
  }
};

[[nodiscard]] inline PhiWeight operator+(PhiWeight left, const PhiWeight &right) {
  for (int i = 0; i < right.count; ++i) {
    left.add(right.term[i].order, right.term[i].node, right.term[i].coefficient);
  }
  return left;
}

[[nodiscard]] inline PhiWeight operator-(PhiWeight left, const PhiWeight &right) {
  for (int i = 0; i < right.count; ++i) {
    left.add(right.term[i].order, right.term[i].node, -right.term[i].coefficient);
  }
  return left;
}

[[nodiscard]] inline PhiWeight operator*(double scale, PhiWeight weight) {
  for (int i = 0; i < weight.count; ++i) {
    weight.term[i].coefficient *= scale;
  }
  return weight;
}

[[nodiscard]] inline Matrix2x2 evaluate_weight(const PhiWeight &weight,
                                               const Matrix2x2 &argument) {
  Matrix2x2 acc{};
  for (int i = 0; i < weight.count; ++i) {
    const Matrix2x2 scaled = weight.term[i].node * argument;
    acc = acc +
          weight.term[i].coefficient * phi_function(scaled, weight.term[i].order);
  }
  return acc;
}

struct SymbolicStages {
  double node[6]{};
  PhiWeight weight[6][6]{};
  PhiWeight order4[6]{};
  PhiWeight embedded[6]{};
  int stage_count{0};
};

/// Table 10 of arXiv:2303.12139v2. Unsubscripted φ_k means φ_k(Z).
[[nodiscard]] inline SymbolicStages erk43zb_symbols() {
  SymbolicStages table{};
  table.stage_count = 5;
  table.node[0] = 0.0;
  table.node[1] = 1.0 / 6.0;
  table.node[2] = 0.5;
  table.node[3] = 0.5;
  table.node[4] = 1.0;

  PhiWeight a11;
  a11.add(2, 0.5, 1.5).add(2, 1.0 / 6.0, 0.5);

  PhiWeight a21;
  a21.add(1, 1.0, 19.0 / 60.0)
      .add(1, 0.5, 0.5)
      .add(1, 1.0 / 6.0, 0.5)
      .add(2, 0.5, 2.0)
      .add(2, 1.0 / 6.0, 13.0 / 6.0)
      .add(3, 0.5, 0.6);

  PhiWeight a22;
  a22.add(1, 1.0, -19.0 / 180.0)
      .add(1, 0.5, -1.0 / 6.0)
      .add(1, 1.0 / 6.0, -1.0 / 6.0)
      .add(2, 0.5, -1.0 / 6.0)
      .add(2, 1.0 / 6.0, 1.0 / 9.0)
      .add(3, 0.5, -0.2);

  PhiWeight a33;
  a33.add(2, 1.0, 1.0).add(2, 0.5, 1.0).add(3, 1.0, -6.0).add(3, 0.5, -3.0);

  const PhiWeight a31 =
      PhiWeight{}.add(2, 1.0, 3.0).add(2, 0.5, -4.5).add(2, 1.0 / 6.0, -2.5) +
      6.0 * a33 + a21;
  const PhiWeight a32 =
      PhiWeight{}.add(3, 1.0, 6.0).add(3, 0.5, 3.0) - 2.0 * a33 + a22;

  PhiWeight half_phi1;
  half_phi1.add(1, 0.5, 0.5);
  PhiWeight phi1;
  phi1.add(1, 1.0, 1.0);

  table.weight[1][0].add(1, 1.0 / 6.0, 1.0 / 6.0);
  table.weight[2][0] = half_phi1 - a11;
  table.weight[2][1] = a11;
  table.weight[3][0] = half_phi1 - a21 - a22;
  table.weight[3][1] = a21;
  table.weight[3][2] = a22;
  table.weight[4][0] = phi1 - a31 - a32 - a33;
  table.weight[4][1] = a31;
  table.weight[4][2] = a32;
  table.weight[4][3] = a33;

  for (int k = 0; k < 4; ++k) {
    table.embedded[k] = table.weight[4][k];
  }
  table.order4[0] =
      phi1 + PhiWeight{}.add(2, 1.0, -67.0 / 9.0).add(3, 1.0, 52.0 / 3.0);
  table.order4[1] = PhiWeight{}.add(2, 1.0, 8.0).add(3, 1.0, -24.0);
  table.order4[2] = PhiWeight{}.add(3, 1.0, 26.0 / 3.0).add(2, 1.0, -11.0 / 9.0);
  table.order4[3] = PhiWeight{}.add(2, 1.0, 7.0 / 9.0).add(3, 1.0, -10.0 / 3.0);
  table.order4[4] = PhiWeight{}.add(3, 1.0, 4.0 / 3.0).add(2, 1.0, -1.0 / 9.0);
  return table;
}

/// Table 3 of arXiv:2303.12139v2. Four stages, classical limit RK4.
[[nodiscard]] inline SymbolicStages erk4k_symbols() {
  SymbolicStages table{};
  table.stage_count = 4;
  table.node[0] = 0.0;
  table.node[1] = 0.5;
  table.node[2] = 0.5;
  table.node[3] = 1.0;
  table.weight[1][0].add(1, 0.5, 0.5);
  table.weight[2][0].add(1, 0.5, 0.5).add(2, 0.5, -1.0);
  table.weight[2][1].add(2, 0.5, 1.0);
  table.weight[3][0].add(1, 1.0, 1.0).add(2, 1.0, -2.0);
  table.weight[3][2].add(2, 1.0, 2.0);
  table.order4[0].add(1, 1.0, 1.0).add(2, 1.0, -3.0).add(3, 1.0, 4.0);
  table.order4[1].add(2, 1.0, 2.0).add(3, 1.0, -4.0);
  table.order4[2].add(2, 1.0, 2.0).add(3, 1.0, -4.0);
  table.order4[3].add(3, 1.0, 4.0).add(2, 1.0, -1.0);
  return table;
}

[[nodiscard]] inline double frobenius(const Matrix2x2 &matrix) noexcept {
  return std::sqrt(matrix.m00 * matrix.m00 + matrix.m01 * matrix.m01 +
                   matrix.m10 * matrix.m10 + matrix.m11 * matrix.m11);
}

[[nodiscard]] inline double power(double base, int exponent) noexcept {
  double value = 1.0;
  for (int i = 0; i < exponent; ++i) {
    value *= base;
  }
  return value;
}

[[nodiscard]] inline double inverse_factorial(int n) noexcept {
  double value = 1.0;
  for (int i = 1; i <= n; ++i) {
    value /= static_cast<double>(i);
  }
  return value;
}

} // namespace erk_detail

/// Evaluate one transcribed tableau at `argument` (the paper's −hL).
[[nodiscard]] inline ExponentialRkEvaluation
evaluate_exponential_rk(ExponentialRkMethod method, const Matrix2x2 &argument) {
  using erk_detail::evaluate_weight;
  const erk_detail::SymbolicStages symbols = [&] {
    if (method == ExponentialRkMethod::erk4k) {
      return erk_detail::erk4k_symbols();
    }
    return erk_detail::erk43zb_symbols();
  }();

  ExponentialRkEvaluation out{};
  const bool embedded = method == ExponentialRkMethod::erk43zb_embedded;
  out.stage_count = embedded ? 4 : symbols.stage_count;
  if (out.stage_count < 1 || out.stage_count > 6) {
    throw std::logic_error("ERK tableau stage count is outside 1..6");
  }
  for (int stage = 0; stage < out.stage_count; ++stage) {
    out.node[stage] = symbols.node[stage];
    for (int prev = 0; prev < stage; ++prev) {
      out.weight[stage][prev] =
          evaluate_weight(symbols.weight[stage][prev], argument);
    }
  }
  for (int stage = 0; stage < out.stage_count; ++stage) {
    const erk_detail::PhiWeight &row =
        embedded ? symbols.embedded[stage] : symbols.order4[stage];
    out.output[stage] = evaluate_weight(row, argument);
  }
  return out;
}

/// Residuals of stiff-order conditions 1–9 at one already evaluated tableau.
/// `argument` must be the matrix passed to @ref evaluate_exponential_rk.
/// `jacobian` and `mixed_derivative` are the paper's J and K. They are not
/// assumed to commute with the φ-weights.
[[nodiscard]] inline StiffOrderResiduals
hochbruck_ostermann_residuals(const ExponentialRkEvaluation &method,
                              const Matrix2x2 &argument, const Matrix2x2 &jacobian,
                              const Matrix2x2 &mixed_derivative) {
  if (method.stage_count < 1 || method.stage_count > 6) {
    throw std::invalid_argument(
        "ERK evaluation stage count must be from 1 through 6");
  }
  const int stages = method.stage_count;
  Matrix2x2 psi_stage[5][6]{};
  for (int stage = 0; stage < stages; ++stage) {
    for (int order = 1; order <= 4; ++order) {
      Matrix2x2 value = erk_detail::power(method.node[stage], order) *
                        phi_function(method.node[stage] * argument, order);
      const double denom = erk_detail::inverse_factorial(order - 1);
      for (int prev = 0; prev < stage; ++prev) {
        const double scale = erk_detail::power(method.node[prev], order - 1) * denom;
        value = value + (-scale) * method.weight[stage][prev];
      }
      psi_stage[order][stage] = value;
    }
  }
  Matrix2x2 psi[5]{};
  for (int order = 1; order <= 4; ++order) {
    Matrix2x2 value = phi_function(argument, order);
    const double denom = erk_detail::inverse_factorial(order - 1);
    for (int stage = 0; stage < stages; ++stage) {
      const double scale = erk_detail::power(method.node[stage], order - 1) * denom;
      value = value + (-scale) * method.output[stage];
    }
    psi[order] = value;
  }

  Matrix2x2 condition5{};
  Matrix2x2 condition7{};
  Matrix2x2 condition8{};
  Matrix2x2 condition9{};
  Matrix2x2 repeated{};
  for (int stage = 1; stage < stages; ++stage) {
    const Matrix2x2 &output = method.output[stage];
    condition5 = condition5 + output * (jacobian * psi_stage[2][stage]);
    condition7 = condition7 + output * (jacobian * psi_stage[3][stage]);
    condition9 = condition9 + (method.node[stage] * output) *
                                  (mixed_derivative * psi_stage[2][stage]);

    Matrix2x2 inner{};
    Matrix2x2 inner_weights{};
    for (int prev = 1; prev < stage; ++prev) {
      inner = inner + method.weight[stage][prev] * (jacobian * psi_stage[2][prev]);
      inner_weights = inner_weights + method.weight[stage][prev];
    }
    condition8 = condition8 + output * (jacobian * inner);
    repeated =
        repeated +
        output * (jacobian * (inner_weights * (jacobian * psi_stage[2][stage])));
  }

  StiffOrderResiduals out{};
  out.condition[0] = erk_detail::frobenius(psi[1]);
  out.condition[1] = erk_detail::frobenius(psi[2]);
  double stage_gap = 0.0;
  for (int stage = 1; stage < stages; ++stage) {
    stage_gap = std::max(stage_gap, erk_detail::frobenius(psi_stage[1][stage]));
  }
  out.condition[2] = stage_gap;
  out.condition[3] = erk_detail::frobenius(psi[3]);
  out.condition[4] = erk_detail::frobenius(condition5);
  out.condition[5] = erk_detail::frobenius(psi[4]);
  out.condition[6] = erk_detail::frobenius(condition7);
  out.condition[7] = erk_detail::frobenius(condition8);
  out.condition[8] = erk_detail::frobenius(condition9);
  out.condition8_repeated_index = erk_detail::frobenius(repeated);
  return out;
}

/** Residuals for the *weak*, fixed-step fourth-order sufficient conditions.
 *
 * Hochbruck–Ostermann (2005), Theorem 4.7: evaluate psi4 at Z=0
 * and conditions 7–9 with ONLY the outer b_i(Z) replaced by b_i(0).
 * The internal a_ij(Z), psi_{p,i}(Z), J and K retain their nonzero argument.
 * Strong conditions 1–5 must still be checked separately at nonzero Z.
 * These numerical probes cannot by themselves prove an identity for all
 * operators or an adaptive variable-step order guarantee.
 */
struct WeakOrder4Residuals {
  double psi4_at_zero{};
  double condition7{};
  double condition8{};
  double condition9{};
};

[[nodiscard]] inline WeakOrder4Residuals hochbruck_ostermann_weak_order4_residuals(
    const ExponentialRkEvaluation &at_argument,
    const ExponentialRkEvaluation &at_zero, const Matrix2x2 &argument,
    const Matrix2x2 &jacobian, const Matrix2x2 &mixed_derivative) {
  if (at_argument.stage_count != at_zero.stage_count ||
      at_argument.stage_count < 1 || at_argument.stage_count > 6) {
    throw std::invalid_argument("weak ERK order: inconsistent stage counts");
  }
  // psi_{p,i}(Z) is calculated from the original tableau, including a_ij(Z).
  // Changing the *outer* output weights alone is precisely the weak condition.
  ExponentialRkEvaluation weak = at_argument;
  for (int i = 0; i < weak.stage_count; ++i) {
    weak.output[i] = at_zero.output[i];
  }
  const auto at_z =
      hochbruck_ostermann_residuals(weak, argument, jacobian, mixed_derivative);
  const Matrix2x2 zero{};
  const auto at_origin =
      hochbruck_ostermann_residuals(at_zero, zero, jacobian, mixed_derivative);
  return {at_origin.condition[5], at_z.condition[6], at_z.condition[7],
          at_z.condition[8]};
}

} // namespace pfc::integrator
