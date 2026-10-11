// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file crystal_plasticity.hpp
 * @brief Finite-strain FCC slip on the local `F -> P` contract.
 *
 * The update is the slip-only phenomenological power law used by DAMASK
 * 3.0 (`phenopowerlaw`): multiplicative `F = Fe Fp`, twelve
 * `{111}<110>` systems, and one saturated hardening matrix. There is no
 * eigenstrain, twinning, or non-Schmid stress.
 *
 * `Q` maps sample-frame components to the crystal, `v_c = Q v_s`. It is
 * the passive matrix of a proper rotation (`P = -1`). `Q = I` aligns the
 * cubic axes with the sample axes. `Fp` is stored in the crystal frame
 * and starts at `I`. The sample stress is the first Piola–Kirchhoff
 * stress, `P_sample = Q^T P_crystal Q`.
 *
 * Slip systems, crystal frame, are direction then plane, then normalized.
 * System 1 is `(0, 1, -1)` on `(1, 1, 1)`. The Schmid tensor is `d ⊗ n`.
 * Resolved shear is `τ = S : (d ⊗ n)`, where `S` is the lattice second
 * Piola–Kirchhoff stress. With no eigenstrain the driving stress is `S`,
 * not `Ce S`. The signed rate is `γ̇ = sign(τ) γ̇0 (|τ|/ξ)^n`.
 * Coplanar systems, including self-hardening, share one interaction
 * weight. Every other pair uses the latent weight.
 *
 * Hardening of the slip resistance is
 * `ξ̇^α = h0 |1 - ξ^α/ξ∞|^a sign(1 - ξ^α/ξ∞) Σ_β h^{αβ} |γ̇^β|`.
 * Accumulated shear stores `∫ |γ̇| dt` and does not feed the stress.
 * `Δt` is a material parameter. The global solver does not pass time.
 *
 * The local unknowns are `(Lp, ξ)`. `Lp` is an implicit residual at
 * frozen resistance. The resistance is then a fixed point. A rejected
 * increment leaves the committed `(Fp, ξ, γ, Lp)` unchanged. The tangent
 * is `dP_ij / dF_kl` in the sample frame, including the hardening
 * sensitivity. `evaluate_fcc` returns that tensor with the trial state
 * from the same slip update.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <openpfc/mechanics/constitutive/history.hpp>
#include <openpfc/mechanics/constitutive/integration_count.hpp>
#include <openpfc/mechanics/piola_tangent.hpp>
#include <openpfc/mechanics/tensor.hpp>

namespace pfc::finite_strain {

/// Cubic stiffness and the slip-only power-law parameters.
struct FccSlipParameters {
  double c11 = 0.0;
  double c12 = 0.0;
  double c44 = 0.0;
  /// Reference shear rate `γ̇0`.
  double reference_rate = 0.0;
  /// Rate exponent `n`.
  double rate_exponent = 0.0;
  /// Hardening exponent `a`.
  double hardening_exponent = 0.0;
  /// Hardening coefficient `h0`.
  double hardening_coefficient = 0.0;
  double initial_resistance = 0.0;
  double saturation_resistance = 0.0;
  /// Load-step duration. The solver does not supply time.
  double time_step = 0.0;
  /// Interaction of two systems on the same `{111}` plane, including self.
  double coplanar = 1.0;
  /// Interaction of systems on different `{111}` planes.
  double latent = 1.4;
};

/// Crystal-frame plastic map, slip resistance, accumulated shear, and `Lp`.
struct FccSlipState {
  Tensor2 plastic{};
  Tensor2 velocity{};
  double resistance[12]{};
  double shear[12]{};
};

struct FccSlipUpdate {
  Tensor2 piola{};
  FccSlipState state{};
  double tau[12]{};
  double shear_rate[12]{};
};

[[nodiscard]] OPENPFC_INLINE_HD FccSlipState
fcc_identity_state(const FccSlipParameters &parameters) {
  FccSlipState state;
  state.plastic = identity2();
  state.velocity = Tensor2{};
  for (int system = 0; system < 12; ++system) {
    state.resistance[system] = parameters.initial_resistance;
    state.shear[system] = 0.0;
  }
  return state;
}

/// Failures returned by the shared host/device arithmetic. No history is committed.
enum class FccStatus {
  Success,
  Parameters,
  Orientation,
  SingularDeformation,
  RateRange,
  VelocityJacobian,
  VelocityLineSearch,
  VelocityIterations,
  HardeningBlock,
  Tangent,
  ResistanceIterations,
  PlasticOrientation
};

inline const char *fcc_status_message(FccStatus status) noexcept {
  switch (status) {
  case FccStatus::Success: return "success";
  case FccStatus::Parameters: return "FCC slip parameters are not admissible";
  case FccStatus::Orientation:
    return "crystal orientation must be a proper rotation";
  case FccStatus::SingularDeformation: return "crystal deformation is singular";
  case FccStatus::RateRange: return "FCC slip rate left the admissible range";
  case FccStatus::VelocityJacobian: return "FCC slip velocity Jacobian is singular";
  case FccStatus::VelocityLineSearch: return "FCC slip velocity line search failed";
  case FccStatus::VelocityIterations: return "FCC slip velocity did not converge";
  case FccStatus::HardeningBlock: return "FCC slip hardening block is singular";
  case FccStatus::Tangent: return "FCC slip tangent is singular";
  case FccStatus::ResistanceIterations:
    return "FCC slip resistance did not converge";
  case FccStatus::PlasticOrientation:
    return "FCC plastic map left the orientation-preserving range";
  }
  return "unknown FCC status";
}

namespace crystal {

inline constexpr int kSystems = 12;
inline constexpr int kLpIterations = 40;
inline constexpr int kStateIterations = 20;
/// Trial shear ratios above this are not a converged state.
inline constexpr double kRateCap = 4.0;

struct Tangent {
  double c[3][3][3][3]{};

  OPENPFC_HD double &operator()(int i, int j, int k, int l) noexcept {
    return c[i][j][k][l];
  }
  OPENPFC_HD double operator()(int i, int j, int k, int l) const noexcept {
    return c[i][j][k][l];
  }
};

struct Integrated {
  FccSlipUpdate update{};
  Tangent tangent{};
  FccStatus status = FccStatus::Success;
};

[[nodiscard]] OPENPFC_INLINE_HD double determinant(const Tensor2 &a) noexcept {
  return a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
         a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
         a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
}

[[nodiscard]] OPENPFC_INLINE_HD Tensor2 inverse(const Tensor2 &a,
                                                FccStatus &status) {
  const double det = determinant(a);
  if (!(std::abs(det) > 1e-30)) {
    status = FccStatus::SingularDeformation;
    return {};
  }
  const double inv = 1.0 / det;
  Tensor2 out;
  out(0, 0) = (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) * inv;
  out(0, 1) = (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * inv;
  out(0, 2) = (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * inv;
  out(1, 0) = (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) * inv;
  out(1, 1) = (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * inv;
  out(1, 2) = (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * inv;
  out(2, 0) = (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) * inv;
  out(2, 1) = (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * inv;
  out(2, 2) = (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * inv;
  return out;
}

[[nodiscard]] OPENPFC_INLINE_HD std::array<Tensor2, kSystems> schmid_tensors() {
  std::array<Tensor2, kSystems> systems = [] {
    constexpr int raw[kSystems][6] = {
        {0, 1, -1, 1, 1, 1},    {-1, 0, 1, 1, 1, 1},    {1, -1, 0, 1, 1, 1},
        {0, -1, -1, -1, -1, 1}, {1, 0, 1, -1, -1, 1},   {-1, 1, 0, -1, -1, 1},
        {0, -1, 1, 1, -1, -1},  {-1, 0, -1, 1, -1, -1}, {1, 1, 0, 1, -1, -1},
        {0, 1, 1, -1, 1, -1},   {1, 0, -1, -1, 1, -1},  {-1, -1, 0, -1, 1, -1},
    };
    std::array<Tensor2, kSystems> out{};
    for (int system = 0; system < kSystems; ++system) {
      double direction[3];
      double normal[3];
      double direction_norm = 0.0;
      double normal_norm = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        direction[axis] = static_cast<double>(raw[system][axis]);
        normal[axis] = static_cast<double>(raw[system][axis + 3]);
        direction_norm += direction[axis] * direction[axis];
        normal_norm += normal[axis] * normal[axis];
      }
      direction_norm = std::sqrt(direction_norm);
      normal_norm = std::sqrt(normal_norm);
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          out[system](row, column) =
              direction[row] / direction_norm * normal[column] / normal_norm;
        }
      }
    }
    return out;
  }();
  return systems;
}

[[nodiscard]] OPENPFC_INLINE_HD double
interaction(const FccSlipParameters &parameters, int alpha, int beta) noexcept {
  return alpha / 3 == beta / 3 ? parameters.coplanar : parameters.latent;
}

[[nodiscard]] OPENPFC_INLINE_HD bool
admissible_parameters(const FccSlipParameters &parameters) {
  const bool stable = parameters.c11 > 0.0 && parameters.c44 > 0.0 &&
                      parameters.c11 > std::abs(parameters.c12) &&
                      (parameters.c11 + 2.0 * parameters.c12) > 0.0;
  const bool flow =
      parameters.reference_rate > 0.0 && parameters.rate_exponent > 1.0 &&
      parameters.hardening_exponent > 1.0 &&
      parameters.hardening_coefficient > 0.0 &&
      parameters.initial_resistance > 0.0 &&
      parameters.saturation_resistance > parameters.initial_resistance &&
      parameters.time_step > 0.0 && parameters.coplanar >= 0.0 &&
      parameters.latent >= 0.0;
  return stable && flow;
}

inline void require_parameters(const FccSlipParameters &parameters) {
  if (!admissible_parameters(parameters))
    throw std::invalid_argument("FCC slip parameters are not admissible");
}

[[nodiscard]] OPENPFC_INLINE_HD bool
proper_rotation(const Tensor2 &orientation) noexcept {
  if (!(std::abs(determinant(orientation) - 1.0) < 1e-8)) {
    return false;
  }
  const Tensor2 product = matmul(transposed(orientation), orientation);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double expected = i == j ? 1.0 : 0.0;
      if (!(std::abs(product(i, j) - expected) < 1e-8)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] OPENPFC_INLINE_HD Tensor2
push_forward(const Tensor2 &orientation, const Tensor2 &sample) noexcept {
  return matmul(orientation, matmul(sample, transposed(orientation)));
}

[[nodiscard]] OPENPFC_INLINE_HD Tensor2
pull_stress(const Tensor2 &orientation, const Tensor2 &crystal) noexcept {
  return matmul(transposed(orientation), matmul(crystal, orientation));
}

[[nodiscard]] OPENPFC_INLINE_HD Tensor2
cubic_stress(const FccSlipParameters &parameters, const Tensor2 &strain) noexcept {
  Tensor2 stress;
  stress(0, 0) =
      parameters.c11 * strain(0, 0) + parameters.c12 * (strain(1, 1) + strain(2, 2));
  stress(1, 1) = parameters.c12 * strain(0, 0) + parameters.c11 * strain(1, 1) +
                 parameters.c12 * strain(2, 2);
  stress(2, 2) =
      parameters.c12 * (strain(0, 0) + strain(1, 1)) + parameters.c11 * strain(2, 2);
  stress(0, 1) = 2.0 * parameters.c44 * strain(0, 1);
  stress(1, 0) = stress(0, 1);
  stress(0, 2) = 2.0 * parameters.c44 * strain(0, 2);
  stress(2, 0) = stress(0, 2);
  stress(1, 2) = 2.0 * parameters.c44 * strain(1, 2);
  stress(2, 1) = stress(1, 2);
  return stress;
}

[[nodiscard]] OPENPFC_INLINE_HD Tensor2
green_lagrange(const Tensor2 &elastic) noexcept {
  Tensor2 metric = matmul(transposed(elastic), elastic);
  metric(0, 0) -= 1.0;
  metric(1, 1) -= 1.0;
  metric(2, 2) -= 1.0;
  return scaled(metric, 0.5);
}

struct Kinetics {
  double tau[kSystems]{};
  double rate[kSystems]{};
  double drate_dtau[kSystems]{};
  double dabs_dtau[kSystems]{};
  double drate_dresistance[kSystems]{};
  double dabs_dresistance[kSystems]{};
  bool finite = true;
};

[[nodiscard]] OPENPFC_INLINE_HD Kinetics
evaluate_kinetics(const FccSlipParameters &parameters, const Tensor2 &second,
                  const double resistance[kSystems]) noexcept {
  const auto &systems = schmid_tensors();
  Kinetics out;
  for (int alpha = 0; alpha < kSystems; ++alpha) {
    out.tau[alpha] = frobenius_dot(systems[alpha], second);
    if (!(resistance[alpha] > 0.0) || out.tau[alpha] == 0.0) {
      continue;
    }
    const double ratio = std::abs(out.tau[alpha]) / resistance[alpha];
    if (ratio > kRateCap) {
      out.rate[alpha] =
          std::copysign(std::numeric_limits<double>::infinity(), out.tau[alpha]);
      out.finite = false;
      continue;
    }
    out.rate[alpha] = std::copysign(parameters.reference_rate *
                                        std::pow(ratio, parameters.rate_exponent),
                                    out.tau[alpha]);
    const double absolute_tau = std::abs(out.tau[alpha]);
    out.drate_dtau[alpha] =
        parameters.rate_exponent * std::abs(out.rate[alpha]) / absolute_tau;
    out.dabs_dtau[alpha] = parameters.rate_exponent * out.rate[alpha] / absolute_tau;
    out.drate_dresistance[alpha] =
        -parameters.rate_exponent * out.rate[alpha] / resistance[alpha];
    out.dabs_dresistance[alpha] =
        -parameters.rate_exponent * std::abs(out.rate[alpha]) / resistance[alpha];
  }
  return out;
}

[[nodiscard]] OPENPFC_INLINE_HD Tensor2
velocity_from(const double rate[kSystems]) noexcept {
  const auto &systems = schmid_tensors();
  Tensor2 velocity{};
  for (int alpha = 0; alpha < kSystems; ++alpha) {
    velocity = axpy(rate[alpha], systems[alpha], velocity);
  }
  return velocity;
}

struct SlipSlope {
  double value[kSystems]{};
  double derivative[kSystems]{};
};

[[nodiscard]] OPENPFC_INLINE_HD SlipSlope
slip_slope(const FccSlipParameters &parameters,
           const double resistance[kSystems]) noexcept {
  SlipSlope slope;
  for (int alpha = 0; alpha < kSystems; ++alpha) {
    const double reduced =
        1.0 - resistance[alpha] / parameters.saturation_resistance;
    const double magnitude = std::abs(reduced);
    slope.value[alpha] =
        std::copysign(std::pow(magnitude, parameters.hardening_exponent), reduced);
    if (magnitude > 0.0) {
      slope.derivative[alpha] =
          parameters.hardening_exponent *
          std::pow(magnitude, parameters.hardening_exponent - 1.0) *
          (-1.0 / parameters.saturation_resistance);
    }
  }
  return slope;
}

[[nodiscard]] OPENPFC_INLINE_HD double
hardening_dot(const FccSlipParameters &parameters, const SlipSlope &slope, int alpha,
              const double absolute_rate[kSystems]) noexcept {
  double accumulated = 0.0;
  for (int beta = 0; beta < kSystems; ++beta) {
    accumulated += interaction(parameters, alpha, beta) * absolute_rate[beta];
  }
  return parameters.hardening_coefficient * slope.value[alpha] * accumulated;
}

struct LpEvaluation {
  Tensor2 velocity{};
  Tensor2 elastic{};
  Tensor2 second{};
  Tensor2 residual{};
  Kinetics kinetics{};
  double residual_norm = 0.0;
  double scale = 0.0;
};

[[nodiscard]] OPENPFC_INLINE_HD LpEvaluation
evaluate_velocity(const FccSlipParameters &parameters, const Tensor2 &mapper,
                  const Tensor2 &velocity, const double resistance[kSystems]) {
  LpEvaluation out;
  const Tensor2 intermediate = axpy(-parameters.time_step, velocity, identity2());
  out.elastic = matmul(mapper, intermediate);
  out.second = cubic_stress(parameters, green_lagrange(out.elastic));
  out.kinetics = evaluate_kinetics(parameters, out.second, resistance);
  const Tensor2 constitutive = velocity_from(out.kinetics.rate);
  out.residual = add(velocity, scaled(constitutive, -1.0));
  out.scale = std::max(frobenius_norm(velocity), frobenius_norm(constitutive));
  out.residual_norm = out.kinetics.finite ? frobenius_norm(out.residual)
                                          : std::numeric_limits<double>::infinity();
  return out;
}

[[nodiscard]] OPENPFC_INLINE_HD int pack_index(int row, int column) noexcept {
  return row * 3 + column;
}

template <int N> OPENPFC_HD bool solve_dense(double *matrix, double *rhs) {
  double scale = 0.0;
  for (int entry = 0; entry < N * N; ++entry) {
    scale = std::max(scale, std::abs(matrix[entry]));
  }
  const double floor = 1e-14 * std::max(scale, 1.0);
  for (int column = 0; column < N; ++column) {
    int pivot = column;
    double best = std::abs(matrix[column * N + column]);
    for (int row = column + 1; row < N; ++row) {
      const double value = std::abs(matrix[row * N + column]);
      if (value > best) {
        best = value;
        pivot = row;
      }
    }
    if (!(best > floor)) {
      return false;
    }
    if (pivot != column) {
      for (int entry = column; entry < N; ++entry) {
        std::swap(matrix[column * N + entry], matrix[pivot * N + entry]);
      }
      std::swap(rhs[column], rhs[pivot]);
    }
    const double diagonal = matrix[column * N + column];
    for (int row = column + 1; row < N; ++row) {
      const double factor = matrix[row * N + column] / diagonal;
      for (int entry = column + 1; entry < N; ++entry) {
        matrix[row * N + entry] -= factor * matrix[column * N + entry];
      }
      rhs[row] -= factor * rhs[column];
    }
  }
  for (int row = N - 1; row >= 0; --row) {
    double value = rhs[row];
    for (int column = row + 1; column < N; ++column) {
      value -= matrix[row * N + column] * rhs[column];
    }
    rhs[row] = value / matrix[row * N + row];
  }
  return true;
}

[[nodiscard]] OPENPFC_INLINE_HD Tensor2
unpack_velocity(const double packed[9]) noexcept {
  Tensor2 velocity;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      velocity(row, column) = packed[pack_index(row, column)];
    }
  }
  return velocity;
}

OPENPFC_INLINE_HD void pack_velocity(const Tensor2 &velocity,
                                     double packed[9]) noexcept {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      packed[pack_index(row, column)] = velocity(row, column);
    }
  }
}

[[nodiscard]] OPENPFC_INLINE_HD Tensor2 velocity_increment(
    const FccSlipParameters &parameters, const Tensor2 &mapper,
    const Tensor2 &elastic, const Kinetics &kinetics, int row, int column) noexcept {
  Tensor2 delastic{};
  for (int i = 0; i < 3; ++i) {
    delastic(i, column) = -parameters.time_step * mapper(i, row);
  }
  const Tensor2 dstrain = scaled(add(matmul(transposed(delastic), elastic),
                                     matmul(transposed(elastic), delastic)),
                                 0.5);
  const Tensor2 dsecond = cubic_stress(parameters, dstrain);
  const auto &systems = schmid_tensors();
  Tensor2 dvelocity{};
  for (int alpha = 0; alpha < kSystems; ++alpha) {
    const double dtau = frobenius_dot(systems[alpha], dsecond);
    dvelocity = axpy(kinetics.drate_dtau[alpha] * dtau, systems[alpha], dvelocity);
  }
  return dvelocity;
}

[[nodiscard]] OPENPFC_INLINE_HD LpEvaluation solve_velocity(
    const FccSlipParameters &parameters, const Tensor2 &mapper, Tensor2 velocity,
    const double resistance[kSystems], FccStatus &status) {
  for (int iteration = 0; iteration < kLpIterations; ++iteration) {
    const LpEvaluation current =
        evaluate_velocity(parameters, mapper, velocity, resistance);
    const double tolerance = std::max(1e-10 * current.scale, 1e-12);
    if (current.kinetics.finite && current.residual_norm <= tolerance) {
      LpEvaluation accepted = current;
      accepted.velocity = velocity;
      return accepted;
    }
    if (!current.kinetics.finite) {
      status = FccStatus::RateRange;
      return {};
    }

    double jacobian[81];
    double rhs[9];
    for (int entry = 0; entry < 81; ++entry) {
      jacobian[entry] = 0.0;
    }
    for (int index = 0; index < 9; ++index) {
      jacobian[index * 9 + index] = 1.0;
    }
    pack_velocity(current.residual, rhs);
    for (int unknown = 0; unknown < 9; ++unknown) {
      rhs[unknown] = -rhs[unknown];
    }
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        const int unknown = pack_index(row, column);
        const Tensor2 dvelocity = velocity_increment(
            parameters, mapper, current.elastic, current.kinetics, row, column);
        for (int equation = 0; equation < 3; ++equation) {
          for (int slot = 0; slot < 3; ++slot) {
            jacobian[pack_index(equation, slot) * 9 + unknown] -=
                dvelocity(equation, slot);
          }
        }
      }
    }
    if (!solve_dense<9>(jacobian, rhs)) {
      status = FccStatus::VelocityJacobian;
      return {};
    }
    const Tensor2 step = unpack_velocity(rhs);
    double lambda = 1.0;
    bool accepted = false;
    while (lambda > 1e-8) {
      const Tensor2 trial = axpy(lambda, step, velocity);
      const LpEvaluation candidate =
          evaluate_velocity(parameters, mapper, trial, resistance);
      if (candidate.residual_norm < current.residual_norm) {
        velocity = trial;
        accepted = true;
        break;
      }
      lambda *= 0.5;
    }
    if (!accepted) {
      status = FccStatus::VelocityLineSearch;
      return {};
    }
  }
  status = FccStatus::VelocityIterations;
  return {};
}

struct TangentFactors {
  Tensor2 mapper{};
  Tensor2 intermediate{};
  Tensor2 elastic{};
  Tensor2 crystal_deformation{};
  Tensor2 plastic_inverse{};
  Tensor2 second{};
  Tensor2 growth{};
  Kinetics kinetics{};
  SlipSlope slope{};
  double absolute_rate[kSystems]{};
  double accumulated[kSystems]{};
  double jll[9][9]{};
  double jlx[9][kSystems]{};
  double jxl[kSystems][9]{};
  double jxx[kSystems][kSystems]{};
};

[[nodiscard]] OPENPFC_INLINE_HD TangentFactors
factor_tangent(const FccSlipParameters &parameters, const Tensor2 &plastic_inverse,
               const Tensor2 &crystal_deformation, const Tensor2 &velocity,
               const double resistance[kSystems]) {
  TangentFactors factors;
  factors.plastic_inverse = plastic_inverse;
  factors.crystal_deformation = crystal_deformation;
  factors.mapper = matmul(crystal_deformation, plastic_inverse);
  factors.intermediate = axpy(-parameters.time_step, velocity, identity2());
  factors.elastic = matmul(factors.mapper, factors.intermediate);
  factors.second = cubic_stress(parameters, green_lagrange(factors.elastic));
  factors.growth = matmul(plastic_inverse, factors.intermediate);
  factors.kinetics = evaluate_kinetics(parameters, factors.second, resistance);
  factors.slope = slip_slope(parameters, resistance);
  for (int alpha = 0; alpha < kSystems; ++alpha) {
    factors.absolute_rate[alpha] = std::abs(factors.kinetics.rate[alpha]);
  }
  for (int alpha = 0; alpha < kSystems; ++alpha) {
    factors.accumulated[alpha] = 0.0;
    for (int beta = 0; beta < kSystems; ++beta) {
      factors.accumulated[alpha] +=
          interaction(parameters, alpha, beta) * factors.absolute_rate[beta];
    }
  }

  const auto &systems = schmid_tensors();
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const int unknown = pack_index(row, column);
      factors.jll[unknown][unknown] = 1.0;
      const Tensor2 dvelocity =
          velocity_increment(parameters, factors.mapper, factors.elastic,
                             factors.kinetics, row, column);
      double dtau[kSystems];
      Tensor2 delastic{};
      for (int i = 0; i < 3; ++i) {
        delastic(i, column) = -parameters.time_step * factors.mapper(i, row);
      }
      const Tensor2 dstrain =
          scaled(add(matmul(transposed(delastic), factors.elastic),
                     matmul(transposed(factors.elastic), delastic)),
                 0.5);
      const Tensor2 dsecond = cubic_stress(parameters, dstrain);
      for (int alpha = 0; alpha < kSystems; ++alpha) {
        dtau[alpha] = frobenius_dot(systems[alpha], dsecond);
      }
      for (int equation = 0; equation < 3; ++equation) {
        for (int slot = 0; slot < 3; ++slot) {
          factors.jll[pack_index(equation, slot)][unknown] -=
              dvelocity(equation, slot);
        }
      }
      for (int alpha = 0; alpha < kSystems; ++alpha) {
        double projected = 0.0;
        for (int beta = 0; beta < kSystems; ++beta) {
          projected += interaction(parameters, alpha, beta) *
                       factors.kinetics.dabs_dtau[beta] * dtau[beta];
        }
        factors.jxl[alpha][unknown] = -parameters.time_step *
                                      parameters.hardening_coefficient *
                                      factors.slope.value[alpha] * projected;
      }
    }
  }
  for (int beta = 0; beta < kSystems; ++beta) {
    const Tensor2 column =
        scaled(systems[beta], factors.kinetics.drate_dresistance[beta]);
    for (int equation = 0; equation < 3; ++equation) {
      for (int slot = 0; slot < 3; ++slot) {
        factors.jlx[pack_index(equation, slot)][beta] = -column(equation, slot);
      }
    }
    for (int alpha = 0; alpha < kSystems; ++alpha) {
      double extra = factors.slope.value[alpha] *
                     interaction(parameters, alpha, beta) *
                     factors.kinetics.dabs_dresistance[beta];
      if (alpha == beta) {
        extra += factors.slope.derivative[alpha] * factors.accumulated[alpha];
      }
      factors.jxx[alpha][beta] =
          (alpha == beta ? 1.0 : 0.0) -
          parameters.time_step * parameters.hardening_coefficient * extra;
    }
  }
  return factors;
}

[[nodiscard]] OPENPFC_INLINE_HD Tensor2
directional_stress(const FccSlipParameters &parameters, const Tensor2 &orientation,
                   const TangentFactors &factors, const Tensor2 &sample_increment,
                   FccStatus &status) {
  const Tensor2 crystal_increment = push_forward(orientation, sample_increment);
  const Tensor2 dmapper = matmul(crystal_increment, factors.plastic_inverse);
  const Tensor2 delastic = matmul(dmapper, factors.intermediate);
  const Tensor2 dstrain = scaled(add(matmul(transposed(delastic), factors.elastic),
                                     matmul(transposed(factors.elastic), delastic)),
                                 0.5);
  const Tensor2 dsecond = cubic_stress(parameters, dstrain);
  const auto &systems = schmid_tensors();
  double dtau[kSystems];
  double rhs_velocity[9]{};
  double rhs_resistance[kSystems]{};
  for (int alpha = 0; alpha < kSystems; ++alpha) {
    dtau[alpha] = frobenius_dot(systems[alpha], dsecond);
  }
  Tensor2 explicit_velocity{};
  for (int alpha = 0; alpha < kSystems; ++alpha) {
    explicit_velocity = axpy(factors.kinetics.drate_dtau[alpha] * dtau[alpha],
                             systems[alpha], explicit_velocity);
  }
  pack_velocity(explicit_velocity, rhs_velocity);
  for (int alpha = 0; alpha < kSystems; ++alpha) {
    double projected = 0.0;
    for (int beta = 0; beta < kSystems; ++beta) {
      projected += interaction(parameters, alpha, beta) *
                   factors.kinetics.dabs_dtau[beta] * dtau[beta];
    }
    rhs_resistance[alpha] = parameters.time_step * parameters.hardening_coefficient *
                            factors.slope.value[alpha] * projected;
  }

  double response[kSystems][9];
  for (int unknown = 0; unknown < 9; ++unknown) {
    double column[kSystems];
    double matrix[kSystems * kSystems];
    for (int alpha = 0; alpha < kSystems; ++alpha) {
      column[alpha] = factors.jxl[alpha][unknown];
      for (int beta = 0; beta < kSystems; ++beta) {
        matrix[alpha * kSystems + beta] = factors.jxx[alpha][beta];
      }
    }
    if (!solve_dense<kSystems>(matrix, column)) {
      status = FccStatus::HardeningBlock;
      return {};
    }
    for (int alpha = 0; alpha < kSystems; ++alpha) {
      response[alpha][unknown] = column[alpha];
    }
  }
  double resistance_response[kSystems];
  {
    double matrix[kSystems * kSystems];
    for (int alpha = 0; alpha < kSystems; ++alpha) {
      resistance_response[alpha] = rhs_resistance[alpha];
      for (int beta = 0; beta < kSystems; ++beta) {
        matrix[alpha * kSystems + beta] = factors.jxx[alpha][beta];
      }
    }
    if (!solve_dense<kSystems>(matrix, resistance_response)) {
      status = FccStatus::HardeningBlock;
      return {};
    }
  }

  double schur[81];
  double rhs[9];
  for (int row = 0; row < 9; ++row) {
    rhs[row] = rhs_velocity[row];
    for (int alpha = 0; alpha < kSystems; ++alpha) {
      rhs[row] -= factors.jlx[row][alpha] * resistance_response[alpha];
    }
    for (int column = 0; column < 9; ++column) {
      double value = factors.jll[row][column];
      for (int alpha = 0; alpha < kSystems; ++alpha) {
        value -= factors.jlx[row][alpha] * response[alpha][column];
      }
      schur[row * 9 + column] = value;
    }
  }
  if (!solve_dense<9>(schur, rhs)) {
    status = FccStatus::Tangent;
    return {};
  }
  const Tensor2 dvelocity = unpack_velocity(rhs);
  const Tensor2 dintermediate = scaled(dvelocity, -parameters.time_step);
  const Tensor2 dgrowth = matmul(factors.plastic_inverse, dintermediate);
  const Tensor2 delastic_total =
      add(delastic, matmul(factors.mapper, dintermediate));
  const Tensor2 dstrain_total =
      scaled(add(matmul(transposed(delastic_total), factors.elastic),
                 matmul(transposed(factors.elastic), delastic_total)),
             0.5);
  const Tensor2 dsecond_total = cubic_stress(parameters, dstrain_total);
  const Tensor2 mandel =
      matmul(factors.growth, matmul(factors.second, transposed(factors.growth)));
  const Tensor2 dmandel = add(
      add(matmul(dgrowth, matmul(factors.second, transposed(factors.growth))),
          matmul(factors.growth, matmul(dsecond_total, transposed(factors.growth)))),
      matmul(factors.growth, matmul(factors.second, transposed(dgrowth))));
  const Tensor2 dcrystal = add(matmul(crystal_increment, mandel),
                               matmul(factors.crystal_deformation, dmandel));
  return pull_stress(orientation, dcrystal);
}

[[nodiscard]] OPENPFC_INLINE_HD Tangent
assemble_tangent(const FccSlipParameters &parameters, const Tensor2 &orientation,
                 const TangentFactors &factors, FccStatus &status) {
  Tangent tangent;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      Tensor2 direction{};
      direction(row, column) = 1.0;
      const Tensor2 stress =
          directional_stress(parameters, orientation, factors, direction, status);
      if (status != FccStatus::Success) return {};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          tangent(i, j, row, column) = stress(i, j);
        }
      }
    }
  }
  return tangent;
}

[[nodiscard]] OPENPFC_INLINE_HD PiolaTangent
published_tangent(const Tangent &tangent) noexcept {
  PiolaTangent out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          out(i, j, k, l) = tangent(i, j, k, l);
        }
      }
    }
  }
  return out;
}

[[nodiscard]] OPENPFC_INLINE_HD Tensor2
apply_tangent(const Tangent &tangent, const Tensor2 &increment) noexcept {
  Tensor2 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          sum += tangent(i, j, k, l) * increment(k, l);
        }
      }
      out(i, j) = sum;
    }
  }
  return out;
}

[[nodiscard]] OPENPFC_INLINE_HD bool same_tensor(const Tensor2 &left,
                                                 const Tensor2 &right) noexcept {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (left(i, j) != right(i, j)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] OPENPFC_INLINE_HD Integrated
try_integrate(const FccSlipParameters &parameters, const Tensor2 &orientation,
              const FccSlipState &committed, const Tensor2 &deformation) {
  Integrated integrated;
  FccStatus &status = integrated.status;
  if (!admissible_parameters(parameters)) {
    status = FccStatus::Parameters;
    return integrated;
  }
  if (!proper_rotation(orientation)) {
    status = FccStatus::Orientation;
    return integrated;
  }
  const Tensor2 crystal_deformation = push_forward(orientation, deformation);
  const Tensor2 plastic_inverse = inverse(committed.plastic, status);
  if (status != FccStatus::Success) return integrated;
  const Tensor2 mapper = matmul(crystal_deformation, plastic_inverse);

  std::array<double, kSystems> resistance{};
  for (int alpha = 0; alpha < kSystems; ++alpha) {
    resistance[alpha] = committed.resistance[alpha];
  }
  Tensor2 velocity = committed.velocity;
  LpEvaluation solved{};
  bool state_converged = false;
  for (int iteration = 0; iteration < kStateIterations; ++iteration) {
    solved = solve_velocity(parameters, mapper, velocity, resistance.data(), status);
    if (status != FccStatus::Success) return integrated;
    velocity = solved.velocity;
    const SlipSlope slope = slip_slope(parameters, resistance.data());
    double absolute_rate[kSystems];
    for (int alpha = 0; alpha < kSystems; ++alpha) {
      absolute_rate[alpha] = std::abs(solved.kinetics.rate[alpha]);
    }
    std::array<double, kSystems> next{};
    double error = 0.0;
    for (int alpha = 0; alpha < kSystems; ++alpha) {
      const double rate = hardening_dot(parameters, slope, alpha, absolute_rate);
      next[alpha] = committed.resistance[alpha] + parameters.time_step * rate;
      const double scale = std::max(1.0, std::abs(next[alpha]));
      error = std::max(error, std::abs(next[alpha] - resistance[alpha]) / scale);
    }
    resistance = next;
    if (error <= 1e-12) {
      solved =
          solve_velocity(parameters, mapper, velocity, resistance.data(), status);
      if (status != FccStatus::Success) return integrated;
      velocity = solved.velocity;
      state_converged = true;
      break;
    }
  }
  if (!state_converged) {
    status = FccStatus::ResistanceIterations;
    return integrated;
  }

  const Tensor2 intermediate = axpy(-parameters.time_step, velocity, identity2());
  const Tensor2 growth = matmul(plastic_inverse, intermediate);
  const Tensor2 crystal_stress =
      matmul(crystal_deformation,
             matmul(growth, matmul(solved.second, transposed(growth))));
  const Tensor2 raw_plastic = inverse(growth, status);
  if (status != FccStatus::Success) return integrated;
  const double plastic_det = determinant(raw_plastic);
  if (!(plastic_det > 0.0)) {
    status = FccStatus::PlasticOrientation;
    return integrated;
  }
  const Tensor2 stored_plastic = scaled(raw_plastic, 1.0 / std::cbrt(plastic_det));

  integrated.update.piola = pull_stress(orientation, crystal_stress);
  integrated.update.state.plastic = stored_plastic;
  integrated.update.state.velocity = velocity;
  for (int alpha = 0; alpha < kSystems; ++alpha) {
    integrated.update.state.resistance[alpha] = resistance[alpha];
    integrated.update.state.shear[alpha] =
        committed.shear[alpha] +
        parameters.time_step * std::abs(solved.kinetics.rate[alpha]);
    integrated.update.tau[alpha] = solved.kinetics.tau[alpha];
    integrated.update.shear_rate[alpha] = solved.kinetics.rate[alpha];
  }
  const TangentFactors factors = factor_tangent(
      parameters, plastic_inverse, crystal_deformation, velocity, resistance.data());
  integrated.tangent = assemble_tangent(parameters, orientation, factors, status);
  return integrated;
}

// Host wrapper preserves the established exception types and integration counter.
[[nodiscard]] inline Integrated integrate(const FccSlipParameters &parameters,
                                          const Tensor2 &orientation,
                                          const FccSlipState &committed,
                                          const Tensor2 &deformation) {
  require_parameters(parameters);
  if (!proper_rotation(orientation))
    throw std::invalid_argument(fcc_status_message(FccStatus::Orientation));
  detail::note_constitutive_integration();
  Integrated result = try_integrate(parameters, orientation, committed, deformation);
  if (result.status != FccStatus::Success)
    throw std::runtime_error(fcc_status_message(result.status));
  return result;
}

} // namespace crystal

/// Stress, sample-frame tangent, and trial slip state from one update.
///
/// `update.state` is the trial. It is not published. The tangent is the
/// linearization of this update, already formed inside the slip solve.
struct FccEvaluation {
  FccSlipUpdate update;
  PiolaTangent tangent;
};

struct FccEvaluationResult {
  FccEvaluation value{};
  FccStatus status = FccStatus::Success;
};

/// Nonthrowing material update; input history remains immutable.
[[nodiscard]] OPENPFC_INLINE_HD FccEvaluationResult
try_evaluate_fcc(const FccSlipParameters &parameters, const Tensor2 &orientation,
                 const FccSlipState &committed, const Tensor2 &deformation) {
  const auto result =
      crystal::try_integrate(parameters, orientation, committed, deformation);
  FccEvaluationResult out;
  out.status = result.status;
  if (out.status == FccStatus::Success) {
    out.value.update = result.update;
    out.value.tangent = crystal::published_tangent(result.tangent);
  }
  return out;
}

[[nodiscard]] inline FccEvaluation evaluate_fcc(const FccSlipParameters &parameters,
                                                const Tensor2 &orientation,
                                                const FccSlipState &state,
                                                const Tensor2 &deformation) {
  const crystal::Integrated integrated =
      crystal::integrate(parameters, orientation, state, deformation);
  FccEvaluation out;
  out.update = integrated.update;
  out.tangent = crystal::published_tangent(integrated.tangent);
  return out;
}

[[nodiscard]] inline FccSlipUpdate integrate_fcc(const FccSlipParameters &parameters,
                                                 const Tensor2 &orientation,
                                                 const FccSlipState &state,
                                                 const Tensor2 &deformation) {
  return crystal::integrate(parameters, orientation, state, deformation).update;
}

[[nodiscard]] inline Tensor2 fcc_tangent_action(const FccSlipParameters &parameters,
                                                const Tensor2 &orientation,
                                                const FccSlipState &state,
                                                const Tensor2 &deformation,
                                                const Tensor2 &increment) {
  const crystal::Integrated integrated =
      crystal::integrate(parameters, orientation, state, deformation);
  return crystal::apply_tangent(integrated.tangent, increment);
}

/**
 * Per-point FCC slip. Orientation is configuration. Plastic history moves
 * only through `begin_trial`, `stage`, `accept`, and `reject`.
 *
 * `evaluate` returns the stress, the tangent of this new slip update, and
 * the trial state. It does not publish that state. `accept` keeps the
 * tangent of the step it publishes. `accepted_tangent` reads that tensor.
 * `tangent_action` returns it when the deformation is the accepted one,
 * which is the predictor used by the first Krylov product of an
 * incremental solve. At any other deformation, `tangent_action` uses the
 * same update as `evaluate`.
 */
class FccSlipMaterial {
public:
  FccSlipMaterial(std::size_t local_count, FccSlipParameters parameters)
      : parameters_(parameters), orientation_(local_count, identity2()),
        history_(
            std::vector<FccSlipState>(local_count, fcc_identity_state(parameters))),
        cache_(local_count), lagged_(local_count), pending_(local_count) {
    crystal::require_parameters(parameters_);
  }

  void set_orientation(std::size_t index, const Tensor2 &orientation) {
    if (!crystal::proper_rotation(orientation)) {
      throw std::invalid_argument("crystal orientation must be a proper rotation");
    }
    orientation_.at(index) = orientation;
    cache_.at(index).valid = false;
  }

  [[nodiscard]] Tensor2 stress(std::size_t index, const Tensor2 &deformation) const {
    return response_at(index, deformation).update.piola;
  }

  /// Fresh slip update at `deformation`. Does not publish the trial state.
  [[nodiscard]] FccEvaluation evaluate(std::size_t index,
                                       const Tensor2 &deformation) const {
    const crystal::Integrated &integrated = response_at(index, deformation);
    FccEvaluation out;
    out.update = integrated.update;
    out.tangent = crystal::published_tangent(integrated.tangent);
    return out;
  }

  [[nodiscard]] bool has_accepted_tangent(std::size_t index) const {
    return lagged_.at(index).valid;
  }

  /// Tangent stored by the last `accept` at this point.
  [[nodiscard]] PiolaTangent accepted_tangent(std::size_t index) const {
    const Linearization &lagged = lagged_.at(index);
    if (!lagged.valid) {
      throw std::logic_error("FCC point has no accepted tangent");
    }
    return crystal::published_tangent(lagged.tangent);
  }

  [[nodiscard]] Tensor2 tangent_action(std::size_t index, const Tensor2 &deformation,
                                       const Tensor2 &increment) const {
    const Linearization &lagged = lagged_.at(index);
    if (lagged.valid && crystal::same_tensor(lagged.deformation, deformation)) {
      return crystal::apply_tangent(lagged.tangent, increment);
    }
    return crystal::apply_tangent(response_at(index, deformation).tangent,
                                  increment);
  }

  void begin_trial() {
    history_.begin();
    clear_pending();
    invalidate();
  }

  void stage(std::size_t index, const Tensor2 &deformation) {
    const crystal::Integrated integrated = crystal::integrate(
        parameters_, orientation_.at(index), history_.committed(index), deformation);
    history_.trial(index) = integrated.update.state;
    Linearization &pending = pending_.at(index);
    pending.deformation = deformation;
    pending.tangent = integrated.tangent;
    pending.valid = true;
  }

  void accept() {
    history_.accept();
    for (std::size_t index = 0; index < pending_.size(); ++index) {
      if (pending_[index].valid) {
        lagged_[index] = pending_[index];
      }
    }
    clear_pending();
    invalidate();
  }

  void reject() {
    history_.reject();
    clear_pending();
    invalidate();
  }

  [[nodiscard]] const FccSlipState &committed_state(std::size_t index) const {
    return history_.committed(index);
  }

  [[nodiscard]] std::size_t size() const noexcept { return history_.size(); }

private:
  [[nodiscard]] const crystal::Integrated &
  response_at(std::size_t index, const Tensor2 &deformation) const {
    Slot &slot = cache_.at(index);
    if (slot.valid && crystal::same_tensor(slot.deformation, deformation)) {
      return slot.integrated;
    }
    slot.deformation = deformation;
    slot.integrated = crystal::integrate(parameters_, orientation_.at(index),
                                         history_.committed(index), deformation);
    slot.valid = true;
    return slot.integrated;
  }

  void invalidate() const {
    for (Slot &slot : cache_) {
      slot.valid = false;
    }
  }

  void clear_pending() {
    for (Linearization &slot : pending_) {
      slot.valid = false;
    }
  }

  struct Linearization {
    Tensor2 deformation{};
    crystal::Tangent tangent{};
    bool valid = false;
  };

  struct Slot {
    Tensor2 deformation{};
    crystal::Integrated integrated{};
    bool valid = false;
  };

  FccSlipParameters parameters_;
  std::vector<Tensor2> orientation_;
  TransactionalHistory<FccSlipState> history_;
  mutable std::vector<Slot> cache_;
  std::vector<Linearization> lagged_;
  std::vector<Linearization> pending_;
};

} // namespace pfc::finite_strain
