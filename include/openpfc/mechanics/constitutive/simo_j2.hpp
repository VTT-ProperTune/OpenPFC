// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simo_j2.hpp
 * @brief Finite-strain Simo J2 map from `F` to the first Piola stress.
 *
 * The update is the elastic-predictor / plastic-corrector used by the
 * GooseFFT program `finite-strain/elasto-plasticity.py`: logarithmic strain
 * of the elastic Finger tensor, a von Mises yield function, associative
 * flow, and linear isotropic hardening. `evaluate_j2` returns `P`, the
 * tangent of the branch that integration takes, and the trial state.
 * Plastic history stays in this law and moves only through `begin_trial`,
 * `stage`, `accept`, and `reject`.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <openpfc/mechanics/constitutive/history.hpp>
#include <openpfc/mechanics/constitutive/integration_count.hpp>
#include <openpfc/mechanics/piola_tangent.hpp>
#include <openpfc/mechanics/tensor.hpp>

namespace pfc::finite_strain {

/// Isotropic J2 moduli. `bulk` is `κ` and `initial_yield` is `τ_y0`.
struct J2Moduli {
  double bulk = 0.0;
  double shear = 0.0;
  double hardening = 0.0;
  double initial_yield = 0.0;
};

/// Committed elastic Finger tensor, deformation, and accumulated plastic strain.
struct J2State {
  Tensor2 finger{};
  Tensor2 deformation{};
  double plastic = 0.0;
};

struct J2Response {
  Tensor2 piola{};
  Tensor2 kirchhoff{};
  Tensor2 finger{};
  double plastic = 0.0;
  /// Von Mises equivalent of the Kirchhoff stress, `sqrt(3/2) ||dev τ||`.
  double equivalent = 0.0;
};

[[nodiscard]] inline J2State identity_j2_state() noexcept {
  J2State state;
  state.finger = identity2();
  state.deformation = identity2();
  return state;
}

namespace detail {

struct Tensor4 {
  double c[3][3][3][3]{};

  double &operator()(int i, int j, int k, int l) noexcept { return c[i][j][k][l]; }
  double operator()(int i, int j, int k, int l) const noexcept {
    return c[i][j][k][l];
  }
};

struct Spectral3 {
  double value[3]{};
  double vector[3][3]{};
};

[[nodiscard]] inline Spectral3 symmetric_eigen(const Tensor2 &tensor) {
  double a[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      a[i][j] = 0.5 * (tensor(i, j) + tensor(j, i));
    }
  }
  double basis[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  for (int sweep = 0; sweep < 16; ++sweep) {
    int p = 0;
    int q = 1;
    double largest = std::abs(a[0][1]);
    if (std::abs(a[0][2]) > largest) {
      largest = std::abs(a[0][2]);
      p = 0;
      q = 2;
    }
    if (std::abs(a[1][2]) > largest) {
      largest = std::abs(a[1][2]);
      p = 1;
      q = 2;
    }
    const double scale =
        1.0 + std::abs(a[0][0]) + std::abs(a[1][1]) + std::abs(a[2][2]);
    if (largest <= 1e-15 * scale) {
      break;
    }
    const double app = a[p][p];
    const double aqq = a[q][q];
    const double apq = a[p][q];
    const double tau = (aqq - app) / (2.0 * apq);
    const double tangent =
        std::copysign(1.0, tau) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
    const double sine = tangent * cosine;
    a[p][p] = app - tangent * apq;
    a[q][q] = aqq + tangent * apq;
    a[p][q] = 0.0;
    a[q][p] = 0.0;
    for (int r = 0; r < 3; ++r) {
      if (r == p || r == q) {
        continue;
      }
      const double arp = a[r][p];
      const double arq = a[r][q];
      a[r][p] = cosine * arp - sine * arq;
      a[p][r] = a[r][p];
      a[r][q] = cosine * arq + sine * arp;
      a[q][r] = a[r][q];
    }
    for (int r = 0; r < 3; ++r) {
      const double vrp = basis[r][p];
      const double vrq = basis[r][q];
      basis[r][p] = cosine * vrp - sine * vrq;
      basis[r][q] = cosine * vrq + sine * vrp;
    }
  }
  Spectral3 out;
  for (int k = 0; k < 3; ++k) {
    out.value[k] = a[k][k];
    for (int i = 0; i < 3; ++i) {
      out.vector[i][k] = basis[i][k];
    }
  }
  return out;
}

[[nodiscard]] inline Tensor2 spectral_map(const Tensor2 &tensor, bool logarithm) {
  const Spectral3 spectral = symmetric_eigen(tensor);
  Tensor2 out;
  for (int k = 0; k < 3; ++k) {
    const double value = spectral.value[k];
    if (logarithm && !(value > 0.0)) {
      throw std::runtime_error("elastic Finger tensor is not positive definite");
    }
    const double mapped = logarithm ? std::log(value) : std::exp(value);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        out(i, j) += mapped * spectral.vector[i][k] * spectral.vector[j][k];
      }
    }
  }
  return out;
}

[[nodiscard]] inline double determinant(const Tensor2 &a) noexcept {
  return a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
         a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
         a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
}

[[nodiscard]] inline Tensor2 inverse(const Tensor2 &a) {
  const double det = determinant(a);
  if (!(std::abs(det) > 0.0)) {
    throw std::runtime_error("deformation gradient is singular");
  }
  Tensor2 out;
  out(0, 0) = (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) / det;
  out(0, 1) = (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) / det;
  out(0, 2) = (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) / det;
  out(1, 0) = (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) / det;
  out(1, 1) = (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) / det;
  out(1, 2) = (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) / det;
  out(2, 0) = (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) / det;
  out(2, 1) = (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) / det;
  out(2, 2) = (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) / det;
  return out;
}

[[nodiscard]] inline Tensor2 deviator(const Tensor2 &tensor) noexcept {
  const double mean = trace(tensor) / 3.0;
  Tensor2 out = tensor;
  out(0, 0) -= mean;
  out(1, 1) -= mean;
  out(2, 2) -= mean;
  return out;
}

[[nodiscard]] inline double equivalent_stress(const Tensor2 &tensor) noexcept {
  const Tensor2 dev = deviator(tensor);
  return std::sqrt(1.5 * frobenius_dot(dev, dev));
}

[[nodiscard]] inline Tensor4 identity_right() {
  Tensor4 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j, j, i) = 1.0;
    }
  }
  return out;
}

[[nodiscard]] inline Tensor4 identity_transpose() {
  Tensor4 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j, i, j) = 1.0;
    }
  }
  return out;
}

[[nodiscard]] inline Tensor4 scaled4(const Tensor4 &tensor, double factor) {
  Tensor4 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          out(i, j, k, l) = factor * tensor(i, j, k, l);
        }
      }
    }
  }
  return out;
}

[[nodiscard]] inline Tensor4 add4(const Tensor4 &left, const Tensor4 &right) {
  Tensor4 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          out(i, j, k, l) = left(i, j, k, l) + right(i, j, k, l);
        }
      }
    }
  }
  return out;
}

[[nodiscard]] inline Tensor4 dyad(const Tensor2 &left, const Tensor2 &right) {
  Tensor4 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          out(i, j, k, l) = left(i, j) * right(k, l);
        }
      }
    }
  }
  return out;
}

/// `C_ij = A_ijkl B_lk`, the GooseFFT `ddot42` contraction.
[[nodiscard]] inline Tensor2 contract42(const Tensor4 &tensor,
                                        const Tensor2 &vector) {
  Tensor2 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          sum += tensor(i, j, k, l) * vector(l, k);
        }
      }
      out(i, j) = sum;
    }
  }
  return out;
}

[[nodiscard]] inline Tensor4 contract44(const Tensor4 &left, const Tensor4 &right) {
  Tensor4 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int m = 0; m < 3; ++m) {
        for (int n = 0; n < 3; ++n) {
          double sum = 0.0;
          for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
              sum += left(i, j, k, l) * right(l, k, m, n);
            }
          }
          out(i, j, m, n) = sum;
        }
      }
    }
  }
  return out;
}

/// `C_ijkm = A_ijkl B_lm`.
[[nodiscard]] inline Tensor4 push42(const Tensor4 &tensor, const Tensor2 &vector) {
  Tensor4 out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int m = 0; m < 3; ++m) {
          double sum = 0.0;
          for (int l = 0; l < 3; ++l) {
            sum += tensor(i, j, k, l) * vector(l, m);
          }
          out(i, j, k, m) = sum;
        }
      }
    }
  }
  return out;
}

/// `C_ikmn = A_ij B_jkmn`.
[[nodiscard]] inline Tensor4 push24(const Tensor2 &vector, const Tensor4 &tensor) {
  Tensor4 out;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      for (int m = 0; m < 3; ++m) {
        for (int n = 0; n < 3; ++n) {
          double sum = 0.0;
          for (int j = 0; j < 3; ++j) {
            sum += vector(i, j) * tensor(j, k, m, n);
          }
          out(i, k, m, n) = sum;
        }
      }
    }
  }
  return out;
}

[[nodiscard]] inline Tensor4 symmetric_identity() {
  return scaled4(add4(identity_right(), identity_transpose()), 0.5);
}

[[nodiscard]] inline Tensor4 elastic_stiffness(J2Moduli moduli) {
  const Tensor4 volumetric = dyad(identity2(), identity2());
  const Tensor4 deviatoric =
      add4(symmetric_identity(), scaled4(volumetric, -1.0 / 3.0));
  return add4(scaled4(volumetric, moduli.bulk),
              scaled4(deviatoric, 2.0 * moduli.shear));
}

[[nodiscard]] inline Tensor4 logarithm_derivative(const Tensor2 &finger) {
  const Spectral3 spectral = symmetric_eigen(finger);
  Tensor4 out;
  for (int m = 0; m < 3; ++m) {
    for (int n = 0; n < 3; ++n) {
      const double left = spectral.value[m];
      const double right = spectral.value[n];
      const double separated = std::abs(right - left);
      const double scale = std::max(1.0, std::abs(left));
      const double coefficient =
          separated <= 1e-12 * scale
              ? 1.0 / left
              : (std::log(right) - std::log(left)) / (right - left);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
              out(i, j, k, l) += coefficient * spectral.vector[i][m] *
                                 spectral.vector[j][n] * spectral.vector[k][m] *
                                 spectral.vector[l][n];
            }
          }
        }
      }
    }
  }
  return out;
}

struct IntegratedJ2 {
  J2Response response;
  Tensor4 tangent;
};

[[nodiscard]] inline Tensor2 kirchhoff_from_log(const Tensor2 &log_finger,
                                                J2Moduli moduli) {
  return contract42(scaled4(elastic_stiffness(moduli), 0.5), log_finger);
}

/// Public `K_ijkl = A_jikl`, so `K.action` matches `consistent_action`.
[[nodiscard]] inline PiolaTangent
published_tangent(const Tensor4 &tangent) noexcept {
  PiolaTangent out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          out(i, j, k, l) = tangent(j, i, k, l);
        }
      }
    }
  }
  return out;
}

[[nodiscard]] inline IntegratedJ2 integrate(J2Moduli moduli, const J2State &state,
                                            const Tensor2 &deformation) {
  if (!(moduli.shear > 0.0) || !(moduli.hardening + 3.0 * moduli.shear > 0.0)) {
    throw std::invalid_argument("J2 shear and plastic modulus must be positive");
  }
  note_constitutive_integration();
  const Tensor2 inverse_previous = inverse(state.deformation);
  const Tensor2 relative = matmul(deformation, inverse_previous);
  const Tensor2 trial_finger =
      matmul(relative, matmul(state.finger, transposed(relative)));
  const Tensor2 trial_log = spectral_map(trial_finger, true);
  const Tensor2 trial_kirchhoff = kirchhoff_from_log(trial_log, moduli);
  const Tensor2 trial_deviator = deviator(trial_kirchhoff);
  const double trial_equivalent = equivalent_stress(trial_kirchhoff);
  const double yield_stress =
      moduli.initial_yield + moduli.hardening * state.plastic;
  const double raw_phi = trial_equivalent - yield_stress;
  const double phi = 0.5 * (raw_phi + std::abs(raw_phi));
  const bool plastic = phi > 0.0;
  if (plastic && !(trial_equivalent > 0.0)) {
    throw std::runtime_error("J2 return map has a vanishing trial stress");
  }
  const Tensor2 flow =
      plastic ? scaled(trial_deviator, 1.5 / trial_equivalent) : Tensor2{};
  const double gamma = phi / (moduli.hardening + 3.0 * moduli.shear);

  IntegratedJ2 out;
  out.response.plastic = state.plastic + gamma;
  out.response.kirchhoff =
      add(trial_kirchhoff, scaled(flow, -2.0 * gamma * moduli.shear));
  const Tensor2 corrected_log = add(trial_log, scaled(flow, -2.0 * gamma));
  out.response.finger = spectral_map(corrected_log, false);
  const Tensor2 inverse_deformation = inverse(deformation);
  out.response.piola =
      matmul(out.response.kirchhoff, transposed(inverse_deformation));
  out.response.equivalent = equivalent_stress(out.response.kirchhoff);

  const Tensor4 volumetric = dyad(identity2(), identity2());
  const Tensor4 isotropic = symmetric_identity();
  const double a0 = plastic ? gamma * moduli.shear / trial_equivalent : 0.0;
  const double a1 = moduli.shear / (moduli.hardening + 3.0 * moduli.shear);
  Tensor4 spatial;
  if (!plastic) {
    spatial = scaled4(elastic_stiffness(moduli), 0.5);
  } else {
    const double volumetric_factor =
        (moduli.bulk - 2.0 * moduli.shear / 3.0) / 2.0 + a0 * moduli.shear;
    spatial = add4(scaled4(volumetric, volumetric_factor),
                   scaled4(isotropic, (1.0 - 3.0 * a0) * moduli.shear));
    spatial =
        add4(spatial, scaled4(dyad(flow, flow), 2.0 * moduli.shear * (a0 - a1)));
  }
  const Tensor4 dlog = logarithm_derivative(trial_finger);
  const Tensor4 dfinger = scaled4(push42(isotropic, trial_finger), 2.0);
  spatial = contract44(spatial, contract44(dlog, dfinger));
  Tensor4 geometric;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int m = 0; m < 3; ++m) {
        geometric(i, j, i, m) = -out.response.kirchhoff(j, m);
      }
    }
  }
  spatial = add4(geometric, spatial);
  out.tangent =
      push42(push24(inverse_deformation, spatial), transposed(inverse_deformation));
  return out;
}

/// Directional derivative `dP = (K : dF^T)^T` in the GooseFFT layout.
[[nodiscard]] inline Tensor2 consistent_action(const Tensor4 &tangent,
                                               const Tensor2 &increment) {
  return transposed(contract42(tangent, transposed(increment)));
}

struct CachedResponse {
  Tensor2 deformation{};
  IntegratedJ2 integrated{};
  bool valid = false;
};

} // namespace detail

/// Stress, branch tangent, and trial history from one return map.
///
/// `trial` is not published. The tangent is the branch this integration
/// takes. A state that already lies on the yield surface takes the
/// unloading branch, because plasticity requires `φ > 0`.
struct J2Evaluation {
  J2Response response;
  PiolaTangent tangent;
  J2State trial;
};

[[nodiscard]] inline J2Evaluation evaluate_j2(J2Moduli moduli, const J2State &state,
                                              const Tensor2 &deformation) {
  const detail::IntegratedJ2 integrated =
      detail::integrate(moduli, state, deformation);
  J2Evaluation out;
  out.response = integrated.response;
  out.tangent = detail::published_tangent(integrated.tangent);
  out.trial.finger = integrated.response.finger;
  out.trial.deformation = deformation;
  out.trial.plastic = integrated.response.plastic;
  return out;
}

[[nodiscard]] inline J2Response integrate_j2(J2Moduli moduli, const J2State &state,
                                             const Tensor2 &deformation) {
  return detail::integrate(moduli, state, deformation).response;
}

[[nodiscard]] inline Tensor2 j2_tangent_action(J2Moduli moduli, const J2State &state,
                                               const Tensor2 &deformation,
                                               const Tensor2 &increment) {
  const detail::IntegratedJ2 integrated =
      detail::integrate(moduli, state, deformation);
  return detail::consistent_action(integrated.tangent, increment);
}

/**
 * Simo J2 response on an FFT inbox.
 *
 * `stress`, `evaluate`, and `tangent_action` read the committed state and
 * do not publish history. `stage` writes the trial that `accept` publishes.
 *
 * `evaluate` returns the stress, the tangent of the branch that this new
 * integration takes, and the trial state. On a state that already lies on
 * the yield surface that branch is unloading, because the return map
 * enters plasticity only for `φ > 0`.
 *
 * `accept` keeps the consistent tangent of the step it publishes.
 * `accepted_tangent` reads that tensor. `tangent_action` returns it when
 * the deformation is the accepted one, which is the loading-side predictor
 * used by the first Krylov product of an incremental solve. At any other
 * deformation, `tangent_action` uses the same branch as `evaluate`.
 */
class SimoJ2Material {
public:
  SimoJ2Material(const std::vector<std::uint8_t> &phase, std::size_t local_count,
                 J2Moduli soft, J2Moduli hard)
      : phase_(phase), soft_(soft), hard_(hard),
        history_(std::vector<J2State>(local_count, identity_j2_state())),
        cache_(local_count), lagged_(local_count), pending_(local_count) {
    if (!phase_.empty() && phase_.size() != local_count) {
      throw std::invalid_argument("Simo J2 phase size does not match the FFT inbox");
    }
    for (const std::uint8_t mark : phase_) {
      if (mark > 1) {
        throw std::invalid_argument("phase indicator must be 0 (soft) or 1 (hard)");
      }
    }
    if (!(soft_.shear > 0.0) || !(hard_.shear > 0.0) ||
        !(soft_.hardening + 3.0 * soft_.shear > 0.0) ||
        !(hard_.hardening + 3.0 * hard_.shear > 0.0)) {
      throw std::invalid_argument("J2 shear and plastic modulus must be positive");
    }
  }

  [[nodiscard]] Tensor2 stress(std::size_t index, const Tensor2 &deformation) const {
    return response_at(index, deformation).response.piola;
  }

  /// Fresh branch at `deformation`. Does not publish `trial`.
  [[nodiscard]] J2Evaluation evaluate(std::size_t index,
                                      const Tensor2 &deformation) const {
    const detail::IntegratedJ2 &integrated = response_at(index, deformation);
    J2Evaluation out;
    out.response = integrated.response;
    out.tangent = detail::published_tangent(integrated.tangent);
    out.trial.finger = integrated.response.finger;
    out.trial.deformation = deformation;
    out.trial.plastic = integrated.response.plastic;
    return out;
  }

  [[nodiscard]] bool has_accepted_tangent(std::size_t index) const {
    return lagged_.at(index).valid;
  }

  /// Loading-side tangent stored by the last `accept` at this point.
  [[nodiscard]] PiolaTangent accepted_tangent(std::size_t index) const {
    const Linearization &lagged = lagged_.at(index);
    if (!lagged.valid) {
      throw std::logic_error("J2 point has no accepted tangent");
    }
    return detail::published_tangent(lagged.tangent);
  }

  [[nodiscard]] Tensor2 tangent_action(std::size_t index, const Tensor2 &deformation,
                                       const Tensor2 &increment) const {
    const Linearization &lagged = lagged_.at(index);
    if (lagged.valid && same_tensor(lagged.deformation, deformation)) {
      return detail::consistent_action(lagged.tangent, increment);
    }
    return detail::consistent_action(response_at(index, deformation).tangent,
                                     increment);
  }

  void begin_trial() {
    history_.begin();
    clear_pending();
    invalidate();
  }

  void stage(std::size_t index, const Tensor2 &deformation) {
    const detail::IntegratedJ2 integrated =
        detail::integrate(moduli_at(index), history_.committed(index), deformation);
    J2State next;
    next.finger = integrated.response.finger;
    next.deformation = deformation;
    next.plastic = integrated.response.plastic;
    history_.trial(index) = next;
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

  [[nodiscard]] const TransactionalHistory<J2State> &history() const noexcept {
    return history_;
  }

  [[nodiscard]] double accumulated(std::size_t index) const {
    return history_.committed(index).plastic;
  }

  [[nodiscard]] std::size_t size() const noexcept { return history_.size(); }

private:
  [[nodiscard]] J2Moduli moduli_at(std::size_t index) const {
    if (phase_.empty() || phase_[index] == 0) {
      return soft_;
    }
    return hard_;
  }

  [[nodiscard]] const detail::IntegratedJ2 &
  response_at(std::size_t index, const Tensor2 &deformation) const {
    detail::CachedResponse &slot = cache_.at(index);
    if (slot.valid && same_tensor(slot.deformation, deformation)) {
      return slot.integrated;
    }
    slot.deformation = deformation;
    slot.integrated =
        detail::integrate(moduli_at(index), history_.committed(index), deformation);
    slot.valid = true;
    return slot.integrated;
  }

  void invalidate() const {
    for (detail::CachedResponse &slot : cache_) {
      slot.valid = false;
    }
  }

  void clear_pending() {
    for (Linearization &slot : pending_) {
      slot.valid = false;
    }
  }

  [[nodiscard]] static bool same_tensor(const Tensor2 &left,
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

  struct Linearization {
    Tensor2 deformation{};
    detail::Tensor4 tangent{};
    bool valid = false;
  };

  std::vector<std::uint8_t> phase_;
  J2Moduli soft_;
  J2Moduli hard_;
  TransactionalHistory<J2State> history_;
  mutable std::vector<detail::CachedResponse> cache_;
  /// Consistent tangent of the last accepted step, keyed by its deformation.
  std::vector<Linearization> lagged_;
  std::vector<Linearization> pending_;
};

} // namespace pfc::finite_strain
