// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file saint_venant_kirchhoff.hpp
 * @brief Local Saint-Venant–Kirchhoff map `F -> (P, dP)`.
 *
 * The strain energy uses the Green–Lagrange strain
 * `E = (F^T F - I) / 2` and the isotropic stiffness
 * `S = λ tr(E) I + 2 μ E`, with `P = F S`. Lamé's first parameter is
 * stored through the bulk modulus, `λ = κ - 2μ/3`, which is the splitting
 * used by the published GooseFFT hyperelastic program.
 *
 * `tangent_action(F, dF)` is the directional derivative `dP`. It is the
 * action that the Newton step writes as `K^{LT} : dF^T` in de Geus et al.,
 * Eq. (23). `evaluate_saint_venant` stores that same derivative by
 * applying it to the nine basis tensors. There is no history and no
 * constitutive integration.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <openpfc/mechanics/piola_tangent.hpp>
#include <openpfc/mechanics/tensor.hpp>

namespace pfc::finite_strain {

/// Isotropic moduli at one material point. `bulk` is `κ`, not Young's modulus.
struct IsotropicModuli {
  double bulk = 0.0;
  double shear = 0.0;
};

struct StressResponse {
  Tensor2 piola{};     ///< First Piola–Kirchhoff stress `P`.
  Tensor2 second_pk{}; ///< Second Piola–Kirchhoff stress `S`.
};

[[nodiscard]] inline Tensor2 green_lagrange(const Tensor2 &deformation) noexcept {
  const Tensor2 ft_f = matmul(transposed(deformation), deformation);
  Tensor2 strain = scaled(ft_f, 0.5);
  strain(0, 0) -= 0.5;
  strain(1, 1) -= 0.5;
  strain(2, 2) -= 0.5;
  return strain;
}

/// `S = λ tr(E) I + 2 μ E` with `λ = κ - 2μ/3`. `E` must be symmetric.
[[nodiscard]] inline Tensor2 second_piola(const Tensor2 &strain,
                                          IsotropicModuli moduli) noexcept {
  const double lambda = moduli.bulk - 2.0 * moduli.shear / 3.0;
  const double volumetric = lambda * trace(strain);
  Tensor2 stress = scaled(strain, 2.0 * moduli.shear);
  stress(0, 0) += volumetric;
  stress(1, 1) += volumetric;
  stress(2, 2) += volumetric;
  return stress;
}

[[nodiscard]] inline StressResponse
constitutive_response(const Tensor2 &deformation, IsotropicModuli moduli) noexcept {
  StressResponse out;
  out.second_pk = second_piola(green_lagrange(deformation), moduli);
  out.piola = matmul(deformation, out.second_pk);
  return out;
}

/**
 * Directional derivative `dP` at `deformation` in the direction `increment`.
 *
 * `dE = sym(F^T dF)`, `dS = λ tr(dE) I + 2 μ dE`, `dP = dF S + F dS`.
 */
[[nodiscard]] inline Tensor2 tangent_action(const Tensor2 &deformation,
                                            const Tensor2 &increment,
                                            IsotropicModuli moduli) noexcept {
  const Tensor2 second_pk = second_piola(green_lagrange(deformation), moduli);
  const Tensor2 strain_inc =
      symmetric_part(matmul(transposed(deformation), increment));
  const Tensor2 stress_inc = second_piola(strain_inc, moduli);
  return add(matmul(increment, second_pk), matmul(deformation, stress_inc));
}

/// Stress and stored tangent. The tangent is nine analytic actions.
struct SaintVenantEvaluation {
  StressResponse response;
  PiolaTangent tangent;
};

[[nodiscard]] inline SaintVenantEvaluation
evaluate_saint_venant(const Tensor2 &deformation, IsotropicModuli moduli) noexcept {
  SaintVenantEvaluation out;
  out.response = constitutive_response(deformation, moduli);
  for (int column = 0; column < 3; ++column) {
    for (int row = 0; row < 3; ++row) {
      Tensor2 increment{};
      increment(row, column) = 1.0;
      const Tensor2 stress = tangent_action(deformation, increment, moduli);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          out.tangent(i, j, row, column) = stress(i, j);
        }
      }
    }
  }
  return out;
}

/**
 * Saint-Venant-Kirchhoff response on an FFT inbox.
 *
 * An empty phase marks every point soft. Otherwise the phase has one byte
 * per inbox point, with `1` selecting the hard moduli.
 */
class SaintVenantKirchhoffMaterial {
public:
  SaintVenantKirchhoffMaterial(const std::vector<std::uint8_t> &phase,
                               std::size_t local_count, IsotropicModuli soft,
                               IsotropicModuli hard)
      : phase_(phase), soft_(soft), hard_(hard) {
    if (!phase_.empty() && phase_.size() != local_count) {
      throw std::invalid_argument(
          "Saint-Venant-Kirchhoff phase size does not match the FFT inbox");
    }
    for (const std::uint8_t mark : phase_) {
      if (mark > 1) {
        throw std::invalid_argument("phase indicator must be 0 (soft) or 1 (hard)");
      }
    }
  }

  [[nodiscard]] Tensor2 stress(std::size_t index, const Tensor2 &deformation) const {
    return constitutive_response(deformation, moduli_at(index)).piola;
  }

  [[nodiscard]] SaintVenantEvaluation evaluate(std::size_t index,
                                               const Tensor2 &deformation) const {
    return evaluate_saint_venant(deformation, moduli_at(index));
  }

  [[nodiscard]] Tensor2 tangent_action(std::size_t index, const Tensor2 &deformation,
                                       const Tensor2 &increment) const {
    return pfc::finite_strain::tangent_action(deformation, increment,
                                              moduli_at(index));
  }

private:
  [[nodiscard]] IsotropicModuli moduli_at(std::size_t index) const {
    if (phase_.empty() || phase_[index] == 0) {
      return soft_;
    }
    return hard_;
  }

  const std::vector<std::uint8_t> &phase_;
  IsotropicModuli soft_;
  IsotropicModuli hard_;
};

} // namespace pfc::finite_strain
