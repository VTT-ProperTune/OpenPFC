// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file piola_tangent.hpp
 * @brief Stored local derivative of the first Piola–Kirchhoff stress.
 *
 * `K_ijkl` is `∂P_ij / ∂F_kl`. The action is the plain sum
 * `dP_ij = sum_kl K_ijkl dF_kl`. A `9×9` matrix uses the component
 * index `3*i+j` as the row and `3*k+l` as the column.
 *
 * Laws that form this tensor in another layout convert at their
 * boundary. Callers apply `action` and do not repeat the constitutive
 * integration that produced `K`.
 */

#pragma once

#include <openpfc/mechanics/tensor.hpp>

namespace pfc::finite_strain {

struct PiolaTangent {
  double c[3][3][3][3]{};

  OPENPFC_HD double &operator()(int i, int j, int k, int l) noexcept {
    return c[i][j][k][l];
  }
  OPENPFC_HD double operator()(int i, int j, int k, int l) const noexcept {
    return c[i][j][k][l];
  }

  [[nodiscard]] OPENPFC_HD Tensor2 action(const Tensor2 &increment) const noexcept {
    Tensor2 out;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        double sum = 0.0;
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            sum += c[i][j][k][l] * increment(k, l);
          }
        }
        out(i, j) = sum;
      }
    }
    return out;
  }
};

} // namespace pfc::finite_strain
