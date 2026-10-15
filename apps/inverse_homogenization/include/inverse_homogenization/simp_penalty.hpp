// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file simp_penalty.hpp
 * @brief SIMP density \(h^p\) and chain-rule \(p h^{p-1}\) used by CPU and HIP.
 *
 * The law is the library's `pfc::solvers::SimpInterpolation`
 * (`openpfc/solvers/microelasticity/simp.hpp`); these are the names the
 * drivers use. \(p=1\) is linear interpolation and \(h\) is clamped to
 * \([0,1]\).
 */

#include <openpfc/solvers/microelasticity/simp.hpp>

namespace pfc::apps::inverse {

[[nodiscard]] inline double clamp01(double h) noexcept {
  return pfc::solvers::SimpInterpolation::clamp01(h);
}

[[nodiscard]] inline double simp_density(double h, double p) noexcept {
  return pfc::solvers::SimpInterpolation{p}.weight(h);
}

[[nodiscard]] inline double simp_chain(double h, double p) noexcept {
  return pfc::solvers::SimpInterpolation{p}.slope(h);
}

} // namespace pfc::apps::inverse
