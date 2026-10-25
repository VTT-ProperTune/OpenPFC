// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once
// Deliberate app-owned copy: these are solidification-model rules, not a
// physics-agnostic OpenPFC API. Independently buildable apps cannot include a
// sibling app. Share them only after extraction to their model's software owner;
// until then retain parity with the other solidification app's boundary rules.

/**
 * @file solidification_bc_rules.hpp
 * @brief The cell formulas of the fixed and moving solidification BCs.
 *
 * @details
 * `FixedBC` and `MovingBC` evaluate these on the host, and their device
 * appliers launch the same functors on device fields, so the two paths
 * share one formula. This header is all the device compiler sees of them.
 */

#include <cmath>

#include <openpfc/kernel/data/host_device.hpp>

namespace tungsten {

/// Logistic blend from rho_high (behind) to rho_low (ahead) across a band.
struct BandBlend {
  double alpha{1.0};
  double rho_low{};
  double rho_high{};

  OPENPFC_HD double operator()(double d) const {
    const double S = 1.0 / (1.0 + std::exp(-alpha * d));
    return (rho_low * S) + (rho_high * (1.0 - S));
  }
};

/// Band of half-width `xwidth` around the absolute position `xpos`.
struct FixedBandRule {
  BandBlend blend;
  double xpos{};
  double xwidth{};

  OPENPFC_HD double operator()(double x, double /*y*/, double /*z*/,
                               double current) const {
    return std::abs(x - xpos) < xwidth ? blend(x - xpos) : current;
  }
};

/**
 * @brief Band around a front at `xpos` measured from the lower x limit `x0`,
 *        with its images one period `l` either side.
 */
struct MovingBandRule {
  BandBlend blend;
  double x0{};
  double l{};
  double xpos{}; ///< in [0, l), relative to x0
  double xwidth{};

  OPENPFC_HD double operator()(double x, double /*y*/, double /*z*/,
                               double current) const {
    const double dist = (x - x0) - xpos;
    if (std::abs(dist) < xwidth) return blend(dist);
    if (xpos < xwidth && std::abs(dist - l) < xwidth) return blend(dist - l);
    if (xpos > l - xwidth && std::abs(dist + l) < xwidth) return blend(dist + l);
    return current;
  }
};

} // namespace tungsten
