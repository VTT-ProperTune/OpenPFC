// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_etd_toys_pointwise.hpp
 * @brief Device-capable pointwise functors of the toy spectral-ETD physics.
 *
 * Kept free of `SimulationState` / JSON so a CUDA or HIP translation unit can
 * include it and instantiate `spectral_pointwise_apply` for each functor
 * (see tests/unit/runtime/gpu/spectral_etd_toys_pointwise.inc).
 */

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace pfc::test {

/// Swift–Hohenberg: \f$N(\psi) = -\psi^3\f$.
struct SHPointwise {
  OPENPFC_HD double nonlinearity(const pfc::sim::SpectralCell &c) const {
    return -c.psi * c.psi * c.psi;
  }
};

/// Mean-field toy: \f$N = p_3\psi^2 + q_3\psi_{MF}^2 - s\,\psi\f$.
struct MeanFieldToyPointwise {
  double p3{-0.5};
  double q3{0.1};
  double stab{0.2};
  OPENPFC_HD double nonlinearity(const pfc::sim::SpectralCell &c) const {
    return p3 * c.psi * c.psi + q3 * c.psi_mf * c.psi_mf - stab * c.psi;
  }
};

/// Moving-frame toy: \f$N = p_3\psi^2 + q_3\psi_{MF}^2 - P{*}\psi + g\,(x - t)\f$
/// with free-energy density \f$\tfrac12\psi^2\f$.
struct MovingFrameToyPointwise {
  double p3{-0.5};
  double q3{0.1};
  double g{0.01};
  OPENPFC_HD double nonlinearity(const pfc::sim::SpectralCell &c) const {
    const double t_var = g * (c.x - c.t);
    return p3 * c.psi * c.psi + q3 * c.psi_mf * c.psi_mf - c.p_star + t_var;
  }
  OPENPFC_HD double free_energy_density(const pfc::sim::SpectralCell &c) const {
    return 0.5 * c.psi * c.psi;
  }
};

} // namespace pfc::test
