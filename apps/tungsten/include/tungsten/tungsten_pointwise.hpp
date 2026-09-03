// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file tungsten_pointwise.hpp
 * @brief Device-capable real-space nonlinearity of the tungsten PFC model.
 *
 * @details
 * \f$N(\psi,\psi_{\mathrm{MF}}) = -s\,\psi + \bar p_3\psi^2 + \bar p_4\psi^3
 *   + \bar q_3\psi_{\mathrm{MF}}^2 + \bar q_4\psi_{\mathrm{MF}}^3\f$
 *
 * This header is intentionally tiny and JSON-free so a CUDA/HIP translation
 * unit can include it and instantiate the device launcher
 * (`OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(tungsten::TungstenPointwise)`).
 * The coefficients are filled by `TungstenPhysics::pointwise()` from
 * `TungstenParams`.
 */

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace tungsten {

struct TungstenPointwise {
  double c_psi{};  ///< \f$-\mathrm{stabP}\f$
  double c_psi2{}; ///< \f$\bar p_3\f$
  double c_psi3{}; ///< \f$\bar p_4\f$
  double c_mf2{};  ///< \f$\bar q_3\f$
  double c_mf3{};  ///< \f$\bar q_4\f$

  OPENPFC_HD double nonlinearity(const pfc::sim::SpectralCell &c) const {
    const double u = c.psi;
    const double v = c.psi_mf;
    const double u2 = u * u;
    const double v2 = v * v;
    return c_psi * u + c_psi2 * u2 + c_psi3 * u2 * u + c_mf2 * v2 + c_mf3 * v2 * v;
  }
};

static_assert(pfc::sim::SpectralPointwise<TungstenPointwise>);

} // namespace tungsten
