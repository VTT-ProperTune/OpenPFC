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
 * Optional thermal drive (same unwrap as aluminum):
 * \f$T_{\mathrm{var}}(x,t) = G\,(x' - x_0 - V t)\f$. When \f$G = 0\f$ the
 * mean-field cubic uses the reference \f$\bar q_3(T)\f$ so existing goldens
 * stay bit-identical. When \f$G \ne 0\f$,
 * \f$\bar q_3\f$ is rebuilt from \f$(T + T_{\mathrm{var}})/T_0\f$.
 *
 * This header is intentionally tiny and JSON-free so a CUDA/HIP translation
 * unit can include it and instantiate the device launcher
 * (`OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(tungsten::TungstenPointwise)`).
 * The coefficients are filled by `TungstenPhysics::pointwise()` from
 * `TungstenParams`.
 */

#include <cmath>

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace tungsten {

struct TungstenPointwise {
  double c_psi{};  ///< \f$-\mathrm{stabP}\f$
  double c_psi2{}; ///< \f$\bar p_3\f$
  double c_psi3{}; ///< \f$\bar p_4\f$
  double c_mf2{};  ///< \f$\bar q_3\f$ at the JSON reference temperature
  double c_mf3{};  ///< \f$\bar q_4\f$
  double T{};
  double T0{1.0};
  double q30_bar{};
  double q31_bar{};
  double G_grid{};
  double V_grid{};
  double x_initial{};
  double front_x{};
  double length_x{};

  OPENPFC_HD double temperature_variation(double x, double t) const {
    if (G_grid == 0.0) {
      return 0.0;
    }
    if (length_x <= 0.0) {
      return G_grid * (x - x_initial - V_grid * t);
    }
    const double fullruns = std::floor(front_x / length_x) * length_x;
    const double steppoint = std::fmod(front_x, length_x);
    const double dist = x + fullruns - (x > steppoint) * length_x;
    return G_grid * (dist - x_initial - V_grid * t);
  }

  OPENPFC_HD double nonlinearity(const pfc::sim::SpectralCell &c) const {
    const double u = c.psi;
    const double v = c.psi_mf;
    const double u2 = u * u;
    const double v2 = v * v;
    double q3 = c_mf2;
    if (G_grid != 0.0) {
      const double T_var = temperature_variation(c.x, c.t);
      q3 = q31_bar * (T + T_var) / T0 + q30_bar;
    }
    return c_psi * u + c_psi2 * u2 + c_psi3 * u2 * u + q3 * v2 + c_mf3 * v2 * v;
  }
};

static_assert(pfc::sim::SpectralPointwise<TungstenPointwise>);

} // namespace tungsten
