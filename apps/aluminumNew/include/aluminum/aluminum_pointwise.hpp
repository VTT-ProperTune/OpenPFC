// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file aluminum_pointwise.hpp
 * @brief Device-capable real-space nonlinearity and free-energy density of the
 *        aluminum moving-frame PFC model.
 *
 * @details
 * Temperature varies along x in a frame moving with the solidification front:
 * \f$T_{\mathrm{var}}(x,t) = G\,(x' - x_0 - V t)\f$ with \f$x'\f$ the periodic
 * unwrap of \f$x\f$ against the front position. Every constant the kernel
 * needs is stored by value so the functor can be passed to a GPU kernel;
 * `AluminumPhysics::pointwise()` fills it from `AluminumParams`.
 *
 * Formulas match the Gen-1 `Aluminum::step` implementation (pinned in
 * tests/test_aluminum_physics.cpp).
 */

#include <cmath>

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace aluminum {

struct AluminumPointwise {
  double p3_bar{};
  double p4_bar{};
  double p2_bar{};
  double q2_bar{};
  double q21_bar{};
  double q30_bar{};
  double q31_bar{};
  double q4_bar{};
  double T0{1.0};
  double T_const{};
  double stabP{};
  double G_grid{};
  double V_grid{};
  double x_initial{};
  double front_x{};  ///< current front position (`m_xpos`)
  double length_x{}; ///< periodic length of the domain along x

  OPENPFC_HD double temperature_variation(double x, double t) const {
    const double fullruns = std::floor(front_x / length_x) * length_x;
    const double steppoint = std::fmod(front_x, length_x);
    const double dist = x + fullruns - (x > steppoint) * length_x;
    return G_grid * (dist - x_initial - V_grid * t);
  }

  OPENPFC_HD double nonlinearity(double psi, double psi_mf, double p_star,
                                 double T_var) const {
    const double q2_bar_n = q21_bar * T_var / T0;
    const double q3_bar_n = q31_bar * (T_const + T_var) / T0 + q30_bar;
    const double kernel_term = -(1.0 - std::exp(-T_var / T0)) * p_star;
    const double u2 = psi * psi;
    const double v2 = psi_mf * psi_mf;
    double n = p3_bar * u2 + p4_bar * u2 * psi + q2_bar_n * psi_mf + q3_bar_n * v2 +
               q4_bar * v2 * psi_mf - kernel_term;
    if (stabP != 0.0) {
      n -= stabP * psi;
    }
    return n;
  }

  OPENPFC_HD double free_energy_density(double psi, double psi_mf, double p_star,
                                        double T_var) const {
    const double q2_bar_n = q21_bar * T_var / T0;
    const double q3_bar_n = q31_bar * (T_const + T_var) / T0 + q30_bar;
    const double kernel_term = -(1.0 - std::exp(-T_var / T0)) * p_star;
    const double u = psi;
    const double v = psi_mf;
    return p3_bar * u * u * u / 3.0 + p4_bar * u * u * u * u / 4.0 +
           q2_bar_n * u * v / 2.0 + q3_bar_n * u * v * v / 3.0 +
           q4_bar * u * v * v * v / 4.0 - u * kernel_term * u / 2.0 -
           u * p_star / 2.0 + p2_bar * u * u / 2.0 + q2_bar * u * v / 2.0;
  }

  OPENPFC_HD double nonlinearity(const pfc::sim::SpectralCell &c) const {
    return nonlinearity(c.psi, c.psi_mf, c.p_star, temperature_variation(c.x, c.t));
  }

  OPENPFC_HD double free_energy_density(const pfc::sim::SpectralCell &c) const {
    return free_energy_density(c.psi, c.psi_mf, c.p_star,
                               temperature_variation(c.x, c.t));
  }
};

static_assert(pfc::sim::SpectralPointwise<AluminumPointwise>);
static_assert(pfc::sim::HasFreeEnergyDensity<AluminumPointwise>);

} // namespace aluminum
