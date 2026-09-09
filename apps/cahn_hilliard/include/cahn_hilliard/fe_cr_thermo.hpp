// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fe_cr_thermo.hpp
 * @brief Redlich–Kister Fe–Cr thermodynamics and the code-to-physical scale map.
 *
 * @details
 * Two things the reduced regular-solution model cannot provide on its own: a
 * substitutional free energy in the form thermodynamic assessments actually
 * publish, and a statement of what one code length and one code time mean in
 * nanometres and seconds.
 *
 * ## Free energy
 *
 * A binary substitutional solution with a Redlich–Kister excess term, per mole
 * of atoms, with \f$c\f$ the Cr mole fraction:
 *
 * \f[
 *   G^{\mathrm{ex}} = c(1-c)\bigl[L_0 + L_1(1-2c)\bigr],
 *   \qquad
 *   L_i(T) = L_i^{a} + L_i^{b}\,T .
 * \f]
 *
 * In units of \f$RT\f$, adding ideal mixing,
 *
 * \f[
 *   f(c) = c(1-c)\bigl[\ell_0 + \ell_1(1-2c)\bigr]
 *          + c\ln c + (1-c)\ln(1-c),
 *   \qquad \ell_i = L_i/(RT).
 * \f]
 *
 * With \f$\ell_1 = 0\f$ this is exactly the regular-solution model the app
 * already shipped, with \f$\ell_0 = \omega\f$ — so the reduced verifier is the
 * degenerate case of this one, not a separate code path.
 *
 * ## Provenance
 *
 * The shipped coefficients are the classical bcc Cr–Fe interaction of
 * Andersson and Sundman, *Thermodynamic properties of the Cr–Fe system*,
 * CALPHAD **11** (1987) 83–92:
 *
 * \f[
 *   L_0 = 20500 - 9.68\,T \ \mathrm{J/mol}, \qquad L_1 = 0 .
 * \f]
 *
 * @warning These coefficients, the molar volume and the kinetic constants are
 * **representative, not verified against the primary sources digit by digit**.
 * Treat results as semi-quantitative and check the numbers before publishing
 * anything calibrated. Everything here is a JSON parameter precisely so that a
 * better assessment can be substituted without touching code.
 *
 * The magnetic contribution to the bcc Cr–Fe Gibbs energy is **not** included.
 * It matters near the Curie temperature and shifts the miscibility gap; this
 * is one reason the model is labelled reduced rather than quantitative.
 */

#include <cmath>

#include <openpfc/kernel/data/host_device.hpp>

namespace cahn_hilliard {

/// Gas constant, J/(mol K).
inline constexpr double kGasConstant = 8.314462618;

/**
 * @brief Redlich–Kister interaction coefficients in J/mol, linear in @p T.
 */
struct RedlichKister {
  double L0_a{20500.0}; ///< \f$L_0\f$ constant term (J/mol)
  double L0_b{-9.68};   ///< \f$L_0\f$ temperature slope (J/(mol K))
  double L1_a{0.0};     ///< \f$L_1\f$ constant term (J/mol)
  double L1_b{0.0};     ///< \f$L_1\f$ temperature slope (J/(mol K))

  [[nodiscard]] double L0(double T) const { return L0_a + L0_b * T; }
  [[nodiscard]] double L1(double T) const { return L1_a + L1_b * T; }

  /// \f$\ell_0 = L_0/(RT)\f$, the coefficient the solver actually uses.
  [[nodiscard]] double l0_nd(double T) const {
    return (T > 0.0) ? L0(T) / (kGasConstant * T) : 0.0;
  }
  [[nodiscard]] double l1_nd(double T) const {
    return (T > 0.0) ? L1(T) / (kGasConstant * T) : 0.0;
  }
};

/**
 * @brief Second derivative of the molar free energy in \f$RT\f$ units.
 *
 * \f$f''(c) = -2\ell_0 + \ell_1(12c-6) + \dfrac{1}{c(1-c)}\f$.
 * Negative means the alloy is inside the chemical spinodal at @p c.
 */
[[nodiscard]] OPENPFC_HD inline double f_second_nd(double c, double l0, double l1) {
  return -2.0 * l0 + l1 * (12.0 * c - 6.0) + 1.0 / (c * (1.0 - c));
}

/// Spinodal limits, i.e. the roots of \f$f''=0\f$ bracketing the unstable band.
struct SpinodalRange {
  bool exists{false};
  double lower{0.0}, upper{0.0};
  [[nodiscard]] bool contains(double c) const {
    return exists && c > lower && c < upper;
  }
};

/**
 * @brief Locate the spinodal band by bisection on \f$f''\f$.
 *
 * Solved numerically rather than in closed form so that a nonzero \f$\ell_1\f$
 * (which makes \f$f''\f$ cubic in @p c) needs no special case. The band is
 * whatever the supplied coefficients say it is, which is the point: a preset
 * can then be *checked* against its own thermodynamics instead of assumed to
 * sit inside the gap.
 */
[[nodiscard]] inline SpinodalRange spinodal_range(double l0, double l1) {
  SpinodalRange out;
  // Scan for a sign change; f'' -> +inf at both ends, so an unstable band
  // shows up as a negative interior region.
  constexpr int kSamples = 2001;
  double best_c = 0.5, best_v = f_second_nd(0.5, l0, l1);
  for (int i = 1; i < kSamples; ++i) {
    const double c = static_cast<double>(i) / kSamples;
    const double v = f_second_nd(c, l0, l1);
    if (v < best_v) {
      best_v = v;
      best_c = c;
    }
  }
  if (best_v >= 0.0) return out; // single-phase everywhere at this T
  auto bisect = [&](double a, double b) {
    for (int i = 0; i < 200; ++i) {
      const double m = 0.5 * (a + b);
      if (f_second_nd(m, l0, l1) < 0.0)
        b = m;
      else
        a = m;
    }
    return 0.5 * (a + b);
  };
  out.exists = true;
  out.lower = bisect(1.0e-12, best_c);
  out.upper = 1.0 - bisect(1.0e-12, 1.0 - best_c);
  return out;
}

/**
 * @brief Physical scales that turn code units into nanometres and seconds.
 *
 * @details
 * The solver runs nondimensionally: lengths in units of \f$\ell_c\f$ and times
 * in units of \f$t_c\f$, with the gradient coefficient and mobility set to one
 * in those units. Fixing the energy density scale \f$f_0 = RT/V_m\f$ and the
 * physical gradient coefficient \f$\kappa\f$ gives
 *
 * \f[
 *   \ell_c = \sqrt{\kappa/f_0},
 *   \qquad
 *   t_c = \frac{\ell_c^2}{M f_0} = \frac{\ell_c^2\,|f''_{\mathrm{nd}}|}{D},
 * \f]
 *
 * the second form following from \f$D = M f''_{\mathrm{vol}}
 * = M f_0 f''_{\mathrm{nd}}\f$, with the interdiffusion coefficient supplied
 * by an Arrhenius law \f$D = D_0\exp(-Q/RT)\f$.
 *
 * The time scale therefore depends on the curvature of the free energy at the
 * alloy composition, so it must be evaluated per composition rather than once
 * per material.
 *
 * These scales do not change the trajectory; they label it. Reporting them is
 * what lets a coarsening curve be read in nm and hours rather than in
 * arbitrary units.
 */
struct PhysicalScales {
  double Vm{7.09e-6};     ///< molar volume of bcc Fe (m³/mol)
  double kappa{1.0e-9};   ///< gradient-energy coefficient (J/m)
  double D0{2.0e-5};      ///< Arrhenius prefactor for Cr interdiffusion (m²/s)
  double Q{2.41e5};       ///< activation energy (J/mol)

  /// Energy density scale \f$f_0 = RT/V_m\f$ (J/m³).
  [[nodiscard]] double f0(double T) const { return kGasConstant * T / Vm; }

  /// Interdiffusion coefficient at @p T (m²/s).
  [[nodiscard]] double D(double T) const {
    return D0 * std::exp(-Q / (kGasConstant * T));
  }

  /// Code length unit in metres.
  [[nodiscard]] double length_m(double T) const {
    return std::sqrt(kappa / f0(T));
  }
  [[nodiscard]] double length_nm(double T) const { return 1.0e9 * length_m(T); }

  /**
   * @brief Code time unit in seconds at a given free-energy curvature.
   * @param T       temperature (K)
   * @param fpp_nd  \f$f''(c_0)\f$ in \f$RT\f$ units; its magnitude is used
   */
  [[nodiscard]] double time_s(double T, double fpp_nd) const {
    const double l = length_m(T);
    const double d = D(T);
    return (d > 0.0) ? l * l * std::abs(fpp_nd) / d : 0.0;
  }
  [[nodiscard]] double time_hours(double T, double fpp_nd) const {
    return time_s(T, fpp_nd) / 3600.0;
  }
};

} // namespace cahn_hilliard
