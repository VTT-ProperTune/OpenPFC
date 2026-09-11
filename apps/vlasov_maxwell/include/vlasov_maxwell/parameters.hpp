// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file parameters.hpp
 * @brief Units, species and grid parameters for the 1D2V Vlasov-Maxwell app.
 *
 * @details
 * ## The system, and what 1D2V keeps of it
 *
 * The full system (issue #84) is, per species `s`,
 *
 *     d_t f_s + v . grad_x f_s + (q_s/m_s)(E + v x B) . grad_v f_s = 0
 *
 * closed by Maxwell with `rho = sum_s q_s int f_s dv` and
 * `J = sum_s q_s int v f_s dv`.
 *
 * In 1D2V -- one spatial coordinate `x`, velocities `(v_x, v_y)`, everything
 * independent of `y` and `z` -- the Maxwell system splits in a way worth
 * writing down once, here, because which components survive is not obvious
 * and getting it wrong is silent:
 *
 *  - `div B = 0` becomes `d_x B_x = 0`, so `B_x` is uniform; periodic `x`
 *    and `B_x(0) = 0` make **`B_x == 0` identically, for free**. It is a
 *    property of the geometry and is *not* evidence about the code.
 *  - `J_z == 0` under the closure `f_s(x, v_x, v_y)`, so the `(E_z, B_y)`
 *    pair is a vacuum wave the plasma cannot drive. Set to zero, it stays
 *    zero.
 *
 * What is left is `E_x`, `E_y`, `B_z` against `f_s(x, v_x, v_y, t)`, and the
 * Lorentz force keeps its cross product,
 *
 *     (v x B)_x = + v_y B_z ,      (v x B)_y = - v_x B_z ,
 *
 * which rotates velocity vectors in the `(v_x, v_y)` plane. That rotation is
 * the whole reason this application is 1D2V and not 1D1V: gyro-motion exists
 * and can be tested (validation stage 3).
 *
 * ## Normalisation
 *
 * Time in `1/omega_pe`, length in the electron skin depth `d_e = c/omega_pe`,
 * velocity in `c`, `E` in `m_e c omega_pe / e`, `B` in `m_e omega_pe / e`,
 * and `f_s` normalised so `int f_s dv = n_s / n_0`. Then `c = eps_0 = 1` and
 * the evolved system is
 *
 *     d_t f_s + v_x d_x f_s
 *       + (sigma_s/mu_s) [ (E_x + v_y B_z) d_vx f_s
 *                        + (E_y - v_x B_z) d_vy f_s ] = 0
 *
 *     d_x E_x = rho                 (Gauss -- a constraint)
 *     d_t E_x = -J_x                (Ampere -- the evolution form)
 *     d_t E_y = -d_x B_z - J_y
 *     d_t B_z = -d_x E_y
 *
 * with `sigma_s = q_s/(-e)` and `mu_s = m_s/m_e`, so electrons are
 * `sigma = -1`, `mu = 1`.
 *
 * The two forms of `E_x` are the sharpest diagnostic in the application.
 * Differentiating Gauss in time and substituting Ampere gives
 * `d_t(d_x E_x - rho) = -(d_x J_x + d_t rho) = 0` by continuity, so **Gauss
 * holds exactly if and only if the discrete scheme conserves charge**. It is
 * therefore reported every sample rather than enforced and forgotten.
 *
 * ## The Debye length, and why it is not the length unit
 *
 * `lambda_D = v_th / omega_pe = (v_th/c) d_e`, so a thermal velocity of
 * `0.1 c` puts ten Debye lengths in a skin depth. Electrostatic benchmarks
 * are quoted in `k lambda_D` and electromagnetic ones in `k d_e`; both
 * appear, and @ref SimParams::k_debye and @ref SimParams::k_skin convert.
 *
 * @see issue #84 for the specification and the validation ladder
 */

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace vlasov {

/// Machine-epsilon-scale floor used where a ratio could divide by zero.
inline constexpr double kTiny = 1.0e-300;

/**
 * @brief One kinetic species.
 *
 * `sigma` and `mu` are the charge and mass in units of `-e` and `m_e`, so an
 * electron is `{-1, 1}` and a proton `{+1, 1836.15267343}`. The charge-to-mass
 * ratio the Vlasov equation actually uses is `sigma/mu`.
 */
struct Species {
  std::string name{"electron"};
  double sigma{-1.0};
  double mu{1.0};
  /// Charge-to-mass ratio in normalised units.
  [[nodiscard]] double qm() const noexcept { return sigma / mu; }
};

/// Proton-to-electron mass ratio (CODATA 2018), for a mobile-ion run.
inline constexpr double kProtonElectronMassRatio = 1836.15267343;

/**
 * @brief Phase-space grid and run control.
 *
 * The phase space is one OpenPFC 3-D `Domain` with axes `(x, v_x, v_y)`.
 * That is the central design claim of this application: a 1D2V kinetic phase
 * space *is* a structured 3-D grid, so `Domain`, `Field`, the halo exchange
 * and the FFT stack apply unmodified.
 */
struct SimParams {
  // ---- grid -------------------------------------------------------------
  /// Cells along `x`. Periodic.
  int nx{64};
  /// Cells along `v_x` and `v_y`. Not periodic; zero inflow.
  int nvx{64};
  int nvy{64};
  /// Spatial period, in skin depths `d_e`.
  double Lx{1.0};
  /// Velocity half-extent, in `c`. The grid spans `[-v_max, +v_max]`.
  double v_max{1.0};

  // ---- time -------------------------------------------------------------
  double dt{0.0};        ///< 0 selects `dt_safety` times the limit.
  double dt_safety{0.2};
  double t_end{10.0};
  int n_sample{100};

  // ---- physics ----------------------------------------------------------
  std::vector<Species> species{Species{}};
  /**
   * @brief Uniform neutralising background charge density.
   *
   * NaN means "choose it so the initial state is exactly neutral", which is
   * what every benchmark here wants and what keeps the `k = 0` mode of Gauss
   * solvable. An explicitly set value is honoured and the resulting net
   * charge is reported rather than silently corrected.
   */
  double rho_background{std::nan("")};
  /**
   * @brief Thermal velocity of the initial state, in `c`.
   *
   * Not used by the evolution -- the Vlasov equation does not know what a
   * temperature is -- but the diagnostics do. The velocity-boundary
   * occupancy the validation ladder requires is "the fraction of particle
   * number within one thermal width of the boundary", and that needs a
   * width. The initial condition sets this; a run that leaves it at zero
   * gets an empty shell and an honest zero rather than a meaningless
   * number.
   */
  double v_thermal{0.0};
  /**
   * @brief Net charge above which a state is not considered neutral.
   *
   * The `k = 0` mode of Gauss is unsolvable for a charged periodic cell, so
   * a non-neutral state is reported rather than silently zeroed. This is
   * the threshold for that report.
   */
  double neutrality_tol{1.0e-12};
  /// Externally imposed, uniform, constant `B_z`. Used by the gyro-motion
  /// stage, where the self-consistent fields are switched off.
  double b_ext{0.0};
  /// Solve the field equations at all. `false` freezes `E` and `B` at their
  /// initial values, which is what stage 3 needs.
  bool self_consistent{true};
  /// Electrostatic reduction: hold `E_y = B_z = 0` and take `E_x` from
  /// Gauss rather than from Ampere. This is the Vlasov-Poisson path and it
  /// is a runtime reduction of the same stepper, not a second code path.
  bool electrostatic{false};

  // ---- numerics ---------------------------------------------------------
  /// Lagrange interpolation order for the semi-Lagrangian velocity shifts.
  /// Odd orders are centred on the departure cell; 5 is the default.
  int interp_order{5};
  /// Apply a Poisson-based divergence correction to `E_x`. Off by default:
  /// the point of the Gauss residual is to be measured, and a correction
  /// that runs silently hides the thing worth reporting.
  bool gauss_correction{false};

  // ---- derived ----------------------------------------------------------
  [[nodiscard]] double dx() const noexcept {
    return Lx / static_cast<double>(nx);
  }
  [[nodiscard]] double dvx() const noexcept {
    return 2.0 * v_max / static_cast<double>(nvx);
  }
  [[nodiscard]] double dvy() const noexcept {
    return 2.0 * v_max / static_cast<double>(nvy);
  }
  /// Cell-centred coordinate of index `i` along `x`.
  [[nodiscard]] double x_of(int i) const noexcept {
    return (static_cast<double>(i) + 0.5) * dx();
  }
  /// Cell-centred `v_x` of index `j`. Symmetric about zero, so with an even
  /// `nvx` no cell sits exactly at `v = 0`.
  [[nodiscard]] double vx_of(int j) const noexcept {
    return -v_max + (static_cast<double>(j) + 0.5) * dvx();
  }
  [[nodiscard]] double vy_of(int k) const noexcept {
    return -v_max + (static_cast<double>(k) + 0.5) * dvy();
  }
  /// Fundamental wave number `2 pi / Lx`, in inverse skin depths.
  [[nodiscard]] double k0() const noexcept {
    return 2.0 * std::acos(-1.0) / Lx;
  }
  /// `k` of mode `m` expressed in inverse Debye lengths, given `v_th / c`.
  [[nodiscard]] double k_debye(int m, double vth_over_c) const noexcept {
    return static_cast<double>(m) * k0() * vth_over_c;
  }
  /// `k` of mode `m` in inverse skin depths.
  [[nodiscard]] double k_skin(int m) const noexcept {
    return static_cast<double>(m) * k0();
  }
  /// Phase-space cells per species. `double` because the product overflows
  /// `int` at resolutions this application is meant to reach.
  [[nodiscard]] double cells() const noexcept {
    return static_cast<double>(nx) * static_cast<double>(nvx) *
           static_cast<double>(nvy);
  }

  /// Throw unless the parameters describe a runnable problem.
  void validate() const {
    if (nx < 4 || nvx < 4 || nvy < 4) {
      throw std::invalid_argument("SimParams: need at least 4 cells per axis");
    }
    if (Lx <= 0.0 || v_max <= 0.0) {
      throw std::invalid_argument("SimParams: Lx and v_max must be positive");
    }
    if (species.empty()) {
      throw std::invalid_argument("SimParams: at least one species");
    }
    if (interp_order < 1 || interp_order > 9) {
      throw std::invalid_argument("SimParams: interp_order must be 1..9");
    }
    for (const auto &s : species) {
      if (s.mu <= 0.0) {
        throw std::invalid_argument("SimParams: species mass must be positive");
      }
    }
  }
};

} // namespace vlasov
