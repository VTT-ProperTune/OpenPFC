// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef TUNGSTEN_PARAMS_HPP
#define TUNGSTEN_PARAMS_HPP

#include <cmath>

/**
 * @file tungsten_params.hpp
 * @brief Tungsten model parameters structure
 *
 * @details
 * This structure contains all user-configurable parameters for the Tungsten PFC
 * model. Only base parameters are stored; derived parameters are calculated
 * on-the-fly when accessed through getter methods. This ensures that derived
 * parameters are always consistent with base parameters, regardless of the
 * order in which parameters are set.
 *
 * @note All derived parameters (tau, p2_bar, p3_bar, p4_bar, q20_bar, q21_bar,
 * q30_bar, q31_bar, q40_bar, q2_bar, q3_bar, q4_bar) are computed
 * automatically from base parameters and should not be set directly.
 *
 * @example
 * @code
 * TungstenParams params;
 * params.set_n0(-0.10);
 * params.set_T(3300.0);
 * params.set_T0(156000.0);
 * double tau = params.get_tau(); // Automatically calculated as T/T0
 * @endcode
 */
struct TungstenParams {
private:
  // Base parameters (user-configurable)
  double m_n0 = -0.10;           ///< Average density of the metastable fluid
  double m_n_sol = -0.047;       ///< Bulk density at solid coexistence
  double m_n_vap = -0.464;       ///< Bulk density at vapor coexistence
  double m_T = 3300.0;           ///< Effective temperature (K)
  double m_T0 = 156000.0;        ///< Reference temperature (K)
  double m_Bx = 0.8582;          ///< Temperature-dependent coefficient
  double m_alpha = 0.50;         ///< Width of C2's peak
  double m_alpha_farTol = 0.001; ///< Tolerance for k=1 peak effect on k=0
  int m_alpha_highOrd =
      4; ///< Power of higher-order Gaussian component (multiple of 2)
  double m_lambda = 0.22; ///< Strength of meanfield filter (avoid >0.28)
  double m_stabP = 0.2; ///< Numerical stability parameter for exponential integrator
  double m_shift_u = 0.3341; ///< Vapor-model shift parameter u
  double m_shift_s = 0.1898; ///< Vapor-model shift parameter s
  double m_p2 = 1.0;         ///< Vapor-model polynomial coefficient p2
  double m_p3 = -0.5;        ///< Vapor-model polynomial coefficient p3
  double m_p4 = 0.333333333; ///< Vapor-model polynomial coefficient p4
  double m_q20 = -0.0037;    ///< Vapor-model coefficient q20
  double m_q21 = 1.0;        ///< Vapor-model coefficient q21
  double m_q30 = -12.4567;   ///< Vapor-model coefficient q30
  double m_q31 = 20.0;       ///< Vapor-model coefficient q31
  double m_q40 = 45.0;       ///< Vapor-model coefficient q40

public:
  /**
   * @brief Default constructor
   *
   * Initializes all parameters to their default values from
   * tungsten_single_seed.json.
   */
  TungstenParams() = default;

  // ============================================================================
  // Setters for base parameters
  // ============================================================================

  /**
   * @brief Set average density of the metastable fluid
   * @param n0 Average density value
   */
  void set_n0(double n0) { m_n0 = n0; }

  /**
   * @brief Set bulk density at solid coexistence
   * @param n_sol Solid density value
   * @note Should be updated according to phase diagram when T is changed
   */
  void set_n_sol(double n_sol) { m_n_sol = n_sol; }

  /**
   * @brief Set bulk density at vapor coexistence
   * @param n_vap Vapor density value
   * @note Should be updated according to phase diagram when T is changed
   */
  void set_n_vap(double n_vap) { m_n_vap = n_vap; }

  /**
   * @brief Set effective temperature
   * @param T Temperature in Kelvin
   * @note Remember to update n_sol and n_vap according to phase diagram
   */
  void set_T(double T) { m_T = T; }

  /**
   * @brief Set reference temperature
   * @param T0 Reference temperature in Kelvin
   */
  void set_T0(double T0) { m_T0 = T0; }

  /**
   * @brief Set temperature-dependent coefficient
   * @param Bx Coefficient value
   */
  void set_Bx(double Bx) { m_Bx = Bx; }

  /**
   * @brief Set width of C2's peak
   * @param alpha Peak width
   */
  void set_alpha(double alpha) { m_alpha = alpha; }

  /**
   * @brief Set tolerance for k=1 peak effect on k=0
   * @param alpha_farTol Tolerance value
   */
  void set_alpha_farTol(double alpha_farTol) { m_alpha_farTol = alpha_farTol; }

  /**
   * @brief Set power of higher-order Gaussian component
   * @param alpha_highOrd Power value (should be a multiple of 2, or 0 to disable)
   */
  void set_alpha_highOrd(int alpha_highOrd) { m_alpha_highOrd = alpha_highOrd; }

  /**
   * @brief Set strength of meanfield filter
   * @param lambda Filter strength (avoid values higher than ~0.28)
   */
  void set_lambda(double lambda) { m_lambda = lambda; }

  /**
   * @brief Set numerical stability parameter
   * @param stabP Stability parameter for exponential integrator method
   */
  void set_stabP(double stabP) { m_stabP = stabP; }

  /**
   * @brief Set vapor-model shift parameter u
   * @param shift_u Shift parameter value
   */
  void set_shift_u(double shift_u) { m_shift_u = shift_u; }

  /**
   * @brief Set vapor-model shift parameter s
   * @param shift_s Shift parameter value
   */
  void set_shift_s(double shift_s) { m_shift_s = shift_s; }

  /**
   * @brief Set vapor-model polynomial coefficient p2
   * @param p2 Coefficient value
   */
  void set_p2(double p2) { m_p2 = p2; }

  /**
   * @brief Set vapor-model polynomial coefficient p3
   * @param p3 Coefficient value
   */
  void set_p3(double p3) { m_p3 = p3; }

  /**
   * @brief Set vapor-model polynomial coefficient p4
   * @param p4 Coefficient value
   */
  void set_p4(double p4) { m_p4 = p4; }

  /**
   * @brief Set vapor-model coefficient q20
   * @param q20 Coefficient value
   */
  void set_q20(double q20) { m_q20 = q20; }

  /**
   * @brief Set vapor-model coefficient q21
   * @param q21 Coefficient value
   */
  void set_q21(double q21) { m_q21 = q21; }

  /**
   * @brief Set vapor-model coefficient q30
   * @param q30 Coefficient value
   */
  void set_q30(double q30) { m_q30 = q30; }

  /**
   * @brief Set vapor-model coefficient q31
   * @param q31 Coefficient value
   */
  void set_q31(double q31) { m_q31 = q31; }

  /**
   * @brief Set vapor-model coefficient q40
   * @param q40 Coefficient value
   */
  void set_q40(double q40) { m_q40 = q40; }

  // ============================================================================
  // Getters for base parameters
  // ============================================================================

  /**
   * @brief Get average density of the metastable fluid
   * @return Average density value
   */
  double get_n0() const { return m_n0; }

  /**
   * @brief Get bulk density at solid coexistence
   * @return Solid density value
   */
  double get_n_sol() const { return m_n_sol; }

  /**
   * @brief Get bulk density at vapor coexistence
   * @return Vapor density value
   */
  double get_n_vap() const { return m_n_vap; }

  /**
   * @brief Get effective temperature
   * @return Temperature in Kelvin
   */
  double get_T() const { return m_T; }

  /**
   * @brief Get reference temperature
   * @return Reference temperature in Kelvin
   */
  double get_T0() const { return m_T0; }

  /**
   * @brief Get temperature-dependent coefficient
   * @return Coefficient value
   */
  double get_Bx() const { return m_Bx; }

  /**
   * @brief Get width of C2's peak
   * @return Peak width
   */
  double get_alpha() const { return m_alpha; }

  /**
   * @brief Get tolerance for k=1 peak effect on k=0
   * @return Tolerance value
   */
  double get_alpha_farTol() const { return m_alpha_farTol; }

  /**
   * @brief Get power of higher-order Gaussian component
   * @return Power value
   */
  int get_alpha_highOrd() const { return m_alpha_highOrd; }

  /**
   * @brief Get strength of meanfield filter
   * @return Filter strength
   */
  double get_lambda() const { return m_lambda; }

  /**
   * @brief Get numerical stability parameter
   * @return Stability parameter value
   */
  double get_stabP() const { return m_stabP; }

  /**
   * @brief Get vapor-model shift parameter u
   * @return Shift parameter value
   */
  double get_shift_u() const { return m_shift_u; }

  /**
   * @brief Get vapor-model shift parameter s
   * @return Shift parameter value
   */
  double get_shift_s() const { return m_shift_s; }

  /**
   * @brief Get vapor-model polynomial coefficient p2
   * @return Coefficient value
   */
  double get_p2() const { return m_p2; }

  /**
   * @brief Get vapor-model polynomial coefficient p3
   * @return Coefficient value
   */
  double get_p3() const { return m_p3; }

  /**
   * @brief Get vapor-model polynomial coefficient p4
   * @return Coefficient value
   */
  double get_p4() const { return m_p4; }

  /**
   * @brief Get vapor-model coefficient q20
   * @return Coefficient value
   */
  double get_q20() const { return m_q20; }

  /**
   * @brief Get vapor-model coefficient q21
   * @return Coefficient value
   */
  double get_q21() const { return m_q21; }

  /**
   * @brief Get vapor-model coefficient q30
   * @return Coefficient value
   */
  double get_q30() const { return m_q30; }

  /**
   * @brief Get vapor-model coefficient q31
   * @return Coefficient value
   */
  double get_q31() const { return m_q31; }

  /**
   * @brief Get vapor-model coefficient q40
   * @return Coefficient value
   */
  double get_q40() const { return m_q40; }

  // ============================================================================
  // Getters for derived parameters (calculated on-the-fly)
  // ============================================================================

  /**
   * @brief Get dimensionless temperature ratio
   * @return Calculated as T / T0
   */
  double get_tau() const { return m_T / m_T0; }

  /**
   * @brief Get derived vapor-model parameter p2_bar
   * @return Calculated as p2 + 2*shift_s*p3 + 3*shift_s^2*p4
   */
  double get_p2_bar() const {
    return m_p2 + 2 * m_shift_s * m_p3 + 3 * pow(m_shift_s, 2) * m_p4;
  }

  /**
   * @brief Get derived vapor-model parameter p3_bar
   * @return Calculated as shift_u * (p3 + 3*shift_s*p4)
   */
  double get_p3_bar() const { return m_shift_u * (m_p3 + 3 * m_shift_s * m_p4); }

  /**
   * @brief Get derived vapor-model parameter p4_bar
   * @return Calculated as shift_u^2 * p4
   */
  double get_p4_bar() const { return pow(m_shift_u, 2) * m_p4; }

  /**
   * @brief Get derived vapor-model parameter q20_bar
   * @return Calculated as q20 + 2*shift_s*q30 + 3*shift_s^2*q40
   */
  double get_q20_bar() const {
    return m_q20 + 2.0 * m_shift_s * m_q30 + 3.0 * pow(m_shift_s, 2) * m_q40;
  }

  /**
   * @brief Get derived vapor-model parameter q21_bar
   * @return Calculated as q21 + 2*shift_s*q31
   */
  double get_q21_bar() const { return m_q21 + 2.0 * m_shift_s * m_q31; }

  /**
   * @brief Get derived vapor-model parameter q30_bar
   * @return Calculated as shift_u * (q30 + 3*shift_s*q40)
   */
  double get_q30_bar() const {
    return m_shift_u * (m_q30 + 3.0 * m_shift_s * m_q40);
  }

  /**
   * @brief Get derived vapor-model parameter q31_bar
   * @return Calculated as shift_u * q31
   */
  double get_q31_bar() const { return m_shift_u * m_q31; }

  /**
   * @brief Get derived vapor-model parameter q40_bar
   * @return Calculated as shift_u^2 * q40
   */
  double get_q40_bar() const { return pow(m_shift_u, 2) * m_q40; }

  /**
   * @brief Get derived vapor-model parameter q2_bar
   * @return Calculated as q21_bar * tau + q20_bar
   */
  double get_q2_bar() const { return get_q21_bar() * get_tau() + get_q20_bar(); }

  /**
   * @brief Get derived vapor-model parameter q3_bar
   * @return Calculated as q31_bar * tau + q30_bar
   */
  double get_q3_bar() const { return get_q31_bar() * get_tau() + get_q30_bar(); }

  /**
   * @brief Get derived vapor-model parameter q4_bar
   * @return Calculated as q40_bar
   */
  double get_q4_bar() const { return get_q40_bar(); }
};

#endif // TUNGSTEN_PARAMS_HPP
