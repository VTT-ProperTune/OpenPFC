// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file aluminum_physics.hpp
 * @brief One aluminum model: params + schema + moving-frame mean-field ETD.
 *
 * Linear symbol, mean-field filter, and correlation kernel @f$P(k)@f$ match
 * Gen-1 `Aluminum::prepare_operators`. Real-space
 * @f$N(\psi,\psi_{\mathrm{MF}},P*\psi,T_{\mathrm{var}})@f$ includes the
 * temperature gradient / moving frame. No backend classes and no k-loops.
 */

#include <cmath>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

namespace aluminum {

/** Public JSON-shaped values for `ParameterSchema` (`aluminumNew.json`). */
struct AluminumSchemaValues {
  double n0{-0.0060};
  double n_sol{-0.036};
  double n_vap{-1.297};
  double T0{89285.0};
  double Bx{0.817900686921996};
  double T_const{980.0};
  double T_max{1280.0};
  double T_min{780.0};
  double G_grid{0.0};
  double V_grid{0.0};
  double x_initial{130.0};
  double alpha{0.20};
  double alpha_farTol{0.001};
  int alpha_highOrd{0};
  double lambda{0.22};
  double stabP{0.0};
  double shift_u{1.0};
  double shift_s{0.0};
  double p2_bar{0.8286531831};
  double p3_bar{-0.04204863};
  double p4_bar{0.007533};
  double q20_bar{0.016531729105214};
  double q21_bar{5.467};
  double q30_bar{1.7152418049986};
  double q31_bar{0.45};
  double q40_bar{0.787482};
};

struct AluminumParams : AluminumSchemaValues {
  double tau_const{0.0};
  double q2_bar{0.0};
  double q2_bar_L{0.0};
  double q3_bar{0.0};
  double q4_bar{0.0};
  double m_xpos{0.0};

  AluminumParams() { recompute_derived(); }

  void recompute_derived() {
    tau_const = T_const / T0;
    q2_bar = q21_bar * tau_const + q20_bar;
    q2_bar_L = q2_bar;
    q3_bar = q31_bar * tau_const + q30_bar;
    q4_bar = q40_bar;
    m_xpos = x_initial;
  }
};

inline void apply_schema_values(const AluminumSchemaValues &v,
                                AluminumParams &p) {
  static_cast<AluminumSchemaValues &>(p) = v;
  p.recompute_derived();
}

inline pfc::sim::ParameterSchema<AluminumSchemaValues> make_aluminum_schema() {
  pfc::sim::ParameterSchema<AluminumSchemaValues> s;
  s.model_name("Aluminum")
      .real(&AluminumSchemaValues::n0, {.name = "n0", .description = "average density", .required = true})
      .real(&AluminumSchemaValues::n_sol, {.name = "n_sol", .description = "solid coexistence density", .required = true})
      .real(&AluminumSchemaValues::n_vap, {.name = "n_vap", .description = "vapor coexistence density", .required = true})
      .real(&AluminumSchemaValues::T0, {.name = "T0", .description = "reference temperature", .required = true, .min = 0.0})
      .real(&AluminumSchemaValues::Bx, {.name = "Bx", .description = "peak coefficient", .required = true})
      .real(&AluminumSchemaValues::T_const, {.name = "T_const", .description = "constant temperature", .required = true})
      .real(&AluminumSchemaValues::T_max, {.name = "T_max", .description = "max temperature", .required = true})
      .real(&AluminumSchemaValues::T_min, {.name = "T_min", .description = "min temperature", .required = true})
      .real(&AluminumSchemaValues::G_grid, {.name = "G_grid", .description = "thermal gradient", .required = true})
      .real(&AluminumSchemaValues::V_grid, {.name = "V_grid", .description = "frame velocity", .required = true})
      .real(&AluminumSchemaValues::x_initial, {.name = "x_initial", .description = "initial front position", .required = true})
      .real(&AluminumSchemaValues::alpha, {.name = "alpha", .description = "C2 peak width", .required = true})
      .real(&AluminumSchemaValues::alpha_farTol, {.name = "alpha_farTol", .description = "k=1 far tolerance", .required = true})
      .integer(&AluminumSchemaValues::alpha_highOrd, {.name = "alpha_highOrd", .description = "higher-order Gaussian power", .required = true})
      .real(&AluminumSchemaValues::lambda, {.name = "lambda", .description = "mean-field filter strength", .required = true})
      .real(&AluminumSchemaValues::stabP, {.name = "stabP", .description = "ETD stabilization", .required = true})
      .real(&AluminumSchemaValues::shift_u, {.name = "shift_u", .description = "vapor shift u", .required = true})
      .real(&AluminumSchemaValues::shift_s, {.name = "shift_s", .description = "vapor shift s", .required = true})
      .real(&AluminumSchemaValues::p2_bar, {.name = "p2_bar", .description = "shifted p2", .required = true})
      .real(&AluminumSchemaValues::p3_bar, {.name = "p3_bar", .description = "shifted p3", .required = true})
      .real(&AluminumSchemaValues::p4_bar, {.name = "p4_bar", .description = "shifted p4", .required = true})
      .real(&AluminumSchemaValues::q20_bar, {.name = "q20_bar", .description = "shifted q20", .required = true})
      .real(&AluminumSchemaValues::q21_bar, {.name = "q21_bar", .description = "shifted q21", .required = true})
      .real(&AluminumSchemaValues::q30_bar, {.name = "q30_bar", .description = "shifted q30", .required = true})
      .real(&AluminumSchemaValues::q31_bar, {.name = "q31_bar", .description = "shifted q31", .required = true})
      .real(&AluminumSchemaValues::q40_bar, {.name = "q40_bar", .description = "shifted q40", .required = true});
  return s;
}

inline void apply_aluminum_json(const nlohmann::json &j, AluminumParams &p) {
  apply_schema_values(make_aluminum_schema().parse(j), p);
}

/** FCC dual-Gaussian peak used by Gen-1 `P_F` (not tungsten's C2). */
[[nodiscard]] inline double fcc_correlation_peak(double k_laplacian,
                                                 double alpha) {
  const double alpha2 = 2.0 * alpha * alpha;
  const double k_abs = std::sqrt(-k_laplacian);
  const double k_wave = k_abs - 1.0;
  const double kp = k_abs - 2.0 / std::sqrt(3.0);
  const double g1 = std::exp(-(k_wave * k_wave) / alpha2);
  const double gp1 = std::exp(-(kp * kp) / alpha2);
  return (g1 > gp1) ? g1 : gp1;
}

/**
 * @tparam RealType   Field element type (default double).
 * @tparam MemorySpace Host or device space for `declare_fields`.
 */
template <class RealType = double, class MemorySpace = pfc::HostSpace>
struct AluminumPhysics {
  using parameters_type = AluminumParams;

  pfc::Domain domain{};
  pfc::Box3i box{};
  AluminumParams params{};

  static pfc::sim::ParameterSchema<AluminumSchemaValues> schema() {
    return make_aluminum_schema();
  }

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<RealType, MemorySpace>(state, "psi", domain,
                                                        box, 0);
  }

  [[nodiscard]] double filter_mf(double k_laplacian) const {
    const double lambda2 = 2.0 * params.lambda * params.lambda;
    return std::exp(k_laplacian / lambda2);
  }

  [[nodiscard]] double correlation_kernel(double k_laplacian) const {
    const double peak = fcc_correlation_peak(k_laplacian, params.alpha);
    return params.Bx * std::exp(-params.tau_const) * peak;
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    const double p_f = correlation_kernel(k_laplacian);
    const double op_ck = params.stabP + params.p2_bar - p_f +
                         params.q2_bar_L * filter_mf(k_laplacian);
    return k_laplacian * op_ck;
  }

  [[nodiscard]] double temperature_variation(double x, double t) const {
    const auto size = pfc::domain::get_size(domain);
    const auto spacing = pfc::domain::get_spacing(domain);
    const double length = static_cast<double>(size[0]) * spacing[0];
    const double fullruns = std::floor(params.m_xpos / length) * length;
    const double steppoint = std::fmod(params.m_xpos, length);
    const double dist = x + fullruns - (x > steppoint) * length;
    return params.G_grid *
           (dist - params.x_initial - params.V_grid * t);
  }

  [[nodiscard]] double nonlinearity(double psi, double psi_mf, double p_star,
                                    double T_var) const {
    const double q2_bar_n = params.q21_bar * T_var / params.T0;
    const double q3_bar_n =
        params.q31_bar * (params.T_const + T_var) / params.T0 +
        params.q30_bar;
    const double kernel_term =
        -(1.0 - std::exp(-T_var / params.T0)) * p_star;
    const double u2 = psi * psi;
    const double v2 = psi_mf * psi_mf;
    double n = params.p3_bar * u2 + params.p4_bar * u2 * psi +
               q2_bar_n * psi_mf + q3_bar_n * v2 +
               params.q4_bar * v2 * psi_mf - kernel_term;
    if (params.stabP != 0.0) {
      n -= params.stabP * psi;
    }
    return n;
  }

  [[nodiscard]] double free_energy_density(double psi, double psi_mf,
                                           double p_star, double T_var) const {
    const double q2_bar_n = params.q21_bar * T_var / params.T0;
    const double q3_bar_n =
        params.q31_bar * (params.T_const + T_var) / params.T0 +
        params.q30_bar;
    const double kernel_term =
        -(1.0 - std::exp(-T_var / params.T0)) * p_star;
    const double u = psi;
    const double v = psi_mf;
    return params.p3_bar * u * u * u / 3.0 +
           params.p4_bar * u * u * u * u / 4.0 + q2_bar_n * u * v / 2.0 +
           q3_bar_n * u * v * v / 3.0 +
           params.q4_bar * u * v * v * v / 4.0 -
           u * kernel_term * u / 2.0 - u * p_star / 2.0 +
           params.p2_bar * u * u / 2.0 + params.q2_bar * u * v / 2.0;
  }
};

static_assert(pfc::sim::MovingFrameMeanFieldETDPhysics<AluminumPhysics<>>);
static_assert(pfc::sim::HasParameters<AluminumPhysics<>>);

} // namespace aluminum
