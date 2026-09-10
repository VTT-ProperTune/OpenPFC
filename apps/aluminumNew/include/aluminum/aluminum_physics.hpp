// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file aluminum_physics.hpp
 * @brief One aluminum model: params + schema + moving-frame mean-field ETD.
 *
 * @details
 * Consumed by `pfc::sim::SpectralETDSystem<AluminumPhysics, MemorySpace>` on
 * every backend:
 *
 * - `linear_symbol`, `filter_mf`, `correlation_kernel` @f$P(k)@f$ match Gen-1
 *   `Aluminum::prepare_operators`;
 * - `nonlinear_symbol(k) = k` (conserved PFC dynamics);
 * - `pointwise()` returns the device-capable `AluminumPointwise`
 *   (`aluminum_pointwise.hpp`): @f$N(\psi,\psi_{\mathrm{MF}},P*\psi,
 *   T_{\mathrm{var}}(x,t))@f$ plus the free-energy density observable.
 *
 * No backend classes, no k-loops, no hand-written kernels.
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

#include <aluminum/aluminum_pointwise.hpp>

namespace aluminum {

/**
 * @brief Public JSON-shaped values for `ParameterSchema` (`aluminumNew.json`).
 *
 * Every field here is **read by the model**. That was not true before 0.2:
 * nine of the twenty-five declared fields were `required` and consumed
 * nowhere, which meant a configuration could not be read as a statement of
 * what the run would do. They were resolved one of two ways.
 *
 * *Implemented* — `T_min` / `T_max` now clamp the temperature (see
 * `AluminumPointwise::clamp_temperature`). The pair names an obvious bound
 * on a quantity the model already computes, the shipped values bracket
 * `T_const = 980`, and the moving-frame profile `T_const + G(x - x0 - Vt)`
 * is otherwise unbounded along a 1393-unit domain. Every shipped preset has
 * `G_grid = 0`, so the clamp is a no-op on them and no pinned result moved.
 *
 * *Removed* — the other seven named nothing this model computes:
 * - `alpha_farTol`, `alpha_highOrd` parametrise **tungsten's** `C2`
 *   construction. The FCC dual-Gaussian peak here (`fcc_correlation_peak`)
 *   has no far-field tolerance and no higher-order exponent to set.
 * - `shift_u`, `shift_s` are the vapour shift that *produced* the `p*_bar` /
 *   `q*_bar` coefficients. The `_bar` names say the shift is already
 *   applied; supplying it again suggested the app re-derived them.
 * - `n0`, `n_sol`, `n_vap` are real quantities, but they act in the
 *   `initial_conditions` and `boundary_conditions` blocks, which carry their
 *   own copies (`constant.n0`, `seed_grid_fcc.rho`, `fixed.rho_low/high`).
 *   A second copy under `model.params` that nothing reads is worse than an
 *   unused parameter: it can disagree with the one that acts.
 *
 * Unknown keys are ignored by `ParameterSchema`, so inputs that still carry
 * the removed seven load unchanged.
 */
struct AluminumSchemaValues {
  double T0{89285.0};
  double Bx{0.817900686921996};
  double T_const{980.0};
  /** Upper clamp on `T_const + T_var`; must exceed `T_min`. */
  double T_max{1280.0};
  /** Lower clamp on `T_const + T_var`; must be below `T_max`. */
  double T_min{780.0};
  double G_grid{0.0};
  double V_grid{0.0};
  double x_initial{130.0};
  double alpha{0.20};
  double lambda{0.22};
  double stabP{0.0};
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

/**
 * @brief Cross-field validation `ParameterSchema` cannot express.
 *
 * The schema checks one field at a time, so it can bound `T_min` but not
 * relate it to `T_max` or to `T_const`. Since 0.2 those bounds actually
 * clamp the temperature, an inverted or non-bracketing pair silently
 * freezes the whole domain at one end of the range rather than being
 * harmlessly ignored — so it has to be rejected here.
 *
 * @throws std::invalid_argument if `T_min < T_max` does not hold, or if
 *         `T_const` sits outside `[T_min, T_max]`.
 */
inline void validate_temperature_window(const AluminumSchemaValues &v) {
  if (!(v.T_min < v.T_max)) {
    throw std::invalid_argument(
        "Invalid configuration: T_min (" + std::to_string(v.T_min) +
        ") must be strictly below T_max (" + std::to_string(v.T_max) +
        "); they clamp the temperature field.");
  }
  if (v.T_const < v.T_min || v.T_const > v.T_max) {
    throw std::invalid_argument(
        "Invalid configuration: T_const (" + std::to_string(v.T_const) +
        ") is outside the clamp window [" + std::to_string(v.T_min) + ", " +
        std::to_string(v.T_max) +
        "]; the whole domain would sit pinned at one end of it.");
  }
}

inline void apply_schema_values(const AluminumSchemaValues &v,
                                AluminumParams &p) {
  validate_temperature_window(v);
  static_cast<AluminumSchemaValues &>(p) = v;
  p.recompute_derived();
}

inline pfc::sim::ParameterSchema<AluminumSchemaValues> make_aluminum_schema() {
  pfc::sim::ParameterSchema<AluminumSchemaValues> s;
  s.model_name("Aluminum")
      .real(&AluminumSchemaValues::T0, {.name = "T0", .description = "reference temperature", .required = true, .min = 0.0})
      .real(&AluminumSchemaValues::Bx, {.name = "Bx", .description = "peak coefficient", .required = true})
      .real(&AluminumSchemaValues::T_const, {.name = "T_const", .description = "constant temperature", .required = true})
      .real(&AluminumSchemaValues::T_max, {.name = "T_max", .description = "upper clamp on T_const + T_var", .required = true})
      .real(&AluminumSchemaValues::T_min, {.name = "T_min", .description = "lower clamp on T_const + T_var", .required = true})
      .real(&AluminumSchemaValues::G_grid, {.name = "G_grid", .description = "thermal gradient", .required = true})
      .real(&AluminumSchemaValues::V_grid, {.name = "V_grid", .description = "frame velocity", .required = true})
      .real(&AluminumSchemaValues::x_initial, {.name = "x_initial", .description = "initial front position", .required = true})
      .real(&AluminumSchemaValues::alpha, {.name = "alpha", .description = "FCC correlation peak width", .required = true})
      .real(&AluminumSchemaValues::lambda, {.name = "lambda", .description = "mean-field filter strength", .required = true})
      .real(&AluminumSchemaValues::stabP, {.name = "stabP", .description = "ETD stabilization", .required = true})
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
 * @tparam RealType    Field element type (default double).
 * @tparam MemorySpace Host or device space for `declare_fields`.
 */
template <class RealType = double, class MemorySpace = pfc::HostSpace>
struct AluminumPhysics {
  using parameters_type = AluminumParams;
  using pointwise_type = AluminumPointwise;

  pfc::Domain domain{};
  pfc::Box3i box{};
  AluminumParams params{};

  static pfc::sim::ParameterSchema<AluminumSchemaValues> schema() {
    return make_aluminum_schema();
  }

  /**
   * @brief JSON `model.params` + geometry → physics (for `SpectralETDSession`).
   *
   * The parse is unconditional. It used to be guarded by
   * `!params_json.is_null() && !params_json.empty()`, which meant
   * `"params": {}` — or no `model.params` block at all — quietly took the
   * C++ member initialisers while the schema declared every field
   * `required`. For a calibrated material model that is the wrong default in
   * both directions: a run's parameters were not recoverable from its input
   * file, and the struct defaults are an uncited second copy of coefficients
   * the repository cites no source for. `required` now means required, and
   * an empty block reports all of the missing fields at once.
   *
   * @throws std::invalid_argument listing every missing or invalid field.
   */
  static AluminumPhysics from_json(const nlohmann::json &params_json,
                                   const pfc::Domain &domain_in,
                                   const pfc::Box3i &box_in) {
    AluminumPhysics p;
    p.domain = domain_in;
    p.box = box_in;
    apply_aluminum_json(params_json, p.params);
    return p;
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

  /// PFC dynamics are conserved: \f$\hat N\f$ enters as \f$k_{\mathrm{lap}}\hat N\f$.
  [[nodiscard]] double nonlinear_symbol(double k_laplacian) const {
    return k_laplacian;
  }

  [[nodiscard]] AluminumPointwise pointwise() const {
    const auto size = pfc::domain::get_size(domain);
    const auto spacing = pfc::domain::get_spacing(domain);
    return {.p3_bar = params.p3_bar,
            .p4_bar = params.p4_bar,
            .p2_bar = params.p2_bar,
            .q2_bar = params.q2_bar,
            .q21_bar = params.q21_bar,
            .q30_bar = params.q30_bar,
            .q31_bar = params.q31_bar,
            .q4_bar = params.q4_bar,
            .T0 = params.T0,
            .T_const = params.T_const,
            .T_min = params.T_min,
            .T_max = params.T_max,
            .stabP = params.stabP,
            .G_grid = params.G_grid,
            .V_grid = params.V_grid,
            .x_initial = params.x_initial,
            .front_x = params.m_xpos,
            .length_x = static_cast<double>(size[0]) * spacing[0]};
  }

  // ---- host conveniences (same math as the pointwise functor) -------------
  [[nodiscard]] double temperature_variation(double x, double t) const {
    return pointwise().temperature_variation(x, t);
  }
  [[nodiscard]] double nonlinearity(double psi, double psi_mf, double p_star,
                                    double T_var) const {
    return pointwise().nonlinearity(psi, psi_mf, p_star, T_var);
  }
  [[nodiscard]] double free_energy_density(double psi, double psi_mf,
                                           double p_star, double T_var) const {
    return pointwise().free_energy_density(psi, psi_mf, p_star, T_var);
  }
};

static_assert(pfc::sim::SpectralETDPhysics<AluminumPhysics<>>);
static_assert(pfc::sim::HasMeanFieldFilter<AluminumPhysics<>>);
static_assert(pfc::sim::HasCorrelationKernel<AluminumPhysics<>>);
static_assert(pfc::sim::HasNonlinearSymbol<AluminumPhysics<>>);
static_assert(pfc::sim::HasParameters<AluminumPhysics<>>);

} // namespace aluminum
