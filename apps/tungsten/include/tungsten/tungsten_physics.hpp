// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file tungsten_physics.hpp
 * @brief One tungsten model: params + schema + spectral mean-field ETD physics.
 *
 * @details
 * Consumed by `pfc::sim::SpectralETDSystem<TungstenPhysics, MemorySpace>` on
 * every backend:
 *
 * - `linear_symbol(k)` and `filter_mf(k)` come from `physics_for_mode`;
 * - `nonlinear_symbol(k) = k` (PFC form \f$\partial_t\hat\psi =
 *   k(C\hat\psi + \hat N)\f$);
 * - `pointwise()` returns the device-capable `TungstenPointwise`
 *   (`tungsten_pointwise.hpp`), which includes the ETD stabilisation.
 *
 * No backend classes, no k-loops, no hand-written kernels.
 */

#include <nlohmann/json.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <tungsten/common/tungsten_params.hpp>
#include <tungsten/common/tungsten_spectral.hpp>
#include <tungsten/tungsten_pointwise.hpp>

namespace tungsten {

/** Public JSON-shaped values for `ParameterSchema` (TungstenParams is private). */
struct TungstenSchemaValues {
  double n0{-0.10};
  double n_sol{-0.047};
  double n_vap{-0.464};
  double T{3300.0};
  double T0{156000.0};
  double Bx{0.8582};
  double alpha{0.50};
  double alpha_farTol{0.001};
  int alpha_highOrd{4};
  double lambda{0.22};
  double stabP{0.2};
  double shift_u{0.3341};
  double shift_s{0.1898};
  double p2{1.0};
  double p3{-0.5};
  double p4{0.333333333};
  double q20{-0.0037};
  double q21{1.0};
  double q30{-12.4567};
  double q31{20.0};
  double q40{45.0};
};

inline void apply_schema_values(const TungstenSchemaValues &v, TungstenParams &p) {
  p.set_n0(v.n0);
  p.set_n_sol(v.n_sol);
  p.set_n_vap(v.n_vap);
  p.set_T(v.T);
  p.set_T0(v.T0);
  p.set_Bx(v.Bx);
  p.set_alpha(v.alpha);
  p.set_alpha_farTol(v.alpha_farTol);
  p.set_alpha_highOrd(v.alpha_highOrd);
  p.set_lambda(v.lambda);
  p.set_stabP(v.stabP);
  p.set_shift_u(v.shift_u);
  p.set_shift_s(v.shift_s);
  p.set_p2(v.p2);
  p.set_p3(v.p3);
  p.set_p4(v.p4);
  p.set_q20(v.q20);
  p.set_q21(v.q21);
  p.set_q30(v.q30);
  p.set_q31(v.q31);
  p.set_q40(v.q40);
}

inline pfc::sim::ParameterSchema<TungstenSchemaValues> make_tungsten_schema() {
  pfc::sim::ParameterSchema<TungstenSchemaValues> s;
  s.model_name("Tungsten")
      .real(&TungstenSchemaValues::n0,
            {.name = "n0", .description = "average density", .required = true})
      .real(&TungstenSchemaValues::n_sol,
            {.name = "n_sol",
             .description = "solid coexistence density",
             .required = true})
      .real(&TungstenSchemaValues::n_vap,
            {.name = "n_vap",
             .description = "vapor coexistence density",
             .required = true})
      .real(&TungstenSchemaValues::T, {.name = "T",
                                       .description = "effective temperature",
                                       .required = true,
                                       .min = 0.0})
      .real(&TungstenSchemaValues::T0, {.name = "T0",
                                        .description = "reference temperature",
                                        .required = true,
                                        .min = 0.0})
      .real(&TungstenSchemaValues::Bx,
            {.name = "Bx", .description = "peak coefficient", .required = true})
      .real(&TungstenSchemaValues::alpha,
            {.name = "alpha", .description = "C2 peak width", .required = true})
      .real(&TungstenSchemaValues::alpha_farTol, {.name = "alpha_farTol",
                                                  .description = "k=1 far tolerance",
                                                  .required = true})
      .integer(&TungstenSchemaValues::alpha_highOrd,
               {.name = "alpha_highOrd",
                .description = "higher-order Gaussian power",
                .required = true})
      .real(&TungstenSchemaValues::lambda,
            {.name = "lambda",
             .description = "mean-field filter strength",
             .required = true})
      .real(&TungstenSchemaValues::stabP,
            {.name = "stabP", .description = "ETD stabilization", .required = true})
      .real(&TungstenSchemaValues::shift_u,
            {.name = "shift_u", .description = "vapor shift u", .required = true})
      .real(&TungstenSchemaValues::shift_s,
            {.name = "shift_s", .description = "vapor shift s", .required = true})
      .real(&TungstenSchemaValues::p2,
            {.name = "p2", .description = "polynomial p2", .required = true})
      .real(&TungstenSchemaValues::p3,
            {.name = "p3", .description = "polynomial p3", .required = true})
      .real(&TungstenSchemaValues::p4,
            {.name = "p4", .description = "polynomial p4", .required = true})
      .real(&TungstenSchemaValues::q20,
            {.name = "q20", .description = "q20", .required = true})
      .real(&TungstenSchemaValues::q21,
            {.name = "q21", .description = "q21", .required = true})
      .real(&TungstenSchemaValues::q30,
            {.name = "q30", .description = "q30", .required = true})
      .real(&TungstenSchemaValues::q31,
            {.name = "q31", .description = "q31", .required = true})
      .real(&TungstenSchemaValues::q40,
            {.name = "q40", .description = "q40", .required = true});
  return s;
}

inline void apply_tungsten_json(const nlohmann::json &j, TungstenParams &p) {
  apply_schema_values(make_tungsten_schema().parse(j), p);
}

/**
 * @tparam RealType    Field element type (default double).
 * @tparam MemorySpace Host or device space for `declare_fields`.
 */
template <class RealType = double, class MemorySpace = pfc::HostSpace>
struct TungstenPhysics {
  using parameters_type = TungstenParams;
  using pointwise_type = TungstenPointwise;

  pfc::Domain domain{};
  pfc::Box3i box{};
  TungstenParams params{};

  static pfc::sim::ParameterSchema<TungstenSchemaValues> schema() {
    return make_tungsten_schema();
  }

  /// JSON `model.params` + geometry → physics (used by `SpectralETDSession`).
  static TungstenPhysics from_json(const nlohmann::json &params_json,
                                   const pfc::Domain &domain_in,
                                   const pfc::Box3i &box_in) {
    TungstenPhysics p;
    p.domain = domain_in;
    p.box = box_in;
    if (!params_json.is_null() && !params_json.empty()) {
      apply_tungsten_json(params_json, p.params);
    }
    return p;
  }

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<RealType, MemorySpace>(state, "psi", domain, box,
                                                        0);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    const auto op = spectral::make_operator_params(params);
    const auto phys = spectral::physics_for_mode(k_laplacian, op);
    return spectral::linear_symbol(k_laplacian, phys.opCk);
  }

  /// PFC dynamics are conserved: \f$\hat N\f$ enters as \f$k_{\mathrm{lap}}\hat N\f$.
  [[nodiscard]] double nonlinear_symbol(double k_laplacian) const {
    return k_laplacian;
  }

  [[nodiscard]] double filter_mf(double k_laplacian) const {
    const auto op = spectral::make_operator_params(params);
    return spectral::physics_for_mode(k_laplacian, op).filterMF;
  }

  [[nodiscard]] TungstenPointwise pointwise() const noexcept {
    return {.c_psi = -params.get_stabP(),
            .c_psi2 = params.get_p3_bar(),
            .c_psi3 = params.get_p4_bar(),
            .c_mf2 = params.get_q3_bar(),
            .c_mf3 = params.get_q4_bar()};
  }

  /// Host convenience: \f$N(\psi,\psi_{\mathrm{MF}})\f$ via the pointwise functor.
  [[nodiscard]] double nonlinearity(double psi, double psi_mf) const {
    return pointwise().nonlinearity(
        pfc::sim::SpectralCell{.psi = psi, .psi_mf = psi_mf});
  }
};

static_assert(pfc::sim::SpectralETDPhysics<TungstenPhysics<>>);
static_assert(pfc::sim::HasMeanFieldFilter<TungstenPhysics<>>);
static_assert(pfc::sim::HasNonlinearSymbol<TungstenPhysics<>>);
static_assert(!pfc::sim::HasCorrelationKernel<TungstenPhysics<>>);
static_assert(pfc::sim::HasParameters<TungstenPhysics<>>);

} // namespace tungsten
