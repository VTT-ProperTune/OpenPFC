// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cahn_hilliard_physics.hpp
 * @brief Regular-solution Cahn–Hilliard (Fe–Cr 475 °C representative) for ETD.
 *
 * @details
 * One conserved composition field \f$c\f$ (Cr mole fraction) with
 *
 * \f[
 *   \partial_t c = M\nabla^2\bigl(f'(c)-\kappa\nabla^2 c\bigr).
 * \f]
 *
 * The bulk density is the regular-solution model
 * \f$f(c)=\omega c(1-c)+c\ln c+(1-c)\ln(1-c)\f$ in units of \f$RT\f$,
 * \f$\omega=\Omega/(RT)\f$. Default \f$T=748.15\,\mathrm{K}\f$ (475 °C) and
 * \f$\Omega=20.1\,\mathrm{kJ\,mol^{-1}}\f$ put Fe–32Cr inside the chemical
 * spinodal; they are a documented representative miscibility-gap model, not a
 * CALPHAD assessment.
 *
 * In Fourier space, with OpenPFC's \f$k_{\mathrm{lap}}=-|k|^2\f$,
 *
 * \f[
 *   L(k)=M\bigl(f''(c_0)\,k_{\mathrm{lap}}-\kappa\,k_{\mathrm{lap}}^2\bigr),
 *   \qquad
 *   M_{\mathrm{nl}}(k)=M\,k_{\mathrm{lap}}.
 * \f]
 *
 * Consumed by `pfc::sim::SpectralETDSystem` on every backend. No k-loops and
 * no hand-written kernels in this header.
 */

#include <cmath>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

#include <cahn_hilliard/cahn_hilliard_pointwise.hpp>

namespace cahn_hilliard {

/// JSON-shaped values. Energy unit is \f$RT\f$; \f$\Omega\f$ is J/mol.
struct CahnHilliardSchemaValues {
  double c0{0.32};       ///< mean Cr mole fraction (linearization point)
  double T{748.15};      ///< temperature (K), 475 °C
  double Omega{2.01e4};  ///< regular-solution interaction (J/mol)
  double R{8.314462618}; ///< gas constant (J/(mol K))
  double kappa{1.0};     ///< gradient-energy coefficient (grid units)
  double M{1.0};         ///< mobility (grid units)
};

struct CahnHilliardParams : CahnHilliardSchemaValues {
  double omega_nd{0.0}; ///< \f$\Omega/(RT)\f$
  double fprime0{0.0};
  double fpp0{0.0};

  CahnHilliardParams() { recompute_derived(); }

  void recompute_derived() {
    const double RT = R * T;
    omega_nd = (RT > 0.0) ? Omega / RT : 0.0;
    CahnHilliardPointwise pw{.omega_nd = omega_nd, .c0 = c0};
    fprime0 = pw.f_prime(c0);
    fpp0 = pw.f_double_prime(c0);
  }
};

inline void apply_schema_values(const CahnHilliardSchemaValues &v,
                                CahnHilliardParams &p) {
  static_cast<CahnHilliardSchemaValues &>(p) = v;
  p.recompute_derived();
}

inline pfc::sim::ParameterSchema<CahnHilliardSchemaValues>
make_cahn_hilliard_schema() {
  pfc::sim::ParameterSchema<CahnHilliardSchemaValues> s;
  s.model_name("CahnHilliard")
      .real(&CahnHilliardSchemaValues::c0,
            {.name = "c0",
             .description = "mean Cr mole fraction (linearization point)",
             .required = false,
             .min = 1.0e-8,
             .max = 1.0 - 1.0e-8,
             .default_value = 0.32})
      .real(&CahnHilliardSchemaValues::T,
            {.name = "T",
             .description = "temperature (K); default 475 C",
             .required = false,
             .min = 1.0,
             .default_value = 748.15,
             .units = "K"})
      .real(&CahnHilliardSchemaValues::Omega,
            {.name = "Omega",
             .description = "regular-solution interaction parameter",
             .required = false,
             .min = 0.0,
             .default_value = 2.01e4,
             .units = "J/mol"})
      .real(&CahnHilliardSchemaValues::R, {.name = "R",
                                           .description = "gas constant",
                                           .required = false,
                                           .min = 1.0,
                                           .default_value = 8.314462618,
                                           .units = "J/(mol K)"})
      .real(&CahnHilliardSchemaValues::kappa,
            {.name = "kappa",
             .description = "gradient-energy coefficient in grid units",
             .required = false,
             .min = 0.0,
             .default_value = 1.0})
      .real(&CahnHilliardSchemaValues::M, {.name = "M",
                                           .description = "mobility in grid units",
                                           .required = false,
                                           .min = 0.0,
                                           .default_value = 1.0});
  return s;
}

inline void apply_cahn_hilliard_json(const nlohmann::json &j,
                                     CahnHilliardParams &p) {
  apply_schema_values(make_cahn_hilliard_schema().parse(j), p);
}

/**
 * @tparam RealType    Field element type (default double).
 * @tparam MemorySpace Host or device space for `declare_fields`.
 */
template <class RealType = double, class MemorySpace = pfc::HostSpace>
struct CahnHilliardPhysics {
  using parameters_type = CahnHilliardParams;
  using pointwise_type = CahnHilliardPointwise;

  pfc::Domain domain{};
  pfc::Box3i box{};
  CahnHilliardParams params{};

  static pfc::sim::ParameterSchema<CahnHilliardSchemaValues> schema() {
    return make_cahn_hilliard_schema();
  }

  static CahnHilliardPhysics from_json(const nlohmann::json &params_json,
                                       const pfc::Domain &domain_in,
                                       const pfc::Box3i &box_in) {
    CahnHilliardPhysics p;
    p.domain = domain_in;
    p.box = box_in;
    if (!params_json.is_null() && !params_json.empty()) {
      apply_cahn_hilliard_json(params_json, p.params);
    } else {
      p.params.recompute_derived();
    }
    return p;
  }

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<RealType, MemorySpace>(state, "c", domain, box, 0);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    return params.M *
           (params.fpp0 * k_laplacian - params.kappa * k_laplacian * k_laplacian);
  }

  /// Conserved dynamics: \f$\hat N\f$ enters as \f$M k_{\mathrm{lap}}\hat N\f$.
  [[nodiscard]] double nonlinear_symbol(double k_laplacian) const {
    return params.M * k_laplacian;
  }

  [[nodiscard]] CahnHilliardPointwise pointwise() const {
    return {.omega_nd = params.omega_nd,
            .c0 = params.c0,
            .fprime0 = params.fprime0,
            .fpp0 = params.fpp0};
  }

  [[nodiscard]] bool in_spinodal(double c) const {
    return pointwise().f_double_prime(c) < 0.0;
  }
};

static_assert(pfc::sim::SpectralETDPhysics<CahnHilliardPhysics<>>);
static_assert(pfc::sim::HasNonlinearSymbol<CahnHilliardPhysics<>>);
static_assert(pfc::sim::HasParameters<CahnHilliardPhysics<>>);

} // namespace cahn_hilliard
