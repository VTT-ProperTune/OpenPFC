// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file ehd_film_physics.hpp
 * @brief Elastohydrodynamic film under a flexible plate for spectral ETD (`#81`).
 *
 * @details
 * A viscous gap \(h\) supports a Kirchhoff plate. Bending sets the pressure,
 * lubrication turns the pressure gradient into a flux, and mass conservation
 * produces a sixth-order evolution:
 *
 * \f[
 *   p=B\nabla^4 h-\gamma\nabla^2 h-\Pi(h),\qquad
 *   \partial_t h=\nabla\cdot\bigl[M_0\nabla p\bigr].
 * \f]
 *
 * Chain: plate curvature \(\to\) bending pressure \(\to\) pressure gradient
 * \(\to\) viscous flux \(\to\) \(\partial_t h\). The highest derivative is
 * \(\nabla^6\). With OpenPFC \(k_{\mathrm{lap}}=-|k|^2\),
 *
 * \f[
 *   L(k)=M_0 B\,k_{\mathrm{lap}}^3-M_0\gamma\,k_{\mathrm{lap}}^2
 *        -M_0\Pi'(h_0)\,k_{\mathrm{lap}}.
 * \f]
 *
 * Linear growth is \(\lambda(k)=-M_0 B k^6-M_0\gamma k^4+M_0\Pi'(h_0)k^2\).
 * Defaults \(A=\gamma=0\) are pure bending relaxation, \(\lambda=-M_0 B k^6\).
 * An explicit step on a grid of spacing \(\Delta x\) would need
 * \(\Delta t\lesssim \Delta x^6/(M_0 B\pi^6)\); ETD treats \(k^6\) as a multiply.
 */

#include <cmath>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

#include <ehd_film/ehd_film_pointwise.hpp>

namespace ehd_film {

struct EhdFilmSchemaValues {
  double h0{1.0};    ///< mean gap
  double B{1.0};     ///< plate bending stiffness
  double M0{1.0};    ///< mobility at h0
  double gamma{0.0}; ///< tension (0 = bending only)
  double A{0.0};     ///< disjoining strength (0 = no van der Waals)
};

struct EhdFilmParams : EhdFilmSchemaValues {
  double Pi0{0.0};
  double Pip0{0.0};

  EhdFilmParams() { recompute_derived(); }

  void recompute_derived() {
    EhdFilmPointwise pw{.A = A, .h0 = h0};
    Pi0 = pw.Pi(h0);
    Pip0 = pw.Pi_prime(h0);
  }
};

inline void apply_schema_values(const EhdFilmSchemaValues &v, EhdFilmParams &p) {
  static_cast<EhdFilmSchemaValues &>(p) = v;
  p.recompute_derived();
}

inline pfc::sim::ParameterSchema<EhdFilmSchemaValues> make_ehd_film_schema() {
  pfc::sim::ParameterSchema<EhdFilmSchemaValues> s;
  s.model_name("EhdFilm")
      .real(&EhdFilmSchemaValues::h0,
            {.name = "h0",
             .description = "mean gap (linearization thickness)",
             .required = false,
             .min = 1.0e-3,
             .default_value = 1.0})
      .real(&EhdFilmSchemaValues::B, {.name = "B",
                                      .description = "plate bending stiffness",
                                      .required = false,
                                      .min = 0.0,
                                      .default_value = 1.0})
      .real(&EhdFilmSchemaValues::M0, {.name = "M0",
                                       .description = "mobility at h0",
                                       .required = false,
                                       .min = 0.0,
                                       .default_value = 1.0})
      .real(&EhdFilmSchemaValues::gamma,
            {.name = "gamma",
             .description = "membrane tension; 0 is bending only",
             .required = false,
             .min = 0.0,
             .default_value = 0.0})
      .real(&EhdFilmSchemaValues::A,
            {.name = "A",
             .description = "disjoining strength; 0 is no van der Waals",
             .required = false,
             .min = 0.0,
             .default_value = 0.0});
  return s;
}

inline void apply_ehd_film_json(const nlohmann::json &j, EhdFilmParams &p) {
  apply_schema_values(make_ehd_film_schema().parse(j), p);
}

template <class RealType = double, class MemorySpace = pfc::HostSpace>
struct EhdFilmPhysics {
  using parameters_type = EhdFilmParams;
  using pointwise_type = EhdFilmPointwise;

  pfc::Domain domain{};
  pfc::Box3i box{};
  EhdFilmParams params{};

  static pfc::sim::ParameterSchema<EhdFilmSchemaValues> schema() {
    return make_ehd_film_schema();
  }

  static EhdFilmPhysics from_json(const nlohmann::json &params_json,
                                  const pfc::Domain &domain_in,
                                  const pfc::Box3i &box_in) {
    EhdFilmPhysics p;
    p.domain = domain_in;
    p.box = box_in;
    if (!params_json.is_null() && !params_json.empty()) {
      apply_ehd_film_json(params_json, p.params);
    } else {
      p.params.recompute_derived();
    }
    return p;
  }

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<RealType, MemorySpace>(state, "h", domain, box, 0);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    const double k2 = k_laplacian * k_laplacian;
    return params.M0 * params.B * k_laplacian * k2 - params.M0 * params.gamma * k2 -
           params.M0 * params.Pip0 * k_laplacian;
  }

  [[nodiscard]] double nonlinear_symbol(double k_laplacian) const {
    return -params.M0 * k_laplacian;
  }

  [[nodiscard]] EhdFilmPointwise pointwise() const {
    return {.A = params.A, .h0 = params.h0, .Pi0 = params.Pi0, .Pip0 = params.Pip0};
  }

  /// Fastest-growing \(k^2\), or 0 when every mode decays.
  [[nodiscard]] double k_peak_sq() const {
    if (params.Pip0 <= 0.0 || params.B <= 0.0) {
      return 0.0;
    }
    const double disc =
        4.0 * params.gamma * params.gamma + 12.0 * params.B * params.Pip0;
    const double q = (-2.0 * params.gamma + std::sqrt(disc)) / (6.0 * params.B);
    return (q > 0.0) ? q : 0.0;
  }
};

static_assert(pfc::sim::SpectralETDPhysics<EhdFilmPhysics<>>);
static_assert(pfc::sim::HasNonlinearSymbol<EhdFilmPhysics<>>);
static_assert(pfc::sim::HasParameters<EhdFilmPhysics<>>);

} // namespace ehd_film
