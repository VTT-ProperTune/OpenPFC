// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file thin_film_physics.hpp
 * @brief Lubrication thin-film / coating model for spectral ETD (`#78`).
 *
 * @details
 * One conserved thickness field \f$h\f$ with constant mobility \f$M_0\f$:
 *
 * \f[
 *   \partial_t h=\nabla\cdot\bigl[M_0\nabla p\bigr],\qquad
 *   p=-\gamma\nabla^2 h-\Pi(h).
 * \f]
 *
 * The leading minus in some textbook flux forms would make capillary
 * \f$k^4\f$ *destabilizing*; the form above is the standard lubrication
 * balance (flow down the pressure gradient) so high-\(k\) capillary damps.
 * Cubic mobility \f$M\propto h^3\f$ is a local-flux nonlinearity and is
 * left for a later slice; \f$M_0=M(h_0)\f$ is exact for the linear band.
 *
 * Disjoining pressure \f$\Pi(h)=A\bigl((h_0/h)^3-(h_0/h)^9\bigr)\f$.
 * With OpenPFC \f$k_{\mathrm{lap}}=-|k|^2\f$,
 *
 * \f[
 *   L(k)=-M_0\gamma\,k_{\mathrm{lap}}^2-M_0\Pi'(h_0)\,k_{\mathrm{lap}},
 *   \qquad
 *   M_{\mathrm{nl}}(k)=-M_0 k_{\mathrm{lap}}.
 * \f]
 *
 * Linear growth is \f$\lambda(k)=M_0 k^2\bigl(\Pi'(h_0)-\gamma k^2\bigr)\f$.
 * \f$A=0\f$ is a leveling coating (every mode decays).
 */

#include <cmath>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

#include <thin_film/thin_film_pointwise.hpp>

namespace thin_film {

struct ThinFilmSchemaValues {
  double h0{1.0};    ///< mean film thickness
  double gamma{1.0}; ///< surface tension (grid units)
  double M0{1.0};    ///< mobility at h0
  double A{0.05};    ///< disjoining strength (0 = leveling)
  double h_star{0.0};///< precursor thickness; >0 selects the rupture-safe form
};

struct ThinFilmParams : ThinFilmSchemaValues {
  double Pi0{0.0};
  double Pip0{0.0};

  ThinFilmParams() { recompute_derived(); }

  void recompute_derived() {
    ThinFilmPointwise pw{.A = A, .h0 = h0, .h_star = h_star};
    Pi0 = pw.Pi(h0);
    Pip0 = pw.Pi_prime(h0);
  }
};

inline void apply_schema_values(const ThinFilmSchemaValues &v, ThinFilmParams &p) {
  static_cast<ThinFilmSchemaValues &>(p) = v;
  p.recompute_derived();
}

inline pfc::sim::ParameterSchema<ThinFilmSchemaValues> make_thin_film_schema() {
  pfc::sim::ParameterSchema<ThinFilmSchemaValues> s;
  s.model_name("ThinFilm")
      .real(&ThinFilmSchemaValues::h0,
            {.name = "h0",
             .description = "mean film thickness (linearization point)",
             .required = false,
             .min = 1.0e-3,
             .default_value = 1.0})
      .real(&ThinFilmSchemaValues::gamma,
            {.name = "gamma",
             .description = "surface tension in grid units",
             .required = false,
             .min = 0.0,
             .default_value = 1.0})
      .real(&ThinFilmSchemaValues::M0, {.name = "M0",
                                        .description = "mobility at h0",
                                        .required = false,
                                        .min = 0.0,
                                        .default_value = 1.0})
      .real(&ThinFilmSchemaValues::A,
            {.name = "A",
             .description = "disjoining strength; 0 is leveling",
             .required = false,
             .min = 0.0,
             .default_value = 0.05})
      .real(&ThinFilmSchemaValues::h_star,
            {.name = "h_star",
             .description = "precursor thickness; >0 uses the rupture-safe "
                            "disjoining pressure with a stable thin film",
             .required = false,
             .min = 0.0,
             .default_value = 0.0});
  return s;
}

inline void apply_thin_film_json(const nlohmann::json &j, ThinFilmParams &p) {
  apply_schema_values(make_thin_film_schema().parse(j), p);
}

template <class RealType = double, class MemorySpace = pfc::HostSpace>
struct ThinFilmPhysics {
  using parameters_type = ThinFilmParams;
  using pointwise_type = ThinFilmPointwise;

  pfc::Domain domain{};
  pfc::Box3i box{};
  ThinFilmParams params{};

  static pfc::sim::ParameterSchema<ThinFilmSchemaValues> schema() {
    return make_thin_film_schema();
  }

  static ThinFilmPhysics from_json(const nlohmann::json &params_json,
                                   const pfc::Domain &domain_in,
                                   const pfc::Box3i &box_in) {
    ThinFilmPhysics p;
    p.domain = domain_in;
    p.box = box_in;
    if (!params_json.is_null() && !params_json.empty()) {
      apply_thin_film_json(params_json, p.params);
    } else {
      p.params.recompute_derived();
    }
    return p;
  }

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<RealType, MemorySpace>(state, "h", domain, box, 0);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    return -params.M0 * params.gamma * k_laplacian * k_laplacian -
           params.M0 * params.Pip0 * k_laplacian;
  }

  [[nodiscard]] double nonlinear_symbol(double k_laplacian) const {
    return -params.M0 * k_laplacian;
  }

  [[nodiscard]] ThinFilmPointwise pointwise() const {
    return {.A = params.A, .h0 = params.h0, .Pi0 = params.Pi0, .Pip0 = params.Pip0};
  }

  /// Fastest-growing wavenumber squared, or 0 when the film is leveling.
  [[nodiscard]] double k_peak_sq() const {
    if (params.gamma <= 0.0 || params.Pip0 <= 0.0) {
      return 0.0;
    }
    return params.Pip0 / (2.0 * params.gamma);
  }
};

static_assert(pfc::sim::SpectralETDPhysics<ThinFilmPhysics<>>);
static_assert(pfc::sim::HasNonlinearSymbol<ThinFilmPhysics<>>);
static_assert(pfc::sim::HasParameters<ThinFilmPhysics<>>);

} // namespace thin_film
