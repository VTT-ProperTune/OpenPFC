// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file kawahara_physics.hpp
 * @brief Kawahara capillary–gravity waves for spectral ETD (`#80`).
 *
 * @details
 * Long weakly nonlinear waves with competing third- and fifth-order
 * dispersion (Kawahara 1972):
 *
 * \f[
 *   \partial_t u + \alpha u\partial_x u - \beta\partial_x^3 u
 *   + \gamma\partial_x^5 u = 0.
 * \f]
 *
 * This app defines beta with a minus sign in the PDE. To use a coefficient
 * b multiplying +u_xxx, set beta=-b. Existing inputs retain their meaning.
 * Defaults \(\alpha=1\), \(\beta=1\), \(\gamma=-1\) give competing dispersion.
 * The linear symbol is purely imaginary,
 * \(L(k)=-i\omega(k)\) with \(\omega(k)=\beta k^3+\gamma k^5\). Phase
 * velocity \(c_p=\beta k^2+\gamma k^4\) changes sign at
 * \(k^2=-\beta/\gamma\) when \(\beta\gamma<0\). That is the opposite of
 * even-order dissipative operators (\(k^4\), \(k^6\)), which damp rather
 * than rotate Fourier modes.
 *
 * The quadratic flux is evaluated as \(N=u^2\) with
 * \(M(k)=-i(\alpha/2)k_x\) (dealiased).
 */

#include <complex>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

#include <kawahara/kawahara_pointwise.hpp>

namespace kawahara {

struct KawaharaSchemaValues {
  double alpha{1.0};  ///< nonlinear coefficient
  double beta{1.0};   ///< third-order dispersion
  double gamma{-1.0}; ///< fifth-order dispersion
};

struct KawaharaParams : KawaharaSchemaValues {
  KawaharaParams() = default;
};

inline void apply_schema_values(const KawaharaSchemaValues &v, KawaharaParams &p) {
  static_cast<KawaharaSchemaValues &>(p) = v;
}

inline pfc::sim::ParameterSchema<KawaharaSchemaValues> make_kawahara_schema() {
  pfc::sim::ParameterSchema<KawaharaSchemaValues> s;
  s.model_name("Kawahara")
      .real(&KawaharaSchemaValues::alpha,
            {.name = "alpha",
             .description = "nonlinear coefficient of u u_x",
             .required = false,
             .default_value = 1.0})
      .real(&KawaharaSchemaValues::beta,
            {.name = "beta",
             .description = "third-order dispersion coefficient",
             .required = false,
             .default_value = 1.0})
      .real(&KawaharaSchemaValues::gamma,
            {.name = "gamma",
             .description = "fifth-order dispersion coefficient",
             .required = false,
             .default_value = -1.0});
  return s;
}

inline void apply_kawahara_json(const nlohmann::json &j, KawaharaParams &p) {
  apply_schema_values(make_kawahara_schema().parse(j), p);
}

[[nodiscard]] inline double omega_k(double k, const KawaharaParams &p) {
  const double k2 = k * k;
  const double k3 = k2 * k;
  return p.beta * k3 + p.gamma * k3 * k2;
}

template <class RealType = double, class MemorySpace = pfc::HostSpace>
struct KawaharaPhysics {
  using parameters_type = KawaharaParams;
  using pointwise_type = KawaharaPointwise;

  pfc::Domain domain{};
  pfc::Box3i box{};
  KawaharaParams params{};

  static pfc::sim::ParameterSchema<KawaharaSchemaValues> schema() {
    return make_kawahara_schema();
  }

  static KawaharaPhysics from_json(const nlohmann::json &params_json,
                                   const pfc::Domain &domain_in,
                                   const pfc::Box3i &box_in) {
    KawaharaPhysics p;
    p.domain = domain_in;
    p.box = box_in;
    if (!params_json.is_null() && !params_json.empty()) {
      apply_kawahara_json(params_json, p.params);
    }
    return p;
  }

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<RealType, MemorySpace>(state, "u", domain, box, 0);
  }

  [[nodiscard]] std::complex<double> linear_symbol(double kx, double, double) const {
    return {0.0, -omega_k(kx, params)};
  }

  [[nodiscard]] std::complex<double> nonlinear_symbol(double kx, double,
                                                      double) const {
    return {0.0, -0.5 * params.alpha * kx};
  }

  [[nodiscard]] KawaharaPointwise pointwise() const { return {}; }
};

static_assert(pfc::sim::HasComplexLinearSymbol<KawaharaPhysics<>>);
static_assert(pfc::sim::HasComplexNonlinearSymbol<KawaharaPhysics<>>);
static_assert(pfc::sim::SpectralETDPhysics<KawaharaPhysics<>>);
static_assert(pfc::sim::HasParameters<KawaharaPhysics<>>);

} // namespace kawahara
