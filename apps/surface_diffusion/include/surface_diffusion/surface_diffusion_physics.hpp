// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file surface_diffusion_physics.hpp
 * @brief Mullins small-slope surface diffusion for spectral ETD (`#79`).
 *
 * @details
 * Curvature drives a chemical-potential gradient, atoms diffuse along the
 * surface, and mass conservation on a small-slope profile yields
 *
 * \f[
 *   \partial_t h = -B\nabla^4 h.
 * \f]
 *
 * Every Fourier mode decays independently:
 * \f$h_k(t)=h_k(0)\exp(-B|k|^4 t)\f$. With OpenPFC
 * \f$k_{\mathrm{lap}}=-|k|^2\f$,
 *
 * \f[
 *   L(k)=-B\,k_{\mathrm{lap}}^2.
 * \f]
 *
 * There is no real-space nonlinearity. Short-wavelength roughness therefore
 * disappears much faster than long-wavelength roughness — the industrial
 * annealing / nanoscale-smoothing story.
 */

#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

#include <surface_diffusion/surface_diffusion_pointwise.hpp>

namespace surface_diffusion {

struct SurfaceDiffusionSchemaValues {
  double B{1.0}; ///< Mullins coefficient (grid units)
};

struct SurfaceDiffusionParams : SurfaceDiffusionSchemaValues {
  SurfaceDiffusionParams() = default;
};

inline void apply_schema_values(const SurfaceDiffusionSchemaValues &v,
                                SurfaceDiffusionParams &p) {
  static_cast<SurfaceDiffusionSchemaValues &>(p) = v;
}

inline pfc::sim::ParameterSchema<SurfaceDiffusionSchemaValues>
make_surface_diffusion_schema() {
  pfc::sim::ParameterSchema<SurfaceDiffusionSchemaValues> s;
  s.model_name("SurfaceDiffusion")
      .real(&SurfaceDiffusionSchemaValues::B,
            {.name = "B",
             .description = "Mullins surface-diffusion coefficient",
             .required = false,
             .min = 0.0,
             .default_value = 1.0});
  return s;
}

inline void apply_surface_diffusion_json(const nlohmann::json &j,
                                         SurfaceDiffusionParams &p) {
  apply_schema_values(make_surface_diffusion_schema().parse(j), p);
}

template <class RealType = double, class MemorySpace = pfc::HostSpace>
struct SurfaceDiffusionPhysics {
  using parameters_type = SurfaceDiffusionParams;
  using pointwise_type = SurfaceDiffusionPointwise;

  pfc::Domain domain{};
  pfc::Box3i box{};
  SurfaceDiffusionParams params{};

  static pfc::sim::ParameterSchema<SurfaceDiffusionSchemaValues> schema() {
    return make_surface_diffusion_schema();
  }

  static SurfaceDiffusionPhysics from_json(const nlohmann::json &params_json,
                                           const pfc::Domain &domain_in,
                                           const pfc::Box3i &box_in) {
    SurfaceDiffusionPhysics p;
    p.domain = domain_in;
    p.box = box_in;
    if (!params_json.is_null() && !params_json.empty()) {
      apply_surface_diffusion_json(params_json, p.params);
    }
    return p;
  }

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<RealType, MemorySpace>(state, "h", domain, box, 0);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    return -params.B * k_laplacian * k_laplacian;
  }

  [[nodiscard]] SurfaceDiffusionPointwise pointwise() const { return {}; }
};

static_assert(pfc::sim::SpectralETDPhysics<SurfaceDiffusionPhysics<>>);
static_assert(pfc::sim::HasParameters<SurfaceDiffusionPhysics<>>);

} // namespace surface_diffusion
