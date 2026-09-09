// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file anisotropy.hpp
 * @brief Orientation-dependent surface-diffusion stiffness \f$B(\theta)\f$
 *        for the science case of `#115`.
 *
 * @details
 * The Mullins verifier (`surface_diffusion_physics.hpp`) keeps a single,
 * orientation-independent coefficient \f$B\f$. Real crystalline surfaces do
 * not: the surface free energy \f$\gamma\f$ depends on the local surface
 * normal, and by the Herring relation the *kinetic* stiffness that actually
 * controls diffusive smoothing is
 * \f[
 *   \tilde\gamma(\theta) = \gamma(\theta) + \gamma''(\theta),
 * \f]
 * so a facet (a cusp or a strongly cusped minimum of \f$\gamma(\theta)\f$)
 * relaxes at a different rate than a vicinal orientation between facets
 * (Mullins 1957; Rettori & Villain, *J. Phys. France* 49, 257 (1988);
 * Bonzel & Preuss, *Surf. Sci.* 336, 209 (1995)).
 *
 * **Honesty note (model maturity).** This header does *not* differentiate a
 * specific literature \f$\gamma(\theta)\f$ to obtain \f$\tilde\gamma(\theta)\f$.
 * It imposes the periodic modulation directly on the *kinetic* coefficient,
 *
 * \f[
 *   B(\theta) = B_0\bigl[1 + \epsilon_a\cos(m\theta)\bigr],
 * \f]
 *
 * with \f$\theta=\operatorname{atan2}(h_y,h_x)\f$ the local surface-slope
 * orientation and \f$m\in\{4,6\}\f$ the crystal symmetry order. That is the
 * same *functional form* the literature above derives for \f$\tilde\gamma\f$
 * near a weakly cusped minimum, and it recovers the qualitative faceting
 * phenomenology (orientations near the stiff lobes of \f$B(\theta)\f$ persist
 * longer than orientations near the soft lobes), but the coefficients
 * \f$(B_0,\epsilon_a,m)\f$ here are illustrative, not fitted to any specific
 * material's measured \f$\gamma(\theta)\f$. Do not read \f$\epsilon_a\f$ as a
 * calibrated anisotropy strength for a real crystal facet.
 *
 * This stays a **small-slope, height-function** model throughout: \f$\theta\f$
 * is the orientation of the local surface *gradient*, not of an arbitrary
 * crystal facet normal on a multivalued or overhanging surface. It does not
 * claim arbitrary-slope crystalline surface evolution (Issue `#115`,
 * "Required model").
 */

#include <cmath>

#include <nlohmann/json.hpp>

#include <openpfc/kernel/simulation/parameter_schema.hpp>

namespace surface_diffusion {

/// Raw JSON-facing fields for the anisotropic model.
struct AnisotropySchemaValues {
  double B0{1.0};    ///< isotropic (orientation-averaged) Mullins coefficient
  double eps_a{0.0}; ///< anisotropy strength; 0 recovers the isotropic model
  int m{4};          ///< symmetry order (4 = cubic/fourfold, 6 = hexagonal)
};

struct AnisotropyParams : AnisotropySchemaValues {};

inline void apply_schema_values(const AnisotropySchemaValues &v,
                                AnisotropyParams &p) {
  static_cast<AnisotropySchemaValues &>(p) = v;
}

inline pfc::sim::ParameterSchema<AnisotropySchemaValues>
make_anisotropy_schema() {
  pfc::sim::ParameterSchema<AnisotropySchemaValues> s;
  s.model_name("SurfaceDiffusionAnisotropic")
      .real(&AnisotropySchemaValues::B0,
            {.name = "B0",
             .description = "orientation-averaged Mullins coefficient",
             .required = false,
             .min = 0.0,
             .default_value = 1.0})
      .real(&AnisotropySchemaValues::eps_a,
            {.name = "eps_a",
             .description =
                 "anisotropy strength; 0 reduces exactly to the isotropic "
                 "model (unfitted, illustrative -- see anisotropy.hpp)",
             .required = false,
             .min = 0.0,
             .max = 1.0,
             .default_value = 0.0})
      .integer(&AnisotropySchemaValues::m,
               {.name = "m",
                .description = "surface-stiffness symmetry order (4 or 6)",
                .required = false,
                .min = 1.0,
                .max = 8.0,
                .default_value = 4.0});
  return s;
}

inline void apply_anisotropy_json(const nlohmann::json &j, AnisotropyParams &p) {
  apply_schema_values(make_anisotropy_schema().parse(j), p);
}

/**
 * @brief Orientation-dependent surface-diffusion stiffness
 *        \f$B(\theta)=B_0[1+\epsilon_a\cos(m\theta)]\f$.
 *
 * By construction, `B0` is a fixed point of the family in @p m: setting
 * `eps_a = 0` makes `operator()` return `B0` for every `theta`, which is the
 * exact isotropic Mullins coefficient used by `SurfaceDiffusionPhysics`. The
 * function is `2*pi/m`-periodic by construction (`cos(m*(theta + 2*pi/m)) =
 * cos(m*theta + 2*pi) = cos(m*theta)`), which is the fourfold/sixfold
 * symmetry the science preset claims.
 */
struct SurfaceStiffness {
  double B0{1.0};
  double eps_a{0.0};
  int m{4};

  [[nodiscard]] double operator()(double theta) const {
    return B0 * (1.0 + eps_a * std::cos(static_cast<double>(m) * theta));
  }

  static SurfaceStiffness from_params(const AnisotropyParams &p) {
    return SurfaceStiffness{p.B0, p.eps_a, p.m};
  }
};

} // namespace surface_diffusion
