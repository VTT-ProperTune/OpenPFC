// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file gradient_elasticity_physics.hpp
 * @brief Isotropic Helmholtz–Navier gradient elasticity (`#82`).
 *
 * @details
 * Classical linear elasticity has no material length and can produce
 * singular high-\(k\) fields near idealized defects. Aifantis-type
 * gradient elasticity applies a Helmholtz operator to the Navier
 * operator,
 *
 * \f[
 *   (1-\ell^2\nabla^2)\,L_{\mathrm{navier}}\,\mathbf{u}=\mathbf{f},
 * \f]
 *
 * which is fourth order in displacement. The internal length \(\ell\)
 * is the scale over which strain gradients are penalized: features
 * much smaller than \(\ell\) are regularized.
 *
 * For constant isotropic moduli on a periodic grid the Fourier problem
 * is a \(2\times 2\) system at each \(\mathbf{k}\). Split
 * \(\hat{\mathbf{f}}=\hat{\mathbf{f}}_L+\hat{\mathbf{f}}_T\) into
 * parts parallel and perpendicular to \(\mathbf{k}\):
 *
 * \f[
 *   \hat{\mathbf{u}}_L=\frac{\hat{\mathbf{f}}_L}{-\alpha(\lambda+2\mu)k^2},
 *   \qquad
 *   \hat{\mathbf{u}}_T=\frac{\hat{\mathbf{f}}_T}{-\alpha\mu k^2},
 * \f]
 *
 * with \(\alpha=1+\ell^2 k^2\) (fourth order). The \(k=0\) mode is
 * projected to \(\hat{\mathbf{u}}=0\) (rigid translation). Stretch
 * sixth-order forms are \(\alpha=(1+\ell^2 k^2)^2\) or
 * \(\alpha=1+\ell^2 k^2+\ell_4^4 k^4\).
 *
 * A 2-D dilatational eigenstrain \(\varepsilon^*=\varepsilon_0 g\,I\)
 * has equivalent body force \(\mathbf{f}=\nabla\cdot(C:\varepsilon^*)
 * =2(\lambda+\mu)\varepsilon_0\nabla g\). For a cosine inclusion
 * \(g=A\cos(\mathbf{k}\cdot\mathbf{x})\) the displacement is the
 * sine field
 * \(\mathbf{u}=B\mathbf{k}\sin(\mathbf{k}\cdot\mathbf{x})\) with
 * \(B=2(\lambda+\mu)\varepsilon_0 A/(\alpha(\lambda+2\mu)k^2)\).
 */

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

namespace gradient_elasticity {

struct GradientElasticitySchemaValues {
  double mu{1.0};     ///< shear modulus
  double lambda{1.0}; ///< first Lamé parameter
  double ell{1.0};    ///< Helmholtz internal length
  double ell4{0.0};   ///< optional second-gradient length (\(\ell_4^4 k^4\))
  double eps0{1.0};   ///< dilatational eigenstrain amplitude
  int order{4};       ///< 4: \(\alpha=1+\ell^2 k^2\); 6: \(\alpha=(1+\ell^2 k^2)^2\)
};

struct GradientElasticityParams : GradientElasticitySchemaValues {
  double singular_threshold{1.0e-14};
};

struct ModeForce {
  std::complex<double> fx{};
  std::complex<double> fy{};
};

struct ModeDisplacement {
  std::complex<double> ux{};
  std::complex<double> uy{};
};

/// Spectral strain-tensor components at one wavevector (tensor, not
/// engineering, shear: \(\hat\varepsilon_{xy}=\tfrac12(ik_y\hat u_x+ik_x\hat
/// u_y)\)).
struct ModeStrain {
  std::complex<double> exx{};
  std::complex<double> eyy{};
  std::complex<double> exy{};
};

/// Real-space local elastic state derived from the classical constitutive
/// law evaluated on the (regularized) displacement field -- see
/// `GradientElasticityPhysics::stress_state`.
struct LocalStressState {
  double sxx{};
  double syy{};
  double sxy{};
  double hydrostatic{};   ///< \((\sigma_{xx}+\sigma_{yy})/2\), 2-D mean normal stress.
  double von_mises{};     ///< in-plane reduced von Mises, see `stress_state`.
  double energy_density{}; ///< \(\tfrac12\sigma_{ij}\varepsilon^e_{ij}\).
};

inline pfc::sim::ParameterSchema<GradientElasticitySchemaValues>
make_gradient_elasticity_schema() {
  pfc::sim::ParameterSchema<GradientElasticitySchemaValues> s;
  s.model_name("GradientElasticity")
      .real(&GradientElasticitySchemaValues::mu, {.name = "mu",
                                                  .description = "shear modulus μ",
                                                  .required = false,
                                                  .min = 1.0e-15,
                                                  .default_value = 1.0})
      .real(&GradientElasticitySchemaValues::lambda,
            {.name = "lambda",
             .description = "first Lamé parameter λ",
             .required = false,
             .default_value = 1.0})
      .real(&GradientElasticitySchemaValues::ell,
            {.name = "ell",
             .description = "Helmholtz internal length ℓ",
             .required = false,
             .min = 0.0,
             .default_value = 1.0})
      .real(&GradientElasticitySchemaValues::ell4,
            {.name = "ell4",
             .description = "optional second-gradient length; 0 ignores this form",
             .required = false,
             .min = 0.0,
             .default_value = 0.0})
      .real(&GradientElasticitySchemaValues::eps0,
            {.name = "eps0",
             .description = "dilatational eigenstrain amplitude ε0",
             .required = false,
             .default_value = 1.0})
      .integer(&GradientElasticitySchemaValues::order,
               {.name = "order",
                .description = "spatial order: 4 (Helmholtz) or 6 ((1+ℓ²k²)²)",
                .required = false,
                .min = 4.0,
                .max = 6.0,
                .default_value = 4.0});
  return s;
}

inline void apply_young_poisson(const nlohmann::json &j,
                                GradientElasticityParams &p) {
  const bool has_E = j.contains("E");
  const bool has_nu = j.contains("nu");
  if (!has_E && !has_nu) {
    return;
  }
  if (!has_E || !has_nu || !j["E"].is_number() || !j["nu"].is_number()) {
    throw std::invalid_argument("gradient_elasticity: 'E' and 'nu' must both be "
                                "numeric when either is set.");
  }
  const double E = j["E"].get<double>();
  const double nu = j["nu"].get<double>();
  if (!(E > 0.0)) {
    throw std::invalid_argument(
        "gradient_elasticity: Young's modulus E must be > 0.");
  }
  if (!(nu > -1.0 && nu < 0.5)) {
    throw std::invalid_argument(
        "gradient_elasticity: Poisson ratio nu must satisfy -1 < nu < 1/2.");
  }
  const double denom = (1.0 + nu) * (1.0 - 2.0 * nu);
  if (!(std::abs(denom) > 0.0)) {
    throw std::invalid_argument("gradient_elasticity: (1+nu)(1-2 nu) is zero.");
  }
  p.mu = E / (2.0 * (1.0 + nu));
  p.lambda = E * nu / denom;
}

inline void validate_params(const GradientElasticityParams &p) {
  if (p.order != 4 && p.order != 6) {
    throw std::invalid_argument("gradient_elasticity: order must be 4 or 6.");
  }
  if (!(p.mu > 0.0)) {
    throw std::invalid_argument("gradient_elasticity: mu must be > 0.");
  }
  if (!(p.lambda + 2.0 * p.mu > 0.0)) {
    throw std::invalid_argument(
        "gradient_elasticity: lambda + 2 mu must be > 0 (P-wave modulus).");
  }
}

inline void apply_gradient_elasticity_json(const nlohmann::json &j,
                                           GradientElasticityParams &p) {
  static_cast<GradientElasticitySchemaValues &>(p) =
      make_gradient_elasticity_schema().parse(j);
  apply_young_poisson(j, p);
  validate_params(p);
}

template <class RealType = double, class MemorySpace = pfc::HostSpace>
struct GradientElasticityPhysics {
  using parameters_type = GradientElasticityParams;

  pfc::Domain domain{};
  pfc::Box3i box{};
  GradientElasticityParams params{};

  static pfc::sim::ParameterSchema<GradientElasticitySchemaValues> schema() {
    return make_gradient_elasticity_schema();
  }

  static GradientElasticityPhysics from_json(const nlohmann::json &params_json,
                                             const pfc::Domain &domain_in,
                                             const pfc::Box3i &box_in) {
    GradientElasticityPhysics p;
    p.domain = domain_in;
    p.box = box_in;
    if (!params_json.is_null() && !params_json.empty()) {
      apply_gradient_elasticity_json(params_json, p.params);
    }
    return p;
  }

  /// Displacement fields plus the derived strain/stress/energy diagnostics
  /// (`#117`): `exx`/`eyy`/`exy` (compatible strain of `u`), `sxx`/`syy`/`sxy`
  /// (Cauchy stress), `stress_hydro`/`stress_vm` (invariants), and
  /// `energy_density`. See `stress_state()` for the constitutive convention.
  void declare_fields(pfc::SimulationState &state) const {
    for (const char *name : {"g", "ux", "uy", "exx", "eyy", "exy", "sxx", "syy",
                             "sxy", "stress_hydro", "stress_vm", "energy_density"}) {
      pfc::sim::add_declared_field<RealType, MemorySpace>(state, name, domain, box,
                                                           0);
    }
  }

  /// Helmholtz factor \(\alpha(k^2)\).
  [[nodiscard]] double helmholtz_alpha(double k2) const {
    const double ell2 = params.ell * params.ell;
    const double a1 = 1.0 + ell2 * k2;
    if (params.ell4 > 0.0) {
      const double e4 = params.ell4;
      const double e44 = (e4 * e4) * (e4 * e4);
      return 1.0 + ell2 * k2 + e44 * k2 * k2;
    }
    if (params.order == 6) {
      return a1 * a1;
    }
    return a1;
  }

  /// Equivalent body force \(\hat f=2(\lambda+\mu)\varepsilon_0 i\mathbf{k}\,\hat
  /// g\).
  [[nodiscard]] ModeForce eigenstrain_force(double kx, double ky, double /*kz*/,
                                            std::complex<double> ghat) const {
    const double pref = 2.0 * (params.lambda + params.mu) * params.eps0;
    const std::complex<double> ik{0.0, 1.0};
    return {pref * ik * kx * ghat, pref * ik * ky * ghat};
  }

  /// Invert the 2-D Helmholtz–Navier symbol at one wavevector.
  [[nodiscard]] ModeDisplacement invert(double kx, double ky, double kz,
                                        ModeForce f) const {
    const double k2 = kx * kx + ky * ky + kz * kz;
    if (!(k2 > params.singular_threshold)) {
      return {};
    }
    const double alpha = helmholtz_alpha(k2);
    const double invk2 = 1.0 / k2;
    const auto kdotf = kx * f.fx + ky * f.fy;
    const ModeForce fL{kdotf * invk2 * kx, kdotf * invk2 * ky};
    const ModeForce fT{f.fx - fL.fx, f.fy - fL.fy};
    const double denL = -alpha * (params.lambda + 2.0 * params.mu) * k2;
    const double denT = -alpha * params.mu * k2;
    return {fL.fx / denL + fT.fx / denT, fL.fy / denL + fT.fy / denT};
  }

  /// Prefactor \(B\) in \(\mathbf{u}=B\mathbf{k}\sin(\mathbf{k}\cdot\mathbf{x})\)
  /// for a cosine inclusion of amplitude 1.
  [[nodiscard]] double cosine_displacement_prefactor(double k2) const {
    const double alpha = helmholtz_alpha(k2);
    return 2.0 * (params.lambda + params.mu) * params.eps0 /
           (alpha * (params.lambda + 2.0 * params.mu) * k2);
  }

  /// Spectral strain \(\hat\varepsilon=\tfrac12(i\mathbf{k}\otimes\hat{\mathbf{u}}
  /// +\hat{\mathbf{u}}\otimes i\mathbf{k})\) from the mode displacement.
  [[nodiscard]] ModeStrain strain_from_displacement(double kx, double ky,
                                                     ModeDisplacement u) const {
    const std::complex<double> ik{0.0, 1.0};
    return {ik * kx * u.ux, ik * ky * u.uy,
            0.5 * (ik * ky * u.ux + ik * kx * u.uy)};
  }

  /**
   * @brief Local Cauchy stress, hydrostatic/von-Mises invariants, and
   * elastic energy density at one grid point, from the compatible strain
   * \(\varepsilon(\mathbf{u})\) and the local eigenstrain field value \(g\).
   *
   * @details
   * The displacement \(\mathbf{u}\) already solves the regularized
   * Helmholtz--Navier equilibrium equation, so the strain
   * \(\varepsilon(\mathbf{u})\) it produces is itself smoothed relative to
   * the classical (\(\ell=0\)) solution near sharp features. This function
   * then applies the **classical, local** isotropic constitutive law to
   * that (already regularized) strain:
   * \(\sigma=\lambda\,\mathrm{tr}(\varepsilon^e)I+2\mu\varepsilon^e\), with
   * elastic strain \(\varepsilon^e=\varepsilon(\mathbf{u})-\varepsilon^*\),
   * \(\varepsilon^*=\varepsilon_0 g I\).
   *
   * This is a deliberate, simpler modeling choice: it is **not** the
   * higher-order Aifantis stress operator
   * \(\sigma_{\mathrm{grad}}=(1-\ell^2\nabla^2)\sigma_{\mathrm{classical}}\),
   * which would need its own (order-dependent) regularizing operator and is
   * not uniquely defined for the `order=6`/`ell4` variants used here.
   * Regularization enters this report only through the smoothed
   * displacement/strain field. See the app README for the caveat.
   *
   * Von Mises is the **in-plane reduced** invariant
   * \(\sigma_{vm}=\sqrt{\sigma_{xx}^2-\sigma_{xx}\sigma_{yy}+\sigma_{yy}^2
   * +3\sigma_{xy}^2}\); it ignores an out-of-plane \(\sigma_{zz}\) because
   * the eigenstrain here is purely planar (no `zz` component), so this is
   * not the full 3-D plane-strain von Mises stress.
   */
  [[nodiscard]] LocalStressState stress_state(double exx, double eyy, double exy,
                                              double g) const {
    const double eps_star = params.eps0 * g;
    const double eexx = exx - eps_star;
    const double eeyy = eyy - eps_star;
    const double eexy = exy; // eigenstrain here has no shear component
    const double trace = eexx + eeyy;
    LocalStressState s{};
    s.sxx = params.lambda * trace + 2.0 * params.mu * eexx;
    s.syy = params.lambda * trace + 2.0 * params.mu * eeyy;
    s.sxy = 2.0 * params.mu * eexy;
    s.hydrostatic = 0.5 * (s.sxx + s.syy);
    s.von_mises = std::sqrt(s.sxx * s.sxx - s.sxx * s.syy + s.syy * s.syy +
                            3.0 * s.sxy * s.sxy);
    s.energy_density = 0.5 * (s.sxx * eexx + s.syy * eeyy + 2.0 * s.sxy * eexy);
    return s;
  }
};

} // namespace gradient_elasticity
