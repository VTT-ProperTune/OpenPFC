// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file higher_order_pfc_physics.hpp
 * @brief Higher-order (two-mode) PFC correlation kernel for spectral ETD.
 *
 * @details
 * One conserved density field \f$\psi\f$ with
 *
 * \f[
 *   F[\psi] = \int \Bigl[\tfrac12 \psi\,\Lambda(\nabla^2)\,\psi
 *                        - \tfrac{g}{3}\psi^3 + \tfrac14\psi^4\Bigr]\,d\mathbf r,
 *   \qquad
 *   \partial_t\psi = M\nabla^2\frac{\delta F}{\delta\psi}.
 * \f]
 *
 * The kernel is the two-mode PFC operator
 *
 * \f[
 *   \Lambda(\nabla^2) = -\varepsilon
 *     + (1+\nabla^2)^2\bigl[r_1 + (q_1^2+\nabla^2)^2\bigr],
 * \f]
 *
 * which contains \f$(\nabla^2)^4\f$ — **eighth order in space**. Conserved
 * dynamics multiplies by another \f$\nabla^2\f$, so the evolution operator runs
 * through \f$(\nabla^2)^5\f$, i.e. **tenth order**.
 *
 * In Fourier space, with OpenPFC's \f$u=k_{\mathrm{lap}}=-|k|^2\f$, both are
 * just polynomials in \f$u\f$ (see `correlation_kernel.hpp`):
 *
 * \f[
 *   L(u) = M\,u\,\Lambda(u), \qquad M_{\mathrm{nl}}(u) = M\,u .
 * \f]
 *
 * `n_modes = 1` selects the classical fourth-order kernel
 * \f$-\varepsilon+(1+\nabla^2)^2\f$ for side-by-side comparison.
 *
 * Consumed by `pfc::sim::SpectralETDSystem` on every backend. No k-loops and
 * no hand-written kernels in this header.
 */

#include <cmath>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

#include <higher_order_pfc/correlation_kernel.hpp>
#include <higher_order_pfc/higher_order_pfc_pointwise.hpp>

namespace higher_order_pfc {

/// JSON-shaped values. Everything is in reduced PFC units.
struct HigherOrderPFCSchemaValues {
  double eps{0.25};   ///< quench depth \f$\varepsilon\f$
  double q1{1.4142135623730951}; ///< second peak \f$q_1\f$; \f$\sqrt2\f$ = 2D square
  double r1{0.0};     ///< second-peak offset; 0 makes both minima degenerate
  double M{1.0};      ///< mobility
  double g{0.0};      ///< cubic coefficient in the local free energy
  int n_modes{2};     ///< 1 = classical \f$k^4\f$ kernel, 2 = \f$k^8\f$ kernel
};

struct HigherOrderPFCParams : HigherOrderPFCSchemaValues {
  QuadraticKernel kernel{};  ///< \f$\Lambda(u)\f$, quartic in \f$u\f$
  EvolutionSymbol symbol{};  ///< \f$L(u)=Mu\Lambda(u)\f$, quintic in \f$u\f$

  HigherOrderPFCParams() { recompute_derived(); }

  void recompute_derived() {
    if (n_modes != 1 && n_modes != 2)
      throw std::invalid_argument("higher_order_pfc: n_modes must be 1 or 2");
    kernel = (n_modes == 1) ? single_mode_kernel(eps) : two_mode_kernel(eps, q1, r1);
    symbol = conserved_symbol(kernel, M);
  }

  /// Depth of the second minimum relative to the first, \f$(1-q_1^2)^2 r_1\f$.
  [[nodiscard]] double second_mode_offset() const {
    if (n_modes != 2) return 0.0;
    const double d = 1.0 - q1 * q1;
    return d * d * r1;
  }
};

inline void apply_schema_values(const HigherOrderPFCSchemaValues &v,
                                HigherOrderPFCParams &p) {
  static_cast<HigherOrderPFCSchemaValues &>(p) = v;
  p.recompute_derived();
}

inline pfc::sim::ParameterSchema<HigherOrderPFCSchemaValues>
make_higher_order_pfc_schema() {
  pfc::sim::ParameterSchema<HigherOrderPFCSchemaValues> s;
  s.model_name("HigherOrderPFC")
      .real(&HigherOrderPFCSchemaValues::eps,
            {.name = "eps",
             .description = "quench depth; the k=1 minimum sits at -eps",
             .required = false,
             .min = 0.0,
             .default_value = 0.25})
      .real(&HigherOrderPFCSchemaValues::q1,
            {.name = "q1",
             .description = "second correlation peak; sqrt(2) squares in 2D, "
                            "2/sqrt(3) FCC in 3D",
             .required = false,
             .min = 1.0e-6,
             .default_value = 1.4142135623730951})
      .real(&HigherOrderPFCSchemaValues::r1,
            {.name = "r1",
             .description = "second-peak offset; 0 makes the two minima degenerate",
             .required = false,
             .min = 0.0,
             .default_value = 0.0})
      .real(&HigherOrderPFCSchemaValues::M, {.name = "M",
                                             .description = "mobility",
                                             .required = false,
                                             .min = 0.0,
                                             .default_value = 1.0})
      .real(&HigherOrderPFCSchemaValues::g,
            {.name = "g",
             .description = "cubic coefficient; breaks psi -> -psi symmetry",
             .required = false,
             .default_value = 0.0})
      .integer(&HigherOrderPFCSchemaValues::n_modes,
               {.name = "n_modes",
                .description = "1 = classical k^4 kernel, 2 = k^8 kernel",
                .required = false,
                .min = 1.0,
                .max = 2.0,
                .default_value = 2.0});
  return s;
}

inline void apply_higher_order_pfc_json(const nlohmann::json &j,
                                        HigherOrderPFCParams &p) {
  apply_schema_values(make_higher_order_pfc_schema().parse(j), p);
}

/**
 * @tparam RealType    Field element type (default double).
 * @tparam MemorySpace Host or device space for `declare_fields`.
 */
template <class RealType = double, class MemorySpace = pfc::HostSpace>
struct HigherOrderPFCPhysics {
  using parameters_type = HigherOrderPFCParams;
  using pointwise_type = HigherOrderPFCPointwise;

  pfc::Domain domain{};
  pfc::Box3i box{};
  HigherOrderPFCParams params{};

  static pfc::sim::ParameterSchema<HigherOrderPFCSchemaValues> schema() {
    return make_higher_order_pfc_schema();
  }

  static HigherOrderPFCPhysics from_json(const nlohmann::json &params_json,
                                         const pfc::Domain &domain_in,
                                         const pfc::Box3i &box_in) {
    HigherOrderPFCPhysics p;
    p.domain = domain_in;
    p.box = box_in;
    if (!params_json.is_null() && !params_json.empty()) {
      apply_higher_order_pfc_json(params_json, p.params);
    } else {
      p.params.recompute_derived();
    }
    return p;
  }

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<RealType, MemorySpace>(state, "psi", domain, box, 0);
  }

  /// \f$L(u)=M\,u\,\Lambda(u)\f$ — tenth order in \f$k\f$, zero at \f$k=0\f$.
  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    return params.symbol(k_laplacian);
  }

  /// Conserved dynamics: \f$\hat N\f$ enters as \f$M k_{\mathrm{lap}}\hat N\f$.
  [[nodiscard]] double nonlinear_symbol(double k_laplacian) const {
    return params.M * k_laplacian;
  }

  /// Free-energy kernel \f$\Lambda(u)\f$ — eighth order in \f$k\f$.
  [[nodiscard]] double quadratic_kernel(double k_laplacian) const {
    return params.kernel(k_laplacian);
  }

  [[nodiscard]] HigherOrderPFCPointwise pointwise() const {
    return {.g = params.g};
  }
};

static_assert(pfc::sim::SpectralETDPhysics<HigherOrderPFCPhysics<>>);
static_assert(pfc::sim::HasNonlinearSymbol<HigherOrderPFCPhysics<>>);
static_assert(pfc::sim::HasParameters<HigherOrderPFCPhysics<>>);

} // namespace higher_order_pfc
