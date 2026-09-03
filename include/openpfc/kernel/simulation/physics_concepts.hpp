// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file physics_concepts.hpp
 * @brief C++20 concepts for method-independent physics on `SimulationState`.
 *
 * @details
 * A 0.2 physics model is data plus concept-conforming callables, not a base
 * class (Audit §13.7). This header names the surfaces the framework drivers
 * consume:
 *
 * 1. **Field declaration** — `declare_fields(SimulationState&)` registers
 *    named fields. `add_declared_field` constructs a canonical
 *    `pfc::data::Field` from domain + owned box + halo so physics does
 *    not hand-roll allocation.
 * 2. **Point-wise RHS** — `rhs(t, G)` as used by `for_each_interior`
 *    (explicit FD / spectral gradient path). `G` is the model's grads
 *    aggregate.
 * 3. **Spectral-ETD descriptors** — the single contract consumed by
 *    `SpectralETDSystem` for every memory space:
 *      - `linear_symbol(k_laplacian)` — real diagonal symbol \f$L(k)\f$;
 *      - `pointwise()` — a device-capable functor (`SpectralPointwise`)
 *        evaluating the real-space nonlinearity per cell;
 *      - optional `nonlinear_symbol(k_laplacian)` — real multiplier
 *        \f$M(k)\f$ applied to \f$\hat N\f$ (defaults to 1; PFC models use
 *        \f$k_{\mathrm{lap}}\f$ so that
 *        \f$\partial_t\hat\psi = L\hat\psi + M\hat N\f$);
 *      - optional `filter_mf(k_laplacian)` — mean-field filter
 *        \f$\chi(k)\f$; the driver then supplies `cell.psi_mf`;
 *      - optional `correlation_kernel(k_laplacian)` — \f$P(k)\f$; the driver
 *        then supplies `cell.p_star`.
 *    The driver detects the optional capabilities at compile time, so one
 *    system class serves plain Swift–Hohenberg-like models, mean-field PFC
 *    (tungsten), and moving-frame mean-field PFC (aluminum).
 * 4. **Steppable physics** — `step(t)` for objects that own their whole
 *    update (e.g. a `SpectralETDSystem` driven by `pfc::sim::run`).
 *
 * Parameter schema (`ParameterSchema`) is a sibling header; physics that
 * expose a nested `parameters_type` model `HasParameters`.
 */

#include <complex>
#include <concepts>
#include <stdexcept>
#include <string>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace pfc::sim {

/**
 * @brief Element kind for a declared field (real density vs complex hat).
 */
enum class FieldElementKind { Real, Complex };

/**
 * @brief Name + element + halo for one field the physics needs.
 *
 * Memory space is a template on `add_declared_field`, matching
 * `Field<T, MemorySpace>`. Physics templated on a memory space passes
 * that space at the allocation call; it is not a runtime field on this
 * struct.
 */
struct FieldDeclaration {
  std::string name;
  FieldElementKind element{FieldElementKind::Real};
  int halo{0};
};

/**
 * @brief Allocate a named field of element type @p T into @p state.
 */
template <class T, class MemorySpace = pfc::HostSpace>
void add_declared_field(SimulationState &state, const std::string &name,
                        const Domain &domain, const Box3i &box, int halo = 0) {
  state.add_field<T, MemorySpace>(
      name, pfc::data::Field<T, MemorySpace>(domain, box, halo));
}

/**
 * @brief Allocate from a @ref FieldDeclaration (real or complex).
 */
template <class MemorySpace = pfc::HostSpace>
void add_declared_field(SimulationState &state, const FieldDeclaration &decl,
                        const Domain &domain, const Box3i &box) {
  switch (decl.element) {
  case FieldElementKind::Real:
    add_declared_field<double, MemorySpace>(state, decl.name, domain, box,
                                            decl.halo);
    break;
  case FieldElementKind::Complex:
    add_declared_field<std::complex<double>, MemorySpace>(state, decl.name, domain,
                                                          box, decl.halo);
    break;
  default:
    throw std::invalid_argument("add_declared_field: unknown FieldElementKind");
  }
}

/**
 * @brief Physics that registers its fields on an owning `SimulationState`.
 *
 * Invocable as `physics.declare_fields(state)`. The physics object is
 * not required to own geometry; tests and drivers typically store
 * `Domain` + `Box3i` on the physics and call @ref add_declared_field.
 */
template <class Physics>
concept DeclaresFields = requires(const Physics &physics, SimulationState &state) {
  physics.declare_fields(state);
};

/**
 * @brief Point-wise right-hand side `rhs(t, G)` (`for_each_interior` path).
 *
 * Return type is unconstrained: single-field models return a scalar;
 * multi-field models return an increments aggregate.
 */
template <class Physics, class Grads>
concept PointwiseRhs = requires(const Physics &physics, double t,
                                const Grads &grads) { physics.rhs(t, grads); };

/**
 * @brief Diagonal linear symbol @f$L(k)@f$ from the Laplacian multiplier.
 *
 * @p k_laplacian is OpenPFC's spectral Laplacian (`-|k|^2`).
 */
template <class Physics>
concept SpectralLinearSymbol = requires(const Physics &physics, double k_laplacian) {
  { physics.linear_symbol(k_laplacian) } -> std::convertible_to<double>;
};

/**
 * @brief Optional multiplier @f$M(k)@f$ on @f$\hat N@f$ (default 1).
 */
template <class Physics>
concept HasNonlinearSymbol = requires(const Physics &physics, double k_laplacian) {
  { physics.nonlinear_symbol(k_laplacian) } -> std::convertible_to<double>;
};

/**
 * @brief Optional mean-field filter @f$\chi(k)@f$; enables `cell.psi_mf`.
 */
template <class Physics>
concept HasMeanFieldFilter = requires(const Physics &physics, double k_laplacian) {
  { physics.filter_mf(k_laplacian) } -> std::convertible_to<double>;
};

/**
 * @brief Optional correlation kernel @f$P(k)@f$; enables `cell.p_star`.
 */
template <class Physics>
concept HasCorrelationKernel =
    requires(const Physics &physics, double k_laplacian) {
      { physics.correlation_kernel(k_laplacian) } -> std::convertible_to<double>;
    };

/**
 * @brief Physics that provides a device-capable pointwise nonlinearity.
 */
template <class Physics>
concept HasSpectralPointwise = requires(const Physics &physics) {
  { physics.pointwise() } -> SpectralPointwise;
};

/**
 * @brief Nested `parameters_type` for `ParameterSchema`.
 */
template <class Physics>
concept HasParameters = requires { typename Physics::parameters_type; };

/**
 * @brief Physics that advances by `step(t)`.
 *
 * Distinct from `PointwiseRhs` / `SpectralETDPhysics`: the callable owns
 * the whole update (ETD systems, `pfc::sim::run` steppers).
 */
template <class Physics>
concept SteppablePhysics = requires(Physics &physics, double t) { physics.step(t); };

/**
 * @brief Field-declaring point-wise physics (explicit FD / spectral path).
 */
template <class Physics, class Grads>
concept PointwisePhysics = DeclaresFields<Physics> && PointwiseRhs<Physics, Grads>;

/**
 * @brief Field-declaring spectral-ETD physics (stiff PFC path).
 *
 * Required: `declare_fields`, `linear_symbol(k)`, `pointwise()`.
 * Optional, detected by the driver: `nonlinear_symbol(k)`, `filter_mf(k)`,
 * `correlation_kernel(k)`, and `free_energy_density(cell)` on the functor.
 */
template <class Physics>
concept SpectralETDPhysics = DeclaresFields<Physics> &&
                             SpectralLinearSymbol<Physics> &&
                             HasSpectralPointwise<Physics>;

/// Functor type returned by `physics.pointwise()`.
template <class Physics>
  requires HasSpectralPointwise<Physics>
using spectral_pointwise_t =
    std::remove_cvref_t<decltype(std::declval<const Physics &>().pointwise())>;

} // namespace pfc::sim
