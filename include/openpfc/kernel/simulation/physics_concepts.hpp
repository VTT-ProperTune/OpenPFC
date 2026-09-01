// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file physics_concepts.hpp
 * @brief C++20 concepts for method-independent physics on `SimulationState`.
 *
 * @details
 * A 0.2 physics model is data plus concept-conforming callables, not a
 * `Model` base class (Audit §13.7). This header names the three surfaces
 * M7 requires:
 *
 * 1. **Field declaration** — `declare_fields(SimulationState&)` registers
 *    named fields. `add_declared_field` constructs a canonical
 *    `pfc::data::Field` from domain + owned box + halo so physics does
 *    not hand-roll allocation.
 * 2. **Point-wise RHS** — `rhs(t, G)` as used by `for_each_interior` and
 *    Gen-3 models (`HeatModel::rhs`). `G` is the model's grads aggregate.
 * 3. **Spectral-diagonal descriptors** — real linear symbol `L(k)` from
 *    the OpenPFC Laplacian `k_laplacian` (`-|k|^2`) plus a real-space
 *    nonlinearity `N(psi)`. Matches the `physics_for_mode` / `linear_symbol`
 *    split already factored in tungsten.
 * 4. **Steppable physics** — `step(t)` for Gen-1 `Model` and adapter A1.
 *
 * Parameter schema (`ParameterSchema`) is a sibling M7 header; physics
 * that expose a nested `parameters_type` model `HasParameters`.
 *
 * A model may satisfy the point-wise surface, the spectral-diagonal
 * surface, or both. `DeclaresFields` is required for the combined
 * `PointwisePhysics` / `SpectralETDPhysics` concepts used by the
 * forthcoming `SpectralETDSystem` driver.
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

namespace pfc::sim {

/** Device-side N(psi, psi_MF) polynomial: c1 psi + c2 psi^2 + c3 psi^3 + same in MF.
 */
struct MeanFieldNonlinearityPoly {
  double c_psi{};
  double c_psi2{};
  double c_psi3{};
  double c_mf{};
  double c_mf2{};
  double c_mf3{};
};

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
 * @brief Point-wise right-hand side `rhs(t, G)` (Gen-3 / `for_each_interior`).
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
 * @p k_laplacian is OpenPFC's spectral Laplacian (`-|k|^2`), the same
 * argument tungsten's `physics_for_mode` / `linear_symbol` take.
 */
template <class Physics>
concept SpectralLinearSymbol = requires(const Physics &physics, double k_laplacian) {
  { physics.linear_symbol(k_laplacian) } -> std::convertible_to<double>;
};

/**
 * @brief Real-space nonlinearity @f$N(\psi)@f$ for spectral ETD.
 *
 * The forthcoming `SpectralETDSystem` evaluates this per cell, then
 * transforms; physics does not own the FFT.
 */
template <class Physics>
concept SpectralNonlinearity = requires(const Physics &physics, double psi) {
  { physics.nonlinearity(psi) } -> std::convertible_to<double>;
};

/**
 * @brief Spectral-diagonal descriptors: @f$L(k)@f$ plus @f$N(\psi)@f$.
 */
template <class Physics>
concept SpectralDiagonalPhysics =
    SpectralLinearSymbol<Physics> && SpectralNonlinearity<Physics>;

/**
 * @brief Nested `parameters_type` for the forthcoming `ParameterSchema`.
 */
template <class Physics>
concept HasParameters = requires { typename Physics::parameters_type; };

/**
 * @brief Physics that advances by `step(t)` (Gen-1 `Model` and A1).
 *
 * Distinct from `PointwiseRhs` / `SpectralDiagonalPhysics`: the callable
 * owns the whole update. `pfc::compat::LegacyModelPhysics` models this
 * by delegating to `Model::step`.
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
 */
template <class Physics>
concept SpectralETDPhysics =
    DeclaresFields<Physics> && SpectralDiagonalPhysics<Physics>;

/**
 * @brief Mean-field spectral ETD: @f$L(k)@f$, filter @f$\chi(k)@f$, and
 *        @f$N(\psi,\psi_{\mathrm{MF}})@f$ (tungsten / aluminum).
 *
 * The driver FFTs @f$\psi@f$, applies @f$\chi@f$ to form @f$\psi_{\mathrm{MF}}@f$,
 * evaluates the two-argument nonlinearity, then ETD-combines with
 * @f$n_{\mathrm{weight}} = k_{\mathrm{lap}}\,\phi_1(L\,\mathrm{d}t)@f$.
 */
template <class Physics>
concept MeanFieldETDPhysics =
    DeclaresFields<Physics> && SpectralLinearSymbol<Physics> &&
    requires(const Physics &physics, double k_laplacian, double psi, double psi_mf) {
      { physics.filter_mf(k_laplacian) } -> std::convertible_to<double>;
      { physics.nonlinearity(psi, psi_mf) } -> std::convertible_to<double>;
    };

/**
 * @brief Mean-field ETD with a correlation kernel @f$P(k)@f$ and a
 *        temperature-dependent @f$N@f$ (aluminum moving frame).
 *
 * The driver FFTs @f$\psi@f$, applies @f$\chi@f$ and @f$P(k)@f$, then
 * evaluates @f$N(\psi,\psi_{\mathrm{MF}},P*\psi,T_{\mathrm{var}})@f$.
 * Free-energy density is the M7 observable hook.
 */
template <class Physics>
concept MovingFrameMeanFieldETDPhysics =
    DeclaresFields<Physics> && SpectralLinearSymbol<Physics> &&
    requires(const Physics &physics, double k_laplacian, double psi, double psi_mf,
             double p_star, double T_var, double x, double t) {
      { physics.filter_mf(k_laplacian) } -> std::convertible_to<double>;
      { physics.correlation_kernel(k_laplacian) } -> std::convertible_to<double>;
      { physics.temperature_variation(x, t) } -> std::convertible_to<double>;
      {
        physics.nonlinearity(psi, psi_mf, p_star, T_var)
      } -> std::convertible_to<double>;
      {
        physics.free_energy_density(psi, psi_mf, p_star, T_var)
      } -> std::convertible_to<double>;
    };

} // namespace pfc::sim
