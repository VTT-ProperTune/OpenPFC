// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file field_modifier.hpp
 * @brief Base class for initial conditions and boundary conditions
 *
 * @details
 * This file defines the FieldModifier abstract base class, which provides a
 * unified interface for modifying field values in simulations. FieldModifiers
 * are used for:
 * - Initial conditions (applied once before time integration)
 * - Boundary conditions (applied every time step or at intervals)
 * - Field perturbations and custom modifications
 *
 * Concrete implementations include:
 * - Initial conditions: Constant, Seed, SeedGrid, RandomSeeds, FileReader
 * - Directional-solidification BCs are app-local (`apps/common`) and
 *   catalog-registered by tungsten/aluminum, not by the kernel.
 *
 * Typical usage:
 * @code
 * class MyInitialCondition : public pfc::FieldModifier {
 * public:
 *     void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
 *                const pfc::Box3i &box, double time) override {
 *         (void)domain; (void)box; (void)time;
 *         // Modify field values
 *     }
 * };
 *
 * ic.apply(field, domain, box, 0.0);
 * @endcode
 *
 * This file is part of the Field Operations module, providing mechanisms
 * for setting initial states and enforcing boundary constraints.
 *
 * @see simulation_driver.hpp for `pfc::sim::run` orchestration
 * @see initial_conditions/ for IC implementations
 */

#ifndef PFC_FIELD_MODIFIER_HPP
#define PFC_FIELD_MODIFIER_HPP

#include <mpi.h>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/simulation_context.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace pfc {

/**
 * @brief Abstract base class for field modifiers in OpenPFC
 *
 * FieldModifier provides a unified interface for modifying field values in
 * simulations, supporting both initial conditions (applied once) and boundary
 * conditions (applied repeatedly during time integration).
 *
 * **Core Concept:**
 * Field modifiers implement
 * `apply(field::FieldOutput<double>, const Domain&, const Box3i&, double)` over
 * the local owned box (x-fastest, halo 0). Sessions apply ICs once at start and
 * BCs from the `pfc::sim::run` apply hook, through `apply_field_modifier`
 * (kernel/simulation/apply_field_modifier.hpp), which handles host and device
 * `Field`s alike.
 *
 * **Design Philosophy:**
 * - **Extensibility**: Users can create custom modifiers without touching OpenPFC
 * - **Composition**: Multiple modifiers can be applied in sequence
 * - **Context**: `SimulationContext` supplies the MPI communicator and rank-0 flag
 * - **Single Responsibility**: Each modifier does one thing well
 *
 * @example Creating a custom initial condition
 * @code
 * class GaussianIC : public pfc::FieldModifier {
 *   Real3 m_center;
 *   double m_amplitude;
 *   double m_width;
 * public:
 *   GaussianIC(Real3 center, double amp, double width)
 *     : m_center(center), m_amplitude(amp), m_width(width) {}
 *
 *   void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
 *              const pfc::Box3i &box, double) override {
 *     pfc::field::apply(field, domain, box, [&](const pfc::Real3 &pos) {
 *       double dx = pos[0] - m_center[0];
 *       double dy = pos[1] - m_center[1];
 *       double dz = pos[2] - m_center[2];
 *       return m_amplitude * std::exp(-(dx*dx + dy*dy + dz*dz) / (m_width*m_width));
 *     });
 *   }
 * };
 * @endcode
 *
 * @example Creating a time-dependent boundary condition
 * @code
 * class DirichletBC : public pfc::FieldModifier {
 *   double m_value;
 *   double m_width;  // Transition width
 * public:
 *   DirichletBC(double value, double width = 5.0)
 *     : m_value(value), m_width(width) {}
 *
 *   void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
 *              const pfc::Box3i &box, double time) override {
 *     const double Lx = pfc::domain::get_size(domain, 0) *
 *                       pfc::domain::get_spacing(domain, 0);
 *     pfc::field::apply_inplace_with_time(
 *         field, domain, box, time, [&](const pfc::Real3 &X, double cur, double t) {
 *           if (X[0] > Lx - m_width) {
 *             double s = (X[0] - (Lx - m_width)) / m_width;
 *             return cur * (1.0 - s) + m_value * std::sin(t) * s;
 *           }
 *           return cur;
 *         });
 *   }
 * };
 * @endcode
 *
 * @example Applying to a canonical Field (host or device)
 * @code
 * pfc::Field<double> psi(domain, box, 0);
 * GaussianIC ic({10, 10, 10}, 0.1, 2.0);
 * pfc::apply_field_modifier(ic, psi, 0.0);     // host: wraps psi.output()
 * pfc::Field<double, pfc::HIPSpace> d_psi(domain, box, 0);
 * pfc::apply_field_modifier(ic, d_psi, 0.0);   // device: brackets with_host_view
 * @endcode
 *
 * **Usage with `pfc::sim::run`:**
 * - Initial conditions: applied once before the time loop
 * - Boundary conditions: applied from the `apply` hook each step
 * - Application order: ICs first, then BCs, in the order listed
 *
 * **Performance Considerations:**
 * - Boundary conditions are in the hot path (applied every step)
 * - Minimize allocations in `apply()`
 * - On device fields every application is a host round-trip of the owned box
 *
 * @note The `time` parameter allows implementing time-dependent boundary
 *       conditions, but most initial conditions ignore it (t=0 at IC application)
 *
 * @see apply_field_modifier.hpp
 * @see simulation_driver.hpp for `pfc::sim::run`
 * @see initial_conditions/ for built-in IC implementations
 */
class FieldModifier {

private:
  std::vector<std::string> m_field_names{"default"};

  static void validate_field_names(const std::vector<std::string> &names) {
    if (names.empty()) {
      throw std::invalid_argument("FieldModifier target field list cannot be empty");
    }
    for (const auto &n : names) {
      if (n.empty()) {
        throw std::invalid_argument("Field name in target list cannot be empty");
      }
    }
  }

public:
  /**
   * @brief Declare every field this modifier may write (for Simulator checks).
   */
  void set_field_names(std::vector<std::string> names) {
    validate_field_names(names);
    m_field_names = std::move(names);
  }

  /**
   * @brief Set the field name this modifier should operate on
   *
   * Names the `SimulationState` field the session hands to `apply`. This allows
   * the same modifier implementation to be reused for different fields.
   *
   * @param field_name Name of the field to modify (e.g., "density", "temperature")
   *
   * @throws std::invalid_argument if field_name is empty
   *
   * @example Basic usage
   * @code
   * auto ic = std::make_unique<pfc::Constant>(0.5);
   * ic->set_field_name("density");
   * session.add_initial_condition(std::move(ic));
   * @endcode
   *
   * @example Multiple fields with same modifier type
   * @code
   * // Apply constant IC to density field
   * auto density_ic = std::make_unique<pfc::Constant>(0.5);
   * density_ic->set_field_name("density");
   * session.add_initial_condition(std::move(density_ic));
   *
   * // Apply constant IC to temperature field
   * auto temp_ic = std::make_unique<pfc::Constant>(300.0);
   * temp_ic->set_field_name("temperature");
   * session.add_initial_condition(std::move(temp_ic));
   * @endcode
   *
   * @note The field name must match a field declared on the `SimulationState`;
   *       JSON wiring fails closed on unknown targets
   * @note Default field name is "default" if not explicitly set
   *
   * @see get_field_name()
   */
  void set_field_name(const std::string &field_name) {
    if (field_name.empty()) {
      throw std::invalid_argument("Field name cannot be empty");
    }
    m_field_names = {field_name};
  }

  const std::vector<std::string> &get_field_names() const { return m_field_names; }

  /**
   * @brief Get the name of the field this modifier operates on
   *
   * Returns the (first) field name set via `set_field_name()`; sessions use it
   * to look up the target field on `SimulationState` before calling `apply()`.
   *
   * @note Returns "default" if field name was never explicitly set
   * @see set_field_name()
   */
  const std::string &get_field_name() const { return m_field_names.front(); }

  virtual const std::string &get_modifier_name() const {
    static const std::string k{"FieldModifier"};
    return k;
  }

  /**
   * @brief Optional MPI communicator for modifiers that use collectives (MPI-IO,
   *        reductions). Default is a no-op; FileReader and app-local
   *        front-tracking BCs override.
   */
  virtual void set_mpi_comm(MPI_Comm /*comm*/) {}

  /**
   * @brief Apply the field modification with explicit simulation context
   *
   * The default implementation ignores the context and calls
   * `apply(field, domain, box, time)`. Override this when the modifier needs
   * `simulation_context.mpi_comm()`.
   */
  virtual void apply(const SimulationContext &simulation_context,
                     pfc::field::FieldOutput<double> field, const Domain &domain,
                     const Box3i &box, double time) {
    (void)simulation_context;
    apply(field, domain, box, time);
  }

  /**
   * @brief Apply the modification to a host field over @p box (pure virtual)
   *
   * @param field Mutable host view with `box` voxel count (x-fastest, halo 0).
   *              A `std::vector<double>` lvalue or `Field::output()` converts
   *              implicitly; for a device `Field` use `apply_field_modifier`
   *              (kernel/simulation/apply_field_modifier.hpp), which brackets
   *              the host mirror.
   * @param domain Global geometry (origin, spacing, size)
   * @param box Local owned index box
   * @param time Current simulation time (typically 0 for ICs)
   */
  virtual void apply(pfc::field::FieldOutput<double> field, const Domain &domain,
                     const Box3i &box, double time = 0.0) = 0;

  /**
   * @brief Destructor for the FieldModifier class.
   *
   * The destructor is declared as default, allowing proper destruction of
   * derived classes.
   */
  virtual ~FieldModifier() = default;
};

} // namespace pfc

#endif // PFC_FIELD_MODIFIER_HPP
