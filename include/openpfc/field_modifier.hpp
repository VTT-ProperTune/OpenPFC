// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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
 * - Boundary conditions: FixedBC, MovingBC
 *
 * Typical usage:
 * @code
 * // Define initial condition
 * class MyInitialCondition : public pfc::FieldModifier {
 * public:
 *     void apply(pfc::Model& model, double time) override {
 *         auto& field = model.get_real_field(get_field_name());
 *         // Modify field values
 *     }
 * };
 *
 * // Use in simulator
 * simulator.add_initial_condition(std::make_unique<MyInitialCondition>());
 * @endcode
 *
 * This file is part of the Field Operations module, providing mechanisms
 * for setting initial states and enforcing boundary constraints.
 *
 * @see model.hpp for field access methods
 * @see simulator.hpp for modifier application orchestration
 * @see initial_conditions/ for IC implementations
 * @see boundary_conditions/ for BC implementations
 */

#ifndef PFC_FIELD_MODIFIER_HPP
#define PFC_FIELD_MODIFIER_HPP

#include "model.hpp"

namespace pfc {

/**
 * @brief Abstract base class for field modifiers in OpenPFC
 *
 * FieldModifier provides a unified interface for modifying field values in
 * simulations, supporting both initial conditions (applied once) and boundary
 * conditions (applied repeatedly during time integration).
 *
 * **Core Concept:**
 * Field modifiers operate by implementing the pure virtual `apply(Model&, double)`
 * method, which receives direct access to the model and current time. This allows
 * modifiers to read/write field data, access FFT transforms, query decomposition,
 * and implement arbitrary modification logic.
 *
 * **Design Philosophy:**
 * - **Extensibility**: Users can create custom modifiers without touching OpenPFC
 * - **Composition**: Multiple modifiers can be applied in sequence
 * - **Transparency**: Direct model access allows full inspection and control
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
 *   void apply(pfc::Model& model, double time) override {
 *     auto& field = model.get_real_field(get_field_name());
 *     const auto& world = model.get_world();
 *     const auto& fft = model.get_fft();
 *     auto inbox = pfc::fft::get_inbox(fft);
 *
 *     int idx = 0;
 *     for (int k = inbox.low[2]; k <= inbox.high[2]; k++) {
 *       for (int j = inbox.low[1]; j <= inbox.high[1]; j++) {
 *         for (int i = inbox.low[0]; i <= inbox.high[0]; i++) {
 *           auto pos = pfc::world::to_coords(world, Int3{i, j, k});
 *           double dx = pos[0] - m_center[0];
 *           double dy = pos[1] - m_center[1];
 *           double dz = pos[2] - m_center[2];
 *           double r2 = dx*dx + dy*dy + dz*dz;
 *           field[idx++] = m_amplitude * std::exp(-r2 / (m_width*m_width));
 *         }
 *       }
 *     }
 *   }
 * };
 * @endcode
 *
 * @example Creating a custom boundary condition
 * @code
 * class DirichletBC : public pfc::FieldModifier {
 *   double m_value;
 *   double m_width;  // Transition width
 * public:
 *   DirichletBC(double value, double width = 5.0)
 *     : m_value(value), m_width(width) {}
 *
 *   void apply(pfc::Model& model, double time) override {
 *     auto& field = model.get_real_field(get_field_name());
 *     const auto& world = model.get_world();
 *     const auto& fft = model.get_fft();
 *     auto inbox = pfc::fft::get_inbox(fft);
 *
 *     double Lx = pfc::world::get_size(world, 0);
 *     double dx = pfc::world::get_spacing(world, 0);
 *
 *     int idx = 0;
 *     for (int k = inbox.low[2]; k <= inbox.high[2]; k++) {
 *       for (int j = inbox.low[1]; j <= inbox.high[1]; j++) {
 *         for (int i = inbox.low[0]; i <= inbox.high[0]; i++) {
 *           double x = i * dx;
 *           // Apply at right boundary with smooth transition
 *           if (x > Lx - m_width) {
 *             double s = (x - (Lx - m_width)) / m_width;
 *             field[idx] = field[idx] * (1.0 - s) + m_value * s;
 *           }
 *           idx++;
 *         }
 *       }
 *     }
 *   }
 * };
 * @endcode
 *
 * @example Space-time varying boundary condition
 * @code
 * class TimeVaryingBC : public pfc::FieldModifier {
 *   double m_frequency;
 * public:
 *   TimeVaryingBC(double freq) : m_frequency(freq) {}
 *
 *   void apply(pfc::Model& model, double time) override {
 *     auto& field = model.get_real_field(get_field_name());
 *     const auto& world = model.get_world();
 *     const auto& fft = model.get_fft();
 *     auto inbox = pfc::fft::get_inbox(fft);
 *
 *     // Time-varying amplitude
 *     double amplitude = std::sin(2.0 * M_PI * m_frequency * time);
 *
 *     double dx = pfc::world::get_spacing(world, 0);
 *     int idx = 0;
 *     for (int k = inbox.low[2]; k <= inbox.high[2]; k++) {
 *       for (int j = inbox.low[1]; j <= inbox.high[1]; j++) {
 *         for (int i = inbox.low[0]; i <= inbox.high[0]; i++) {
 *           if (i == 0) {  // Left boundary
 *             field[idx] = amplitude;
 *           }
 *           idx++;
 *         }
 *       }
 *     }
 *   }
 * };
 * @endcode
 *
 * @example Composing multiple modifiers
 * @code
 * // Set uniform background
 * auto constant = std::make_unique<pfc::Constant>(0.5);
 * constant->set_field_name("density");
 * simulator.add_initial_condition(std::move(constant));
 *
 * // Add localized perturbation
 * auto gaussian = std::make_unique<GaussianIC>(
 *   Real3{10.0, 10.0, 10.0}, 0.1, 2.0);
 * gaussian->set_field_name("density");
 * simulator.add_initial_condition(std::move(gaussian));
 *
 * // Enforce boundary condition every step
 * auto bc = std::make_unique<DirichletBC>(0.0, 5.0);
 * bc->set_field_name("density");
 * simulator.add_boundary_condition(std::move(bc));
 * @endcode
 *
 * @example Accessing multiple fields (for future multi-field support)
 * @code
 * class CoupledIC : public pfc::FieldModifier {
 * public:
 *   void apply(pfc::Model& model, double time) override {
 *     // Current: single field via get_field_name()
 *     auto& density = model.get_real_field(get_field_name());
 *
 *     // Future: access multiple fields
 *     // auto& temperature = model.get_real_field("temperature");
 *
 *     // Set coupled initial state
 *     std::fill(density.begin(), density.end(), 0.5);
 *   }
 * };
 * @endcode
 *
 * **Usage in Simulator:**
 * - Initial conditions: Applied once before time integration via
 *   `Simulator::add_initial_condition()`
 * - Boundary conditions: Applied every time step (or at intervals) via
 *   `Simulator::add_boundary_condition()`
 * - Application order: ICs first, then BCs, in the order added
 *
 * **Performance Considerations:**
 * - Boundary conditions are in the hot path (applied every step)
 * - Minimize allocations in `apply()`
 * - Consider caching computed values if expensive
 * - Use direct indexing (avoid coordinate transformations when possible)
 *
 * **Known Limitations:**
 * - Currently designed for single-field access via `get_field_name()`
 * - TODO: Support modifying multiple fields in one modifier (see TODO comment)
 * - No built-in support for parallel reduction operations
 *
 * @note The `time` parameter allows implementing time-dependent boundary
 *       conditions, but most initial conditions ignore it (t=0 at IC application)
 *
 * @warning Modifiers have direct mutable access to model state. Ensure your
 *          `apply()` implementation maintains physical correctness and doesn't
 *          violate model invariants.
 *
 * @see Model::get_real_field() for field access
 * @see Model::get_complex_field() for k-space operations
 * @see Simulator::add_initial_condition() for IC registration
 * @see Simulator::add_boundary_condition() for BC registration
 * @see initial_conditions/ for built-in IC implementations
 * @see boundary_conditions/ for built-in BC implementations
 */
class FieldModifier {

private:
  std::string m_field_name = "default";
  std::string m_default_name = "default";

public:
  /**
   * @brief Apply the field modification to the model at a specific time.
   *
   * This function is responsible for applying the field modification to the
   * provided model at the given time.
   *
   * @param model The model to apply the field modification to.
   * @param field_name To which field the modification is done.
   * @param time The current time.
   */
  // TODO: we need a way to modify arbitrary fields, not just default one
  // virtual void apply(Model &model, const std::string &field_name, double
  // time) = 0;

  /**
   * @brief Set the field name this modifier should operate on
   *
   * Specifies which field in the Model this modifier will access via
   * `Model::get_real_field()` or `Model::get_complex_field()`. This allows
   * the same modifier implementation to be reused for different fields.
   *
   * @param field_name Name of the field to modify (e.g., "density", "temperature")
   *
   * @example Basic usage
   * @code
   * auto ic = std::make_unique<pfc::Constant>(0.5);
   * ic->set_field_name("density");
   * simulator.add_initial_condition(std::move(ic));
   * @endcode
   *
   * @example Multiple fields with same modifier type
   * @code
   * // Apply constant IC to density field
   * auto density_ic = std::make_unique<pfc::Constant>(0.5);
   * density_ic->set_field_name("density");
   * simulator.add_initial_condition(std::move(density_ic));
   *
   * // Apply constant IC to temperature field
   * auto temp_ic = std::make_unique<pfc::Constant>(300.0);
   * temp_ic->set_field_name("temperature");
   * simulator.add_initial_condition(std::move(temp_ic));
   * @endcode
   *
   * @note The field name must match a field registered in the Model,
   *       otherwise `Model::get_real_field()` will throw
   * @note Default field name is "default" if not explicitly set
   *
   * @see get_field_name()
   * @see Model::get_real_field()
   */
  void set_field_name(const std::string &field_name) { m_field_name = field_name; }

  /**
   * @brief Get the name of the field this modifier operates on
   *
   * Returns the field name set via `set_field_name()`, which is used to
   * retrieve the appropriate field from the Model in the `apply()` method.
   *
   * @return Reference to the field name string
   *
   * @example Using in custom modifier
   * @code
   * class MyModifier : public pfc::FieldModifier {
   * public:
   *   void apply(pfc::Model& model, double time) override {
   *     // Retrieve the field this modifier should act on
   *     auto& field = model.get_real_field(get_field_name());
   *     // Modify field...
   *   }
   * };
   * @endcode
   *
   * @note Returns "default" if field name was never explicitly set
   *
   * @see set_field_name()
   */
  const std::string &get_field_name() const { return m_field_name; }

  /**
   * @brief Get the name of the field modifier.
   *
   * This function is responsible for getting the name of the field modifier.
   *
   * @return The modifier name.
   */

  virtual const std::string &get_modifier_name() const { return m_default_name; }

  /**
   * @brief Apply the field modification to the model (pure virtual)
   *
   * This is the main interface method that derived classes must implement to
   * define their modification logic. The method receives full mutable access
   * to the Model and current simulation time, allowing arbitrary modifications.
   *
   * **Implementation Responsibilities:**
   * - Retrieve field(s) via `model.get_real_field()` or `model.get_complex_field()`
   * - Access geometry via `model.get_world()` and `model.get_fft()`
   * - Modify field values according to modifier's purpose
   * - Handle MPI parallelism (operate on local subdomain)
   *
   * **Typical Implementation Pattern:**
   * @code
   * void apply(pfc::Model& model, double time) override {
   *   // 1. Get field to modify
   *   auto& field = model.get_real_field(get_field_name());
   *
   *   // 2. Get geometry information
   *   const auto& world = model.get_world();
   *   const auto& fft = model.get_fft();
   *   auto inbox = pfc::fft::get_inbox(fft);
   *
   *   // 3. Loop over local subdomain
   *   int idx = 0;
   *   for (int k = inbox.low[2]; k <= inbox.high[2]; k++) {
   *     for (int j = inbox.low[1]; j <= inbox.high[1]; j++) {
   *       for (int i = inbox.low[0]; i <= inbox.high[0]; i++) {
   *         // Compute modification based on position and/or time
   *         auto pos = pfc::world::to_coords(world, Int3{i, j, k});
   *         field[idx++] = compute_value(pos, time);
   *       }
   *     }
   *   }
   * }
   * @endcode
   *
   * @param model Mutable reference to the Model containing fields to modify
   * @param time Current simulation time (useful for time-dependent BCs)
   *
   * @note For initial conditions, `time` is typically 0.0
   * @note For boundary conditions, `time` reflects current simulation time
   * @note Method is called on every MPI rank; each rank operates on its subdomain
   *
   * @warning Ensure modifications maintain physical correctness and don't violate
   *          model invariants (e.g., mass conservation if required)
   *
   * @see Model::get_real_field() for field access
   * @see Model::get_world() for domain geometry
   * @see Model::get_fft() for subdomain bounds
   */
  virtual void apply(Model &model, double time) = 0;

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
