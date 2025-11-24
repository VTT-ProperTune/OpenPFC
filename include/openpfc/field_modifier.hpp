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
 * @brief The FieldModifier class represents a base class for field modifiers in
 * PFC.
 *
 * Field modifiers are used to apply modifications to the fields in a PFC model
 * at specific times. Users can derive from this class and implement the `apply`
 * function to define their own field modification logic.
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
   * @brief Set the field name for the field modifier.
   *
   * This function is responsible for setting the field name for the field
   * modifier.
   *
   * @param field_name The field name to set.
   */
  void set_field_name(const std::string &field_name) { m_field_name = field_name; }

  /**
   * @brief Get the field name for the field modifier.
   *
   * This function is responsible for getting the field name for the field
   * modifier.
   *
   * @return The field name.
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
   * @brief Apply the field modification to the model at a specific time.
   *
   * This function is responsible for applying the field modification to the
   * provided model at the given time.
   *
   * @param model The model to apply the field modification to.
   * @param time The current time.
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
