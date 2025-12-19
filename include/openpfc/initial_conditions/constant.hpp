// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file constant.hpp
 * @brief Constant value initial condition
 *
 * @details
 * This file defines the Constant class, which sets all field values to a
 * uniform constant value. This is the simplest initial condition, useful for:
 * - Homogeneous starting states
 * - Testing and validation
 * - Baseline conditions before perturbations
 *
 * Usage:
 * @code
 * auto ic = std::make_unique<pfc::Constant>(0.5);
 * ic->set_field_name("density");
 * simulator.add_initial_condition(std::move(ic));
 * @endcode
 *
 * @see field_modifier.hpp for base class
 * @see simulator.hpp for how initial conditions are applied
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_INITIAL_CONDITIONS_CONSTANT_HPP
#define PFC_INITIAL_CONDITIONS_CONSTANT_HPP

#include "../field_modifier.hpp"
#include "openpfc/field/operations.hpp"

namespace pfc {

/**
 * @brief A class that represents a constant field modifier for use as an
 * initial condition in a partial differential equation (PDE) model.
 *
 * The `Constant` class inherits from the `FieldModifier` abstract base class
 * and overrides the `apply` method to set the field to a constant value.
 */
class Constant : public FieldModifier {
private:
  double m_n0;

public:
  /**
   * @brief Default constructor for the Constant class.
   */
  Constant() = default;

  /**
   * @brief Constructor for the Constant class that sets the initial density
   * value.
   * @param n0 The constant value to set for the field.
   */
  Constant(double n0) : m_n0(n0) {}

  /**
   * @brief Get the current density value.
   * @return The density value.
   */
  double get_density() const { return m_n0; }

  /**
   * @brief Set the density value.
   * @param n0 The new density value to set.
   */
  void set_density(double n0) { m_n0 = n0; }

  /**
   * @brief Apply the constant field modifier to the given model.
   *
   * This method sets the field in the model to the constant density value.
   *
   * @param m The model to apply the field modifier to.
   * @param t The current time (unused in this implementation).
   */
  void apply(Model &m, double t_unused) override {
    (void)t_unused; // Silence unused parameter warning
    // Apply constant in coordinate space over the local inbox
    // Preserves distributed behavior and keeps API consistent with new ops
    pfc::field::apply(m, get_field_name(),
                      [n0 = m_n0](const pfc::Real3 &) { return n0; });
  }
};

} // namespace pfc

#endif // PFC_INITIAL_CONDITIONS_CONSTANT_HPP
