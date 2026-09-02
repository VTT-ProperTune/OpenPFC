// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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

#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

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
  double m_n0 = 0.0;

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
   * @brief Fill @p field with the constant density over @p box.
   *
   * 0.2 path: no Model / FFT. @p field must have `box` voxel count.
   */
  void apply(RealField &field, const Domain &domain, const Box3i &box,
             double time = 0.0) override {
    (void)time;
    pfc::field::apply(field, domain, box,
                      [n0 = m_n0](const pfc::Real3 &) { return n0; });
  }
};

} // namespace pfc

#endif // PFC_INITIAL_CONDITIONS_CONSTANT_HPP
