/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

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
  // virtual void apply(Model &model, const std::string &field_name, double time) = 0;

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
