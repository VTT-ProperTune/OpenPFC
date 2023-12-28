#pragma once

#include "../field_modifier.hpp"

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
   * @brief Constructor for the Constant class that sets the initial density value.
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
  void apply(Model &m, double) override {
    Field &field = m.get_real_field(get_field_name());
    std::fill(field.begin(), field.end(), m_n0);
  }
};

} // namespace pfc
