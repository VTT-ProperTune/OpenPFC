#pragma once

#include "../field_modifier.hpp"

namespace pfc {

/**
 * A class that represents a constant field modifier for use as an initial
 * condition in a partial differential equation (PDE) model.
 *
 * The `Constant` class inherits from the `FieldModifier` abstract base class
 * and overrides the `apply` method to set the field to a constant value.
 *
 * Example usage:
 *
 * ```
 * // Create a constant field modifier with value 1.0
 * Constant c(1.0);
 *
 * // Apply the constant field modifier to a model
 * Model m;
 * c.apply(m, 0.0);
 * ```
 */

class Constant : public FieldModifier {
private:
  double m_n0;

public:
  Constant(double n0) : m_n0(n0) {}

  void apply(Model &m, double) override {
    Field &field = m.get_field();
    std::fill(field.begin(), field.end(), m_n0);
  }
};

} // namespace pfc
