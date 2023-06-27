#pragma once

#include "model.hpp"

namespace pfc {

/**
 * @brief The FieldModifier class represents a base class for field modifiers in PFC.
 *
 * Field modifiers are used to apply modifications to the fields in a PFC model at specific times.
 * Users can derive from this class and implement the `apply` function to define their own field modification logic.
 */
class FieldModifier {

public:
  /**
   * @brief Apply the field modification to the model at a specific time.
   *
   * This function is responsible for applying the field modification to the provided model at the given time.
   *
   * @param m The model to apply the field modification to.
   * @param t The current time.
   */
  virtual void apply(Model &m, double t) = 0;

  /**
   * @brief Destructor for the FieldModifier class.
   *
   * The destructor is declared as default, allowing proper destruction of derived classes.
   */
  virtual ~FieldModifier() = default;
};

} // namespace pfc
