#pragma once

#include "model.hpp"

namespace pfc {

class FieldModifier {

public:
  virtual void apply(Model &m, Field &f, double t) = 0;
};

} // namespace pfc
