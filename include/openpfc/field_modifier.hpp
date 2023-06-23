#pragma once

#include "model.hpp"

namespace pfc {

class FieldModifier {

public:
  virtual void apply(Model &m, double t) = 0;
  virtual ~FieldModifier() = default;
};

} // namespace pfc
