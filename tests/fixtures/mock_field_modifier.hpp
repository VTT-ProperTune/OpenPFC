// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <string>

#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace pfc {
namespace testing {

class MockFieldModifier : public FieldModifier {
public:
  bool applied = false;
  void apply(pfc::field::FieldOutput<double> /*field*/, const Domain & /*domain*/,
             const Box3i & /*box*/,
             double /*time*/) override {
    applied = true;
  }
};

class MockIC : public FieldModifier {
public:
  void apply(pfc::field::FieldOutput<double> field, const Domain & /*domain*/, const Box3i & /*box*/,
             double /*time*/) override {
    std::fill(field.begin(), field.end(), 1.0);
  }
};

} // namespace testing
} // namespace pfc
