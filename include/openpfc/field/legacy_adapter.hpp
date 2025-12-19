// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file legacy_adapter.hpp
 * @brief Adapter to wrap functional field ops into FieldModifier interface
 */

#pragma once

#include <memory>
#include <string>
#include <utility>

#include "openpfc/field/operations.hpp"
#include "openpfc/field_modifier.hpp"

namespace pfc {
namespace field {

/**
 * @brief Create a FieldModifier from a coordinate-space lambda
 *
 * @tparam Fn Callable signature: double(const Real3&) or double(const Real3&,
 * double)
 * @param field_name Name of the target real field in the model
 * @param fn Function mapping coordinates (and optionally time) to values
 * @return std::unique_ptr<FieldModifier> usable with existing Simulator APIs
 */
template <typename Fn>
std::unique_ptr<FieldModifier> make_legacy_modifier(std::string field_name, Fn fn) {
  struct LambdaModifier final : FieldModifier {
    std::string name;
    Fn func;
    explicit LambdaModifier(std::string n, Fn f)
        : name(std::move(n)), func(std::move(f)) {
      set_field_name(name);
    }
    void apply(Model &m, double t) override {
      // Dispatch based on callable arity (with or without time)
      if constexpr (std::is_invocable_r_v<double, Fn, const Real3 &>) {
        pfc::field::apply(m, get_field_name(), func);
      } else if constexpr (std::is_invocable_r_v<double, Fn, const Real3 &,
                                                 double>) {
        pfc::field::apply_with_time(m, get_field_name(), t, func);
      } else {
        static_assert(sizeof(Fn) == 0,
                      "Unsupported lambda signature for legacy modifier. Expected"
                      " double(Real3) or double(Real3,double)");
      }
    }
  };

  return std::make_unique<LambdaModifier>(std::move(field_name), std::move(fn));
}

} // namespace field
} // namespace pfc
