// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulator_modifier_registration.hpp
 * @brief Shared validation when appending `FieldModifier`s to IC/BC lists
 *
 * @details
 * Keeps rank-0 warning strings and “all target fields exist on `Model`” logic
 * in one place so `Simulator` stays thinner and tests can reuse the same rules
 * without duplicating string literals (DRY).
 */

#ifndef PFC_KERNEL_SIMULATION_SIMULATOR_MODIFIER_REGISTRATION_HPP
#define PFC_KERNEL_SIMULATION_SIMULATOR_MODIFIER_REGISTRATION_HPP

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/model.hpp>

namespace pfc {

/** @brief Warning fragments for `try_push_field_modifier_with_model_check` */
struct FieldModifierRegistrationMessages {
  std::string_view default_field_usage_warning;
  std::string_view missing_field_intro;
  std::string_view not_applied_suffix;
};

/** Built-in messages for `Simulator::add_initial_conditions` */
inline constexpr FieldModifierRegistrationMessages
    k_initial_condition_registration_msg{
        "Warning: adding initial condition to modify field 'default'",
        "Warning: tried to add initial condition for inexistent field ",
        ", INITIAL CONDITIONS ARE NOT APPLIED!"};

/** Built-in messages for `Simulator::add_boundary_conditions` */
inline constexpr FieldModifierRegistrationMessages
    k_boundary_condition_registration_msg{
        "Warning: adding boundary condition to modify field 'default'",
        "Warning: tried to add boundary condition for inexistent field ",
        ", BOUNDARY CONDITIONS ARE NOT APPLIED!"};

/**
 * @brief Append modifier to @p bucket if every `get_field_names()` entry exists on
 *        `model`; otherwise warn on rank 0 and return false.
 *
 * @param warn_rank0 Callable invoked only on rank-0 path (e.g. `Simulator` logger).
 */
template <typename WarnRank0>
bool try_push_field_modifier_with_model_check(
    Model &model, std::vector<std::unique_ptr<FieldModifier>> &bucket,
    std::unique_ptr<FieldModifier> modifier,
    const FieldModifierRegistrationMessages &msg, WarnRank0 &&warn_rank0) {
  for (const std::string &field_name : modifier->get_field_names()) {
    if (field_name == "default") {
      std::forward<WarnRank0>(warn_rank0)(
          std::string(msg.default_field_usage_warning));
    }
    if (!pfc::has_field(model, field_name)) {
      std::forward<WarnRank0>(warn_rank0)(std::string(msg.missing_field_intro) +
                                          field_name +
                                          std::string(msg.not_applied_suffix));
      return false;
    }
  }
  bucket.push_back(std::move(modifier));
  return true;
}

} // namespace pfc

#endif // PFC_KERNEL_SIMULATION_SIMULATOR_MODIFIER_REGISTRATION_HPP
