// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/field_modifier_registry.hpp
 * @brief Field modifier catalog: register and create IC/BC modifiers from JSON
 *
 * @details
 * `FieldModifierCatalog` holds type string → factory mappings.
 *
 * **Default catalog:** `default_field_modifier_catalog()` returns a
 * function-local static (process-wide singleton): built-ins plus anything
 * registered via `register_field_modifier<T>(type)` without an explicit catalog.
 * It is **not** thread-safe to mutate from multiple threads; typical MPI apps
 * register custom types once from rank 0 before wiring.
 *
 * **Dependency injection:** Pass a `FieldModifierCatalog` (e.g. a copy from
 * `make_builtin_field_modifier_catalog()` with extra `register_modifier` calls)
 * into `add_*_conditions_from_json` / `wire_simulator_and_runtime_from_json` /
 * `SpectralSimulationSession` so tests and alternate drivers avoid touching the
 * global default.
 *
 * The historical name `FieldModifierRegistry` is a type alias for
 * `FieldModifierCatalog`.
 */

#ifndef PFC_UI_FIELD_MODIFIER_REGISTRY_HPP
#define PFC_UI_FIELD_MODIFIER_REGISTRY_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <openpfc/frontend/ui/errors.hpp>
#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/kernel/simulation/boundary_conditions/fixed_bc.hpp>
#include <openpfc/kernel/simulation/boundary_conditions/moving_bc.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>
#include <openpfc/kernel/simulation/initial_conditions/file_reader.hpp>
#include <openpfc/kernel/simulation/initial_conditions/random_seeds.hpp>
#include <openpfc/kernel/simulation/initial_conditions/seed_grid.hpp>
#include <openpfc/kernel/simulation/initial_conditions/single_seed.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pfc::ui {

using FieldModifier_p = std::unique_ptr<FieldModifier>;

/**
 * @brief Mutable catalog of field modifier factories (IC/BC types from JSON)
 */
class FieldModifierCatalog {
public:
  using CreatorFunction = std::function<FieldModifier_p(const json &)>;

  FieldModifierCatalog() = default;

  void register_modifier(const std::string &type, CreatorFunction creator) {
    modifiers[type] = std::move(creator);
  }

  /**
   * @brief All registered type strings (sorted for stable diagnostics).
   */
  [[nodiscard]] std::vector<std::string> registered_modifier_types() const {
    std::vector<std::string> out;
    out.reserve(modifiers.size());
    for (const auto &kv : modifiers) {
      out.push_back(kv.first);
    }
    std::sort(out.begin(), out.end());
    return out;
  }

  /**
   * @throw std::invalid_argument if @p type is not registered
   */
  [[nodiscard]] FieldModifier_p create_modifier(const std::string &type,
                                                const json &data) const {
    auto it = modifiers.find(type);
    if (it != modifiers.end()) {
      return it->second(data);
    }
    throw std::invalid_argument(format_unknown_modifier_error(type));
  }

private:
  std::unordered_map<std::string, CreatorFunction> modifiers;
};

template <typename T>
void register_field_modifier(const std::string &type,
                             FieldModifierCatalog &catalog) {
  catalog.register_modifier(type, [](const json &params) -> std::unique_ptr<T> {
    std::unique_ptr<T> modifier = std::make_unique<T>();
    from_json(params, *modifier);
    return modifier;
  });
}

/** @brief Built-in OpenPFC modifier types (constant, seeds, file, fixed, moving). */
[[nodiscard]] inline FieldModifierCatalog make_builtin_field_modifier_catalog() {
  FieldModifierCatalog c;
  register_field_modifier<Constant>("constant", c);
  register_field_modifier<SingleSeed>("single_seed", c);
  register_field_modifier<RandomSeeds>("random_seeds", c);
  register_field_modifier<SeedGrid>("seed_grid", c);
  register_field_modifier<FileReader>("from_file", c);
  register_field_modifier<FixedBC>("fixed", c);
  register_field_modifier<MovingBC>("moving", c);
  return c;
}

/**
 * @brief Process-wide catalog: builtins plus any `register_field_modifier` calls
 *
 * Used by default `create_field_modifier` and JSON wiring when no catalog is
 * passed explicitly.
 */
[[nodiscard]] inline FieldModifierCatalog &default_field_modifier_catalog() {
  static FieldModifierCatalog instance = make_builtin_field_modifier_catalog();
  return instance;
}

/**
 * @brief Register a type on the process-wide default catalog (application
 * extensions)
 */
template <typename T> void register_field_modifier(const std::string &type) {
  register_field_modifier<T>(type, default_field_modifier_catalog());
}

[[nodiscard]] inline std::unique_ptr<FieldModifier>
create_field_modifier(const std::string &type, const json &params,
                      const FieldModifierCatalog &catalog) {
  return catalog.create_modifier(type, params);
}

[[nodiscard]] inline std::unique_ptr<FieldModifier>
create_field_modifier(const std::string &type, const json &params) {
  return create_field_modifier(type, params, default_field_modifier_catalog());
}

/** @brief Historical name; prefer `FieldModifierCatalog` in new code. */
using FieldModifierRegistry = FieldModifierCatalog;

} // namespace pfc::ui

#endif // PFC_UI_FIELD_MODIFIER_REGISTRY_HPP
