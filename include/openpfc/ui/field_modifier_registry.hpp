// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/field_modifier_registry.hpp
 * @brief Registry for field modifiers (initial conditions and boundary conditions)
 *
 * @details
 * This header provides a registry system for field modifiers that allows
 * dynamic registration and creation of field modifiers from JSON configuration.
 * It supports both initial conditions and boundary conditions.
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UI_FIELD_MODIFIER_REGISTRY_HPP
#define PFC_UI_FIELD_MODIFIER_REGISTRY_HPP

#include "errors.hpp"
#include "from_json.hpp"
#include "openpfc/boundary_conditions/fixed_bc.hpp"
#include "openpfc/boundary_conditions/moving_bc.hpp"
#include "openpfc/field_modifier.hpp"
#include "openpfc/initial_conditions/constant.hpp"
#include "openpfc/initial_conditions/file_reader.hpp"
#include "openpfc/initial_conditions/random_seeds.hpp"
#include "openpfc/initial_conditions/seed_grid.hpp"
#include "openpfc/initial_conditions/single_seed.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace pfc {
namespace ui {

using FieldModifier_p = std::unique_ptr<FieldModifier>;

/**
 * @class FieldModifierRegistry
 * @brief A registry for field modifiers used in the application.
 *
 * The FieldModifierRegistry class provides a centralized registry for field
 * modifiers. It allows registration of field modifiers along with their
 * corresponding creator functions, and provides a way to create instances of
 * field modifiers based on their registered types.
 */
class FieldModifierRegistry {
public:
  using CreatorFunction = std::function<FieldModifier_p(const json &)>;

  /**
   * @brief Get the singleton instance of the FieldModifierRegistry.
   * @return Reference to the singleton instance of FieldModifierRegistry.
   */
  static FieldModifierRegistry &get_instance() {
    static FieldModifierRegistry instance;
    return instance;
  }

  /**
   * @brief Register a field modifier with its creator function.
   * @param type The type string associated with the field modifier.
   * @param creator The creator function that creates an instance of the field
   * modifier.
   */
  void register_modifier(const std::string &type, CreatorFunction creator) {
    modifiers[type] = creator;
  }

  /**
   * @brief Create an instance of a field modifier based on its registered type.
   * @param type The type string of the field modifier to create.
   * @param data A json object defining the field modifier parameters.
   * @return Pointer to the created field modifier instance.
   * @throw std::invalid_argument if the specified type is not registered.
   */
  FieldModifier_p create_modifier(const std::string &type, const json &data) {
    auto it = modifiers.find(type);
    if (it != modifiers.end()) {
      return it->second(data);
    }
    throw std::invalid_argument(format_unknown_modifier_error(type));
  }

private:
  /**
   * @brief Private constructor to enforce singleton pattern.
   */
  FieldModifierRegistry() {}

  std::unordered_map<std::string, CreatorFunction>
      modifiers; /**< Map storing the registered field modifiers and their
                    creator functions. */
};

/**
 * @brief Register a field modifier type with the FieldModifierRegistry.
 * @tparam T The type of the field modifier to register.
 * @param type The type string associated with the field modifier.
 *
 * This function registers a field modifier type with the FieldModifierRegistry.
 * It associates the specified type string with a creator function that creates
 * an instance of the field modifier.
 */
template <typename T> void register_field_modifier(const std::string &type) {
  FieldModifierRegistry::get_instance().register_modifier(
      type, [](const json &params) -> std::unique_ptr<T> {
        std::unique_ptr<T> modifier = std::make_unique<T>();
        from_json(params, *modifier);
        return modifier;
      });
}

/**
 * @brief Create an instance of a field modifier based on its type.
 * @param type The type string of the field modifier to create.
 * @param params A json object describing the parameters for field modifier.
 * @return Pointer to the created field modifier instance.
 * @throw std::invalid_argument if the specified type is not registered.
 *
 * This function creates an instance of a field modifier based on its registered
 * type. It retrieves the registered creator function associated with the
 * specified type string from the FieldModifierRegistry and uses it to create
 * the field modifier instance.
 */
std::unique_ptr<FieldModifier> create_field_modifier(const std::string &type,
                                                     const json &params) {
  return FieldModifierRegistry::get_instance().create_modifier(type, params);
}

/**
 * @struct FieldModifierInitializer
 * @brief Helper struct for registering field modifiers during static
 * initialization.
 *
 * The FieldModifierInitializer struct provides a convenient way to register
 * field modifiers during static initialization by utilizing its constructor.
 * Inside the constructor, various field modifiers can be registered using the
 * `register_field_modifier` function.
 */
struct FieldModifierInitializer {
  /**
   * @brief Constructor for FieldModifierInitializer.
   *
   * This constructor is automatically executed during static initialization.
   * It can be used to register field modifiers by calling the
   * `register_field_modifier` function for each desired field modifier type.
   */
  FieldModifierInitializer() {
    // Initial conditions
    register_field_modifier<Constant>("constant");
    register_field_modifier<SingleSeed>("single_seed");
    register_field_modifier<RandomSeeds>("random_seeds");
    register_field_modifier<SeedGrid>("seed_grid");
    register_field_modifier<FileReader>("from_file");
    // Boundary conditions
    register_field_modifier<FixedBC>("fixed");
    register_field_modifier<MovingBC>("moving");
    // Register other field modifiers here ...
  }
};

static FieldModifierInitializer
    fieldModifierInitializer; /**< Static instance of FieldModifierInitializer
                                 to trigger field modifier registration during
                                 static initialization. */

} // namespace ui
} // namespace pfc

#endif // PFC_UI_FIELD_MODIFIER_REGISTRY_HPP
