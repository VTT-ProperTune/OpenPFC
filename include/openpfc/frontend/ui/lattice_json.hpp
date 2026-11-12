// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file lattice_json.hpp
 * @brief The optional `lattice` key (`"bcc"` or `"fcc"`) in JSON/TOML inputs.
 *
 * @details
 * Two places read the key: a model's `model.params` (the physics chooses the
 * correlation-kernel shells) and the library seed initial conditions
 * `single_seed`, `seed_grid` and `random_seeds` (the seed chooses its wave
 * vectors). The acceptance rule is that both see the same lattice, so the
 * session copies the model's lattice into every seed that leaves the key out
 * and rejects a seed that names a different one. Unknown names and
 * non-string values fail closed.
 */

#ifndef PFC_UI_LATTICE_JSON_HPP
#define PFC_UI_LATTICE_JSON_HPP

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include <openpfc/kernel/simulation/lattice.hpp>

namespace pfc::ui {

/**
 * @brief Read an optional `lattice` key from a JSON object.
 *
 * @param context Prefix for the error message, e.g. `"single_seed"`.
 * @return The lattice, or `std::nullopt` when the key is absent.
 * @throws std::invalid_argument when the key is present but not `"bcc"` or
 *         `"fcc"`.
 */
[[nodiscard]] inline std::optional<Lattice>
lattice_from_json(const nlohmann::json &j, std::string_view context) {
  if (!j.contains("lattice")) return std::nullopt;
  const nlohmann::json &v = j["lattice"];
  if (!v.is_string()) {
    throw std::invalid_argument(std::string(context) +
                                ": 'lattice' must be one of bcc, fcc.");
  }
  try {
    return lattice_from_string(v.get<std::string>());
  } catch (const std::invalid_argument &e) {
    throw std::invalid_argument(std::string(context) + ": " + e.what());
  }
}

/** @brief Built-in initial conditions whose geometry depends on the lattice. */
[[nodiscard]] inline constexpr std::array<std::string_view, 3>
lattice_seed_modifier_types() noexcept {
  return {"single_seed", "seed_grid", "random_seeds"};
}

/**
 * @brief Make every lattice-aware seed in `initial_conditions` use the model's
 *        lattice.
 *
 * A seed without a `lattice` key receives the model's; a seed naming a
 * different lattice is an error, because its geometry and the operator's
 * peaks would then describe two crystals.
 *
 * @throws std::invalid_argument on a mismatch or an invalid key.
 */
inline void apply_model_lattice_to_initial_conditions(nlohmann::json &settings,
                                                      Lattice model_lattice) {
  if (!settings.contains("initial_conditions") ||
      !settings["initial_conditions"].is_array()) {
    return;
  }
  for (nlohmann::json &ic : settings["initial_conditions"]) {
    if (!ic.is_object() || !ic.contains("type") || !ic["type"].is_string()) continue;
    const std::string type = ic["type"].get<std::string>();
    bool lattice_aware = false;
    for (std::string_view t : lattice_seed_modifier_types()) {
      lattice_aware = lattice_aware || (t == type);
    }
    if (!lattice_aware) continue;
    const std::optional<Lattice> seed_lattice = lattice_from_json(ic, type);
    if (!seed_lattice) {
      ic["lattice"] = std::string(to_string(model_lattice));
    } else if (*seed_lattice != model_lattice) {
      throw std::invalid_argument(
          "initial condition '" + type + "' sets lattice '" +
          std::string(to_string(*seed_lattice)) + "' but model.params.lattice is '" +
          std::string(to_string(model_lattice)) + "'; the seed geometry and the " +
          "linear operator must describe the same crystal.");
    }
  }
}

} // namespace pfc::ui

#endif // PFC_UI_LATTICE_JSON_HPP
