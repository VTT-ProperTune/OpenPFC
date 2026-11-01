// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_dirichlet.hpp
 * @brief JSON/TOML object for `DirichletBoundary<double>`.
 *
 * @details
 * One object:
 * `{"centering": "node", "faces": [{"face": "zm", "value": 0.0}]}`.
 * `centering` is optional (`node` or `cell`). Each face name is `xm`,
 * `xp`, `ym`, `yp`, `zm` or `zp`, used once. Values are finite numbers.
 * Callbacks are not part of the file format.
 */

#ifndef PFC_UI_FROM_JSON_DIRICHLET_HPP
#define PFC_UI_FROM_JSON_DIRICHLET_HPP

#include <cmath>
#include <string>

#include <openpfc/frontend/ui/errors_config_format.hpp>
#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/kernel/decomposition/dirichlet_boundary.hpp>

namespace pfc::ui {

template <>
[[nodiscard]] inline pfc::communication::DirichletBoundary<double>
from_json<pfc::communication::DirichletBoundary<double>>(const json &j) {
  using pfc::communication::cartesian_face_from_name;
  using pfc::communication::CartesianFace;
  using pfc::communication::DirichletBoundary;
  using pfc::communication::DirichletCentering;
  if (!j.is_object()) {
    throw std::invalid_argument(format_config_error(
        "dirichlet", "named Cartesian Dirichlet faces", "object with a faces array",
        j.dump(), {}, "{\"faces\": [{\"face\": \"zm\", \"value\": 0.0}]}"));
  }
  DirichletBoundary<double> boundary;
  if (j.contains("centering")) {
    if (!j.at("centering").is_string()) {
      throw std::invalid_argument(format_config_error(
          "dirichlet.centering", "where the wall sits", "\"node\" or \"cell\"",
          j.at("centering").dump(), {"node", "cell"}, "\"node\""));
    }
    const std::string centering = j.at("centering").get<std::string>();
    if (centering == "node") {
      boundary.centering = DirichletCentering::Node;
    } else if (centering == "cell") {
      boundary.centering = DirichletCentering::CellFace;
    } else {
      throw std::invalid_argument(format_config_error(
          "dirichlet.centering", "where the wall sits", "\"node\" or \"cell\"",
          centering, {"node", "cell"}, "\"node\""));
    }
  }
  if (!j.contains("faces") || !j.at("faces").is_array() || j.at("faces").empty()) {
    throw std::invalid_argument(format_config_error(
        "dirichlet.faces", "selected Cartesian faces",
        "non-empty array of {face, value}",
        j.contains("faces") ? j.at("faces").dump() : std::string("missing"), {},
        "[{\"face\": \"zm\", \"value\": 0.0}]"));
  }
  for (const json &entry : j.at("faces")) {
    if (!entry.is_object() || !entry.contains("face") ||
        !entry.at("face").is_string() || !entry.contains("value") ||
        !entry.at("value").is_number()) {
      throw std::invalid_argument(
          format_config_error("dirichlet.faces", "one Cartesian face",
                              "object {\"face\": \"zm\", \"value\": 0.0}",
                              entry.dump(), {"xm", "xp", "ym", "yp", "zm", "zp"},
                              "{\"face\": \"zm\", \"value\": 0.0}"));
    }
    const std::string face_name = entry.at("face").get<std::string>();
    CartesianFace face;
    try {
      face = cartesian_face_from_name(face_name);
    } catch (const std::invalid_argument &) {
      throw std::invalid_argument(format_config_error(
          "dirichlet.faces.face", "Cartesian face name", "xm, xp, ym, yp, zm or zp",
          face_name, {"xm", "xp", "ym", "yp", "zm", "zp"}, "\"zm\""));
    }
    if (boundary.enabled(face)) {
      throw std::invalid_argument(
          format_config_error("dirichlet.faces.face", "Cartesian face name",
                              "a face used once", face_name, {}, "\"zm\""));
    }
    const double value = entry.at("value").get<double>();
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          format_config_error("dirichlet.faces.value", "Dirichlet value",
                              "finite number", entry.at("value").dump(), {}, "0.0"));
    }
    boundary.set_constant(face, value);
  }
  return boundary;
}

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_DIRICHLET_HPP
