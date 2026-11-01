// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_robin.hpp
 * @brief JSON/TOML object for `RobinBoundary<double>`.
 *
 * @details
 * One object:
 * `{"faces": [{"face": "xm", "alpha": 1.0, "beta": 0.5, "gamma": -0.25}]}`.
 * An optional `centering` value must be `"node"`. Each face name is `xm`,
 * `xp`, `ym`, `yp`, `zm` or `zp`, used once. Coefficients are finite
 * numbers, and `(alpha, beta) = (0, 0)` is rejected. Callbacks are not
 * part of the file format.
 */

#ifndef PFC_UI_FROM_JSON_ROBIN_HPP
#define PFC_UI_FROM_JSON_ROBIN_HPP

#include <cmath>
#include <string>

#include <openpfc/frontend/ui/errors_config_format.hpp>
#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/kernel/decomposition/robin_boundary.hpp>

namespace pfc::ui {

template <>
[[nodiscard]] inline pfc::communication::RobinBoundary<double>
from_json<pfc::communication::RobinBoundary<double>>(const json &j) {
  using pfc::communication::cartesian_face_from_name;
  using pfc::communication::CartesianFace;
  using pfc::communication::RobinBoundary;
  if (!j.is_object()) {
    throw std::invalid_argument(format_config_error(
        "robin", "named Cartesian Robin faces", "object with a faces array",
        j.dump(), {},
        "{\"faces\": [{\"face\": \"xm\", \"alpha\": 1.0, \"beta\": 0.5, "
        "\"gamma\": 0.0}]}"));
  }
  if (j.contains("centering")) {
    if (!j.at("centering").is_string() ||
        j.at("centering").get<std::string>() != "node") {
      throw std::invalid_argument(
          format_config_error("robin.centering", "where the wall sits", "\"node\"",
                              j.at("centering").dump(), {"node"}, "\"node\""));
    }
  }
  if (!j.contains("faces") || !j.at("faces").is_array() || j.at("faces").empty()) {
    throw std::invalid_argument(format_config_error(
        "robin.faces", "selected Cartesian faces",
        "non-empty array of {face, alpha, beta, gamma}",
        j.contains("faces") ? j.at("faces").dump() : std::string("missing"), {},
        "[{\"face\": \"xm\", \"alpha\": 1.0, \"beta\": 0.5, \"gamma\": 0.0}]"));
  }
  RobinBoundary<double> boundary;
  for (const json &entry : j.at("faces")) {
    if (!entry.is_object() || !entry.contains("face") ||
        !entry.at("face").is_string() || !entry.contains("alpha") ||
        !entry.at("alpha").is_number() || !entry.contains("beta") ||
        !entry.at("beta").is_number() || !entry.contains("gamma") ||
        !entry.at("gamma").is_number()) {
      throw std::invalid_argument(format_config_error(
          "robin.faces", "one Cartesian face",
          "object {\"face\": \"xm\", \"alpha\": 1.0, \"beta\": 0.5, \"gamma\": 0.0}",
          entry.dump(), {"xm", "xp", "ym", "yp", "zm", "zp"},
          "{\"face\": \"xm\", \"alpha\": 1.0, \"beta\": 0.5, \"gamma\": 0.0}"));
    }
    const std::string face_name = entry.at("face").get<std::string>();
    CartesianFace face;
    try {
      face = cartesian_face_from_name(face_name);
    } catch (const std::invalid_argument &) {
      throw std::invalid_argument(format_config_error(
          "robin.faces.face", "Cartesian face name", "xm, xp, ym, yp, zm or zp",
          face_name, {"xm", "xp", "ym", "yp", "zm", "zp"}, "\"xm\""));
    }
    if (boundary.enabled(face)) {
      throw std::invalid_argument(
          format_config_error("robin.faces.face", "Cartesian face name",
                              "a face used once", face_name, {}, "\"xm\""));
    }
    const double alpha = entry.at("alpha").get<double>();
    const double beta = entry.at("beta").get<double>();
    const double gamma = entry.at("gamma").get<double>();
    if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(gamma)) {
      throw std::invalid_argument(format_config_error(
          "robin.faces", "Robin coefficients", "finite alpha, beta and gamma",
          entry.dump(), {},
          "{\"face\": \"xm\", \"alpha\": 1.0, \"beta\": 0.5, \"gamma\": 0.0}"));
    }
    try {
      boundary.set_constant(face, alpha, beta, gamma);
    } catch (const std::invalid_argument &) {
      throw std::invalid_argument(format_config_error(
          "robin.faces", "Robin coefficients", "alpha and beta not both zero",
          entry.dump(), {},
          "{\"face\": \"xm\", \"alpha\": 1.0, \"beta\": 0.5, \"gamma\": 0.0}"));
    }
  }
  return boundary;
}

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_ROBIN_HPP
