// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_affine_map.hpp
 * @brief JSON/TOML object for a constant `AffineMap`.
 *
 * @details
 * Columns are the images of the reference basis expressed in physical
 * Cartesian coordinates `x`, not reference-coordinate components `ξ`
 * or a row-major physical matrix written by eye:
 * `{"columns": [[2, 0, 0], [0.5, 1, 0], [0, 0, 3]], "translation": [1, -2, 4]}`.
 * `translation` is optional and defaults to zero. A singular or non-finite
 * matrix is rejected. This object does not select a physics model.
 */

#ifndef PFC_UI_FROM_JSON_AFFINE_MAP_HPP
#define PFC_UI_FROM_JSON_AFFINE_MAP_HPP

#include <openpfc/frontend/ui/errors_config_format.hpp>
#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/kernel/geometry/affine_map.hpp>

#include <string>
#include <utility>

namespace pfc::ui {

[[nodiscard]] inline json affine_map_to_json(const pfc::geometry::AffineMap &map) {
  json columns = json::array();
  for (int j = 0; j < 3; ++j) {
    const pfc::geometry::Vec3 column = map.column(j);
    columns.push_back(json::array({column.x, column.y, column.z}));
  }
  return json{{"columns", std::move(columns)},
              {"translation", json::array({map.translation[0], map.translation[1],
                                           map.translation[2]})}};
}

template <>
[[nodiscard]] inline pfc::geometry::AffineMap
from_json<pfc::geometry::AffineMap>(const json &j) {
  if (!j.is_object() || !j.contains("columns") || !j.at("columns").is_array() ||
      j.at("columns").size() != 3) {
    throw std::invalid_argument(format_config_error(
        "affine_map.columns", "images of the three reference basis vectors",
        "array of three [x, y, z] columns",
        j.is_object() && j.contains("columns") ? j.at("columns").dump()
                                               : std::string("missing"),
        {}, "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"));
  }
  double matrix[9];
  for (int col = 0; col < 3; ++col) {
    const json &column = j.at("columns").at(col);
    if (!column.is_array() || column.size() != 3) {
      throw std::invalid_argument(format_config_error(
          "affine_map.columns", "one reference basis image",
          "three finite components [x, y, z]", column.dump(), {}, "[1, 0, 0]"));
    }
    for (int row = 0; row < 3; ++row) {
      if (!column.at(row).is_number()) {
        throw std::invalid_argument(format_config_error(
            "affine_map.columns", "one matrix component", "a finite number",
            column.at(row).dump(), {}, "1.0"));
      }
      matrix[col * 3 + row] = column.at(row).get<double>();
    }
  }
  pfc::geometry::Vec3 translation{0, 0, 0};
  if (j.contains("translation")) {
    const json &shift = j.at("translation");
    if (!shift.is_array() || shift.size() != 3 || !shift.at(0).is_number() ||
        !shift.at(1).is_number() || !shift.at(2).is_number()) {
      throw std::invalid_argument(format_config_error(
          "affine_map.translation", "physical translation b",
          "three finite components", shift.dump(), {}, "[0, 0, 0]"));
    }
    translation = {shift.at(0).get<double>(), shift.at(1).get<double>(),
                   shift.at(2).get<double>()};
  }
  try {
    return pfc::geometry::affine_map_from_matrix(matrix, translation);
  } catch (const std::invalid_argument &error) {
    throw std::invalid_argument(
        format_config_error("affine_map", "a nonsingular constant map x = A xi + b",
                            "finite columns with nonzero determinant", error.what(),
                            {}, "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"));
  }
}

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_AFFINE_MAP_HPP
