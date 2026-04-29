// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_world_time.hpp
 * @brief `from_json` specializations for `World` and `Time`
 */

#ifndef PFC_UI_FROM_JSON_WORLD_TIME_HPP
#define PFC_UI_FROM_JSON_WORLD_TIME_HPP

#include <stdexcept>
#include <string>

#include <openpfc/frontend/ui/errors_config_format.hpp>
#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/frontend/ui/json_helpers.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace pfc::ui {

/**
 * Creates a World object from a JSON input.
 *
 * @param j A JSON object containing the following fields:
 *   - Lx (int): The number of grid points in the x direction.
 *   - Ly (int): The number of grid points in the y direction.
 *   - Lz (int): The number of grid points in the z direction.
 *   - dx (float): The grid spacing in the x direction.
 *   - dy (float): The grid spacing in the y direction.
 *   - dz (float): The grid spacing in the z direction.
 *   - origin (string): The origin of the coordinate system. Must be one of
 *     "center" or "corner".
 *
 * @return A World object.
 *
 * @throws std::invalid_argument if any of the required fields are missing
 *         or have an invalid value.
 */
template <> [[nodiscard]] inline World from_json<World>(const json &j) {
  int Lx = 0;
  int Ly = 0;
  int Lz = 0;
  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;
  double x0 = 0.0;
  double y0 = 0.0;
  double z0 = 0.0;
  std::string origin;

  // Use helper to support both flat and nested structures
  auto lx_val = get_json_value(j, "Lx", "domain");
  if (lx_val.is_null() || !lx_val.is_number_integer()) {
    throw std::invalid_argument(format_config_error(
        "Lx", "number of grid points in X direction", "positive integer",
        get_json_value_string(j, "Lx"), {}, "\"Lx\": 256"));
  }
  Lx = lx_val;

  auto ly_val = get_json_value(j, "Ly", "domain");
  if (ly_val.is_null() || !ly_val.is_number_integer()) {
    std::string ly_str =
        ly_val.is_null() ? "missing" : get_json_value_string(j, "Ly");
    throw std::invalid_argument(
        format_config_error("Ly", "number of grid points in Y direction",
                            "positive integer", ly_str, {}, "\"Ly\": 256"));
  }
  Ly = ly_val;

  auto lz_val = get_json_value(j, "Lz", "domain");
  if (lz_val.is_null() || !lz_val.is_number_integer()) {
    std::string lz_str =
        lz_val.is_null() ? "missing" : get_json_value_string(j, "Lz");
    throw std::invalid_argument(
        format_config_error("Lz", "number of grid points in Z direction",
                            "positive integer", lz_str, {}, "\"Lz\": 256"));
  }
  Lz = lz_val;

  auto dx_val = get_json_value(j, "dx", "domain");
  if (dx_val.is_null() || !dx_val.is_number_float()) {
    std::string dx_str =
        dx_val.is_null() ? "missing" : get_json_value_string(j, "dx");
    throw std::invalid_argument(
        format_config_error("dx", "grid spacing in X direction", "positive float",
                            dx_str, {}, "\"dx\": 1.0"));
  }
  dx = dx_val;

  auto dy_val = get_json_value(j, "dy", "domain");
  if (dy_val.is_null() || !dy_val.is_number_float()) {
    std::string dy_str =
        dy_val.is_null() ? "missing" : get_json_value_string(j, "dy");
    throw std::invalid_argument(
        format_config_error("dy", "grid spacing in Y direction", "positive float",
                            dy_str, {}, "\"dy\": 1.0"));
  }
  dy = dy_val;

  auto dz_val = get_json_value(j, "dz", "domain");
  if (dz_val.is_null() || !dz_val.is_number_float()) {
    std::string dz_str =
        dz_val.is_null() ? "missing" : get_json_value_string(j, "dz");
    throw std::invalid_argument(
        format_config_error("dz", "grid spacing in Z direction", "positive float",
                            dz_str, {}, "\"dz\": 1.0"));
  }
  dz = dz_val;

  // Support both "origin" (new) and "origo" (legacy) for backward compatibility
  auto origin_val = get_json_value(j, "origin", "domain");
  if (origin_val.is_null()) {
    origin_val = get_json_value(j, "origo", "domain");
  }
  if (origin_val.is_null() || !origin_val.is_string()) {
    std::string origin_key = j.contains("origin") ? "origin" : "origo";
    std::string origin_str =
        origin_val.is_null() ? "missing" : get_json_value_string(j, origin_key);
    throw std::invalid_argument(format_config_error(
        origin_key, "coordinate system origin", "string ('center' or 'corner')",
        origin_str, {"center", "corner"}, R"("origin": "center")"));
  }
  origin = origin_val;

  std::string origin_key = j.contains("origin") ? "origin" : "origo";
  if (origin != "center" && origin != "corner") {
    throw std::invalid_argument(format_config_error(
        origin_key, "coordinate system origin", "string ('center' or 'corner')",
        "\"" + origin + "\"", {"center", "corner"}, R"("origin": "center")"));
  }

  if (origin == "center") {
    x0 = -0.5 * dx * Lx;
    y0 = -0.5 * dy * Ly;
    z0 = -0.5 * dz * Lz;
  }

  World world = world::create(GridSize({Lx, Ly, Lz}), PhysicalOrigin({x0, y0, z0}),
                              GridSpacing({dx, dy, dz}));

  return world;
}

template <> [[nodiscard]] inline Time from_json<Time>(const json &j) {
  // Support both flat and nested structures
  auto t0_val = get_json_value(j, "t0", "timestepping");
  auto t1_val = get_json_value(j, "t1", "timestepping");
  auto dt_val = get_json_value(j, "dt", "timestepping");
  auto saveat_val = get_json_value(j, "saveat", "timestepping");

  if (t0_val.is_null() || t1_val.is_null() || dt_val.is_null() ||
      saveat_val.is_null()) {
    throw std::invalid_argument(
        "Missing required time stepping parameters (t0, t1, dt, saveat)");
  }

  double t0 = t0_val;
  double t1 = t1_val;
  double dt = dt_val;
  double saveat = saveat_val;
  Time time({t0, t1, dt}, saveat);
  return time;
}

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_WORLD_TIME_HPP
