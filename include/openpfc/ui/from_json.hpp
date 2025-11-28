// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/from_json.hpp
 * @brief JSON deserialization functions for OpenPFC types
 *
 * @details
 * This header provides template specializations and overloads for converting
 * JSON objects into OpenPFC types. It handles:
 * - World configuration (domain size, spacing, origin)
 * - Time stepping parameters
 * - Initial conditions (constant, seeds, file input)
 * - Boundary conditions (fixed, moving)
 * - FFT backend options
 * - Model parameters
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UI_FROM_JSON_HPP
#define PFC_UI_FROM_JSON_HPP

#include "errors.hpp"
#include "json_helpers.hpp"
#include "openpfc/boundary_conditions/fixed_bc.hpp"
#include "openpfc/boundary_conditions/moving_bc.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/field_modifier.hpp"
#include "openpfc/initial_conditions/constant.hpp"
#include "openpfc/initial_conditions/file_reader.hpp"
#include "openpfc/initial_conditions/random_seeds.hpp"
#include "openpfc/initial_conditions/seed_grid.hpp"
#include "openpfc/initial_conditions/single_seed.hpp"
#include "openpfc/model.hpp"
#include "openpfc/time.hpp"
#include <heffte.h>
#include <stdexcept>

namespace pfc {
namespace ui {

template <class T> T from_json(const json &settings);

/**
 * @brief Converts a JSON object to heffte::plan_options.
 *
 * This function parses the provided JSON object and constructs a
 * heffte::plan_options object based on the values found in the JSON. The
 * function prints debug information to the console regarding the options being
 * parsed.
 *
 * @param j The JSON object to parse.
 * @return The heffte::plan_options object constructed from the JSON.
 */
template <> heffte::plan_options from_json<heffte::plan_options>(const json &j) {
  std::cout << "\nParsing backend options ...\n";
  heffte::plan_options options = heffte::default_options<heffte::backend::fftw>();
  if (j.contains("use_reorder")) {
    std::cout << "Using strided 1d fft operations" << std::endl;
    options.use_reorder = j["use_reorder"];
  }
  if (j.contains("reshape_algorithm")) {
    if (j["reshape_algorithm"] == "alltoall") {
      std::cout << "Using alltoall reshape algorithm" << std::endl;
      options.algorithm = heffte::reshape_algorithm::alltoall;
    } else if (j["reshape_algorithm"] == "alltoallv") {
      std::cout << "Using alltoallv reshape algorithm" << std::endl;
      options.algorithm = heffte::reshape_algorithm::alltoallv;
    } else if (j["reshape_algorithm"] == "p2p") {
      std::cout << "Using p2p reshape algorithm" << std::endl;
      options.algorithm = heffte::reshape_algorithm::p2p;
    } else if (j["reshape_algorithm"] == "p2p_plined") {
      std::cout << "Using p2p_plined reshape algorithm" << std::endl;
      options.algorithm = heffte::reshape_algorithm::p2p_plined;
    } else {
      std::cerr << "Unknown reshape algorithm " << j["reshape_algorithm"]
                << std::endl;
    }
  }
  if (j.contains("use_pencils")) {
    std::cout << "Using pencil decomposition" << std::endl;
    options.use_pencils = j["use_pencils"];
  }
  if (j.contains("use_gpu_aware")) {
    std::cout << "Using gpu aware fft" << std::endl;
    options.use_gpu_aware = j["use_gpu_aware"];
  }
  std::cout << "Backend options: " << options << "\n\n";
  return options;
}

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
template <> World from_json<World>(const json &j) {
  int Lx = 0, Ly = 0, Lz = 0;
  double dx = 0.0, dy = 0.0, dz = 0.0;
  double x0 = 0.0, y0 = 0.0, z0 = 0.0;
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
        origin_str, {"center", "corner"}, "\"origin\": \"center\""));
  }
  origin = origin_val;

  std::string origin_key = j.contains("origin") ? "origin" : "origo";
  if (origin != "center" && origin != "corner") {
    throw std::invalid_argument(format_config_error(
        origin_key, "coordinate system origin", "string ('center' or 'corner')",
        "\"" + origin + "\"", {"center", "corner"}, "\"origin\": \"center\""));
  }

  if (origin == "center") {
    x0 = -0.5 * dx * Lx;
    y0 = -0.5 * dy * Ly;
    z0 = -0.5 * dz * Lz;
  }

  World world = world::create({Lx, Ly, Lz}, {x0, y0, z0}, {dx, dy, dz});

  return world;
}

template <> Time from_json<Time>(const json &settings) {
  // Support both flat and nested structures
  auto t0_val = get_json_value(settings, "t0", "timestepping");
  auto t1_val = get_json_value(settings, "t1", "timestepping");
  auto dt_val = get_json_value(settings, "dt", "timestepping");
  auto saveat_val = get_json_value(settings, "saveat", "timestepping");

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

void from_json(const json &j, Constant &ic) {
  // Check that the JSON input has the correct type field
  if (!j.contains("type") || j["type"] != "constant") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }
  // Check that the JSON input has the required 'n0' field
  if (!j.contains("n0") || !j["n0"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'n0' field.");
  }
  ic.set_density(j["n0"]);
}

void from_json(const json &j, SingleSeed &seed) {
  if (!j.count("type") || j["type"] != "single_seed") {
    throw std::invalid_argument(
        "JSON object does not contain a 'single_seed' type.");
  }

  if (!j.count("amp_eq")) {
    throw std::invalid_argument("JSON object does not contain an 'amp_eq' key.");
  }

  if (!j.count("rho_seed")) {
    throw std::invalid_argument("JSON object does not contain a 'rho_seed' key.");
  }

  seed.set_amplitude(j["amp_eq"]);
  seed.set_density(j["rho_seed"]);
}

void from_json(const json &j, RandomSeeds &ic) {
  // Check that the JSON input has the correct type field
  if (!j.contains("type") || j["type"] != "random_seeds") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }

  // Check that the JSON input has the required 'amplitude' field
  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'amplitude' field.");
  }

  // Check that the JSON input has the required 'rho' field
  if (!j.contains("rho") || !j["rho"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho' field.");
  }

  ic.set_amplitude(j["amplitude"]);
  ic.set_density(j["rho"]);
}

void from_json(const json &j, SeedGrid &ic) {
  if (!j.contains("type") || j["type"] != "seed_grid") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("Ny") || !j["Ny"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'Ny' field.");
  }

  if (!j.contains("Nz") || !j["Nz"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'Nz' field.");
  }

  if (!j.contains("X0") || !j["X0"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'X0' field.");
  }

  if (!j.contains("radius") || !j["radius"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'radius' field.");
  }

  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'amplitude' field.");
  }

  if (!j.contains("rho") || !j["rho"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho' field.");
  }

  ic.set_Ny(j["Ny"]);
  ic.set_Nz(j["Nz"]);
  ic.set_X0(j["X0"]);
  ic.set_radius(j["radius"]);
  ic.set_amplitude(j["amplitude"]);
  ic.set_density(j["rho"]);
}

void from_json(const json &j, FileReader &ic) {
  if (!j.contains("type") || j["type"] != "from_file") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("filename") || !j["filename"].is_string()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'filename' field.");
  }

  ic.set_filename(j["filename"]);
}

void from_json(const json &j, FixedBC &bc) {
  if (!j.contains("type") || j["type"] != "fixed") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("rho_low") || !j["rho_low"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_low' field.");
  }

  if (!j.contains("rho_high") || !j["rho_high"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_high' field.");
  }

  bc.set_rho_low(j["rho_low"]);
  bc.set_rho_high(j["rho_high"]);
}

void from_json(const json &j, MovingBC &bc) {
  if (!j.contains("type") || j["type"] != "moving") {
    throw std::invalid_argument(
        "Invalid JSON input: missing or incorrect 'type' field.");
  }

  if (!j.contains("rho_low") || !j["rho_low"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_low' field.");
  }

  if (!j.contains("rho_high") || !j["rho_high"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_high' field.");
  }

  if (!j.contains("width") || !j["width"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'width' field.");
  }

  if (!j.contains("alpha") || !j["alpha"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'alpha' field.");
  }

  if (!j.contains("disp") || !j["disp"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'disp' field.");
  }

  if (!j.contains("xpos") || !j["xpos"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'xpos' field.");
  }

  bc.set_rho_low(j["rho_low"]);
  bc.set_rho_high(j["rho_high"]);
  bc.set_xwidth(j["width"]);
  bc.set_alpha(j["alpha"]);
  bc.set_disp(j["disp"]);
  bc.set_xpos(j["xpos"]);
}

void from_json(const json &, Model &) {
  std::cout << "Warning: This model does not implement reading parameters from "
               "json file. In order to read parameters from json file, one needs to "
               "implement 'void from_json(const json &, Model &)'"
            << std::endl;
}

} // namespace ui
} // namespace pfc

#endif // PFC_UI_FROM_JSON_HPP
