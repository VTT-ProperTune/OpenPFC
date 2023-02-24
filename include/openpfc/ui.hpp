#pragma once

#include "field_modifier.hpp"
#include "initial_conditions/constant.hpp"
#include "initial_conditions/random_seeds.hpp"
#include "initial_conditions/single_seed.hpp"
#include "time.hpp"
#include "world.hpp"

#include <iostream>
#include <nlohmann/json.hpp>

namespace pfc {
namespace ui {

/*
Functions and classes to construct objects from json file, and other
"ui"-related things.
*/

using json = nlohmann::json;

template <class T> T from_json(const json &settings);

template <>
heffte::plan_options from_json<heffte::plan_options>(const json &j) {
  heffte::plan_options options =
      heffte::default_options<heffte::backend::fftw>();
  if (j.contains("use_reorder")) {
    options.use_reorder = j["use_reorder"];
  }
  if (j.contains("reshape_algorithm")) {
    if (j["reshape_algorithm"] == "alltoall") {
      options.algorithm = heffte::reshape_algorithm::alltoall;
    } else if (j["reshape_algorithm"] == "alltoallv") {
      options.algorithm = heffte::reshape_algorithm::alltoallv;
    } else if (j["reshape_algorithm"] == "p2p") {
      options.algorithm = heffte::reshape_algorithm::p2p;
    } else if (j["reshape_algorithm"] == "p2p_plined") {
      options.algorithm = heffte::reshape_algorithm::p2p_plined;
    } else {
      std::cerr << "Unknown communcation model " << j["reshape_algorithm"]
                << std::endl;
    }
  }
  if (j.contains("use_pencils")) {
    options.use_pencils = j["use_pencils"];
  }
  if (j.contains("use_gpu_aware")) {
    options.use_gpu_aware = j["use_gpu_aware"];
  }
  std::cout << "backend options: " << options << "\n\n";
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
 *   - origo (string): The origin of the coordinate system. Must be one of
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
  std::string origo;

  if (!j.count("Lx") || !j["Lx"].is_number_integer()) {
    throw std::invalid_argument("Missing or invalid 'Lx' field in JSON input.");
  }
  Lx = j["Lx"];

  if (!j.count("Ly") || !j["Ly"].is_number_integer()) {
    throw std::invalid_argument("Missing or invalid 'Ly' field in JSON input.");
  }
  Ly = j["Ly"];

  if (!j.count("Lz") || !j["Lz"].is_number_integer()) {
    throw std::invalid_argument("Missing or invalid 'Lz' field in JSON input.");
  }
  Lz = j["Lz"];

  if (!j.count("dx") || !j["dx"].is_number_float()) {
    throw std::invalid_argument("Missing or invalid 'dx' field in JSON input.");
  }
  dx = j["dx"];

  if (!j.count("dy") || !j["dy"].is_number_float()) {
    throw std::invalid_argument("Missing or invalid 'dy' field in JSON input.");
  }
  dy = j["dy"];

  if (!j.count("dz") || !j["dz"].is_number_float()) {
    throw std::invalid_argument("Missing or invalid 'dz' field in JSON input.");
  }
  dz = j["dz"];

  if (!j.count("origo") || !j["origo"].is_string()) {
    throw std::invalid_argument(
        "Missing or invalid 'origo' field in JSON input.");
  }
  origo = j["origo"];

  if (origo != "center" && origo != "corner") {
    throw std::invalid_argument("Invalid 'origo' field in JSON input.");
  }

  if (origo == "center") {
    x0 = -0.5 * dx * Lx;
    y0 = -0.5 * dy * Ly;
    z0 = -0.5 * dz * Lz;
  }

  World world({Lx, Ly, Lz}, {x0, y0, z0}, {dx, dy, dz});

  return world;
}

template <> Time from_json<Time>(const json &settings) {
  double t0 = settings["t0"];
  double t1 = settings["t1"];
  double dt = settings["dt"];
  double saveat = settings["saveat"];
  Time time({t0, t1, dt}, saveat);
  return time;
}

using Constant_p = std::unique_ptr<Constant>;

/**
 * A factory function that creates a `std::unique_ptr` to a `Constant` object
 * from a JSON input.
 *
 * The JSON input must have the following format:
 *
 * ```
 * {
 *   "type": "constant",
 *   "n0": <double>
 * }
 * ```
 *
 * where `<double>` is a floating-point number that represents the constant
 * value to set the field to.
 *
 * If the JSON input is invalid or missing required fields, an
 * `std::invalid_argument` exception is thrown.
 *
 * Example usage:
 *
 * ```
 * // Create a constant field modifier with value 1.0 from JSON input
 * json input = {
 *   {"type", "constant"},
 *   {"n0", 1.0}
 * };
 * Constant_p c = from_json<Constant_p>(input);
 * ```
 */
template <> Constant_p from_json<Constant_p>(const json &j) {

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

  double n0 = j["n0"];
  return std::make_unique<Constant>(n0);
}

using SingleSeed_p = std::unique_ptr<SingleSeed>;

/**
 * Convert a JSON object to a unique_ptr to a SingleSeed object.
 *
 * This function converts a JSON object to a unique_ptr to a SingleSeed object.
 * The JSON object must contain the following fields:
 *   - type: A string with the value "single_seed".
 *   - amp_eq: A double with the amplitude of the equalizer.
 *   - rho_seed: A double with the seed value for the Rho function.
 *
 * If any of these fields are missing or have an invalid type, this function
 * throws a std::invalid_argument exception.
 *
 * @tparam T The type of the deleter object to use for the unique_ptr.
 * @param j The JSON object to convert.
 * @return A unique_ptr to a SingleSeed object.
 * @throws std::invalid_argument if any required fields are missing or have an
 * invalid type.
 */
template <> SingleSeed_p from_json<SingleSeed_p>(const json &j) {

  if (!j.count("type") || j["type"] != "single_seed") {
    throw std::invalid_argument(
        "JSON object does not contain a 'single_seed' type.");
  }

  if (!j.count("amp_eq")) {
    throw std::invalid_argument(
        "JSON object does not contain an 'amp_eq' key.");
  }

  if (!j.count("rho_seed")) {
    throw std::invalid_argument(
        "JSON object does not contain a 'rho_seed' key.");
  }

  std::unique_ptr<SingleSeed> seed = std::make_unique<SingleSeed>();
  seed->amp_eq = j["amp_eq"];
  seed->rho_seed = j["rho_seed"];
  return seed;
}

using RandomSeeds_p = std::unique_ptr<RandomSeeds>;

template <> RandomSeeds_p from_json<RandomSeeds_p>(const json &j) {

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

  RandomSeeds_p ic = std::make_unique<RandomSeeds>();
  ic->amplitude = j["amplitude"];
  ic->rho = j["rho"];
  return ic;
}

using FieldModifier_p = std::unique_ptr<FieldModifier>;

template <> FieldModifier_p from_json<FieldModifier_p>(const json &j) {
  std::cout << "Creating FieldModifier from data " << j << std::endl;
  std::string type = j["type"];
  if (type == "single_seed") {
    std::cout << "Creating SingleSeed <: FieldModifier" << std::endl;
    return from_json<SingleSeed_p>(j);
  }
  if (type == "constant") {
    std::cout << "Creating Constant <: FieldModifier" << std::endl;
    return from_json<Constant_p>(j);
  }
  if (type == "random_seeds") {
    std::cout << "Creating RandomSeeds <: FieldModifier" << std::endl;
    return from_json<RandomSeeds_p>(j);
  }
  throw std::invalid_argument("Unknown FieldModifier type: " + type);
}

} // namespace ui
} // namespace pfc
