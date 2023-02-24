#pragma once

#include "field_modifier.hpp"
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

template <> World from_json<World>(const json &settings) {
  int Lx = settings["Lx"];
  int Ly = settings["Ly"];
  int Lz = settings["Lz"];
  double dx = settings["dx"];
  double dy = settings["dy"];
  double dz = settings["dz"];
  double x0 = 0.0;
  double y0 = 0.0;
  double z0 = 0.0;
  std::string origo = settings["origo"];
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

using FieldModifier_p = std::unique_ptr<FieldModifier>;

template <> FieldModifier_p from_json<FieldModifier_p>(const json &j) {
  std::cout << "Creating FieldModifier from data " << j << std::endl;
  std::string type = j["type"];
  if (type == "single_seed") {
    std::cout << "Creating SingleSeed <: FieldModifier" << std::endl;
    return from_json<SingleSeed_p>(j);
  }
  throw std::invalid_argument("Unknown FieldModifier type: " + type);
}

} // namespace ui
} // namespace pfc
