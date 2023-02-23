#pragma once

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

} // namespace ui
} // namespace pfc
