// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <memory>
#include <openpfc/openpfc.hpp>
#include <openpfc/ui.hpp>

using namespace pfc;
using namespace pfc::ui;

/*
Remember: openpfc/ui.hpp uses nlohmann_json, thus when linking target, one must
link that also!

  add_executable(10_ui_register_ic 10_ui_register_ic.cpp)
  target_link_libraries(10_ui_register_ic PRIVATE OpenPFC
nlohmann_json::nlohmann_json)
*/

// Define custom initial condition
class MyIC : public FieldModifier {
private:
  double m_value = 1.0;

public:
  void set_value(double value) { m_value = value; }
  double get_value() const { return m_value; }

  void apply(Model &, double) override {
    std::cout << "Applying MyIC with value " << get_value() << std::endl;
  }
};

// Parse initial condition data from json file
void from_json(const json &params, MyIC &ic) {
  std::cout << "Parsing MyIC from json" << std::endl;
  if (!params.contains("value") || !params["value"].is_number()) {
    throw std::invalid_argument(
        "Reading MyIC failed: missing or invalid 'value' field.");
  }
  ic.set_value(params["value"]);
}

// Define model
class MyModel : public Model {
public:
  /**
   * @brief Constructs a MyModel instance with the given World object.
   *
   * @param world The World object to initialize the model.
   */
  explicit MyModel(const World &world) : Model(world) {
    // Additional initialization if needed
  }

  void initialize(double) override { std::cout << "initialize()" << std::endl; }
  void step(double) override { std::cout << "MyModel.step()" << std::endl; }
};

// Parse model settings from json file
void from_json(const json &, MyModel &) {
  std::cout << "MyModel: reading settings from json file" << std::endl;
}

int main() {
  // Register initial condition 'MyIC' to 'my_initial_condition'
  register_field_modifier<MyIC>("my_initial_condition");

  json settings = R"(
  {
      "model": {
          "name": "mymodel",
          "params": {
              "n0": -0.10
          }
      },
      "Lx": 64,
      "Ly": 64,
      "Lz": 64,
      "dx": 1.1107207345395915,
      "dy": 1.1107207345395915,
      "dz": 1.1107207345395915,
      "origo": "corner",
      "t0": 0.0,
      "t1": 10.0,
      "dt": 1.0,
      "saveat": -1.0,
      "results": "data/u_%04d.bin",
      "fields": [],
      "initial_conditions": [
          {
              "type": "my_initial_condition",
              "value": 42.0
          }
      ],
      "boundary_conditions": []
  }
  )"_json;

  App<MyModel> app(settings);
  return app.main();
}
