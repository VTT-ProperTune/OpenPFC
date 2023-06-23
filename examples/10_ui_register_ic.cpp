#include <memory>
#include <openpfc/openpfc.hpp>
#include <openpfc/ui.hpp>

using namespace pfc;
using namespace pfc::ui;

/*

Remember: openpfc/ui.hpp uses nlohmann_json, thus when linking target, one must
link that also!


    add_executable(10_ui_register_ic 10_ui_register_ic.cpp)
    target_link_libraries(10_ui_register_ic PRIVATE OpenPFC nlohmann_json::nlohmann_json)


Example json file looks like:

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
*/

// Define custom initial condition
class MyIC : public FieldModifier {
 private:
  double m_value = 1.0;

 public:
  MyIC(double value) : m_value(value) {
    std::cout << "Constructor: value = " << value << std::endl;
  }

  void apply(Model &, double) override {
    std::cout << "Applying MyIC with value " << m_value << std::endl;
  }
};

// Parse initial condition from json file
namespace pfc {
namespace ui {
template <>
std::unique_ptr<MyIC> from_json<std::unique_ptr<MyIC>>(const json &params) {
  std::cout << "Parsing MyIC from json" << std::endl;
  if (!params.contains("value") || !params["value"].is_number()) {
    throw std::invalid_argument(
        "Reading MyIC failed: missing or invalid 'value' field.");
  }
  double value = params["value"];
  return std::make_unique<MyIC>(value);
}
}  // namespace ui
}  // namespace pfc

// Define model
class MyModel : public Model {
  using Model::Model;

 public:
  void initialize(double) override {
    std::cout << "MyModel.initialize()" << std::endl;
  }

  void step(double) override { std::cout << "MyModel.step()" << std::endl; }
};

// Parse model settings from json file
void from_json(const json &, MyModel &) {
  std::cout << "MyModel: reading settings from json file" << std::endl;
}

int main(int argc, char *argv[]) {
  // Register initial condition 'MyIC' to 'my_initial_condition'
  register_field_modifier<MyIC>("my_initial_condition");
  App<MyModel> app(argc, argv);
  return app.main();
}
