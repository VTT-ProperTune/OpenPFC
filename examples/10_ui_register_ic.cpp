// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using json = nlohmann::json;
using namespace pfc;

/*
Remember: this example uses nlohmann_json, thus when linking the target,
one must link that also:

  add_executable(10_ui_register_ic 10_ui_register_ic.cpp)
  target_link_libraries(10_ui_register_ic PRIVATE OpenPFC
nlohmann_json::nlohmann_json)
*/

using HostIc = std::function<void(data::Field<double> &, const json &)>;

int main(int argc, char **argv) {
  try {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    std::map<std::string, HostIc> catalog;
    catalog["my_initial_condition"] = [](data::Field<double> &field,
                                         const json &params) {
      if (!params.contains("value") || !params["value"].is_number()) {
        throw std::invalid_argument(
            "Reading MyIC failed: missing or invalid 'value' field.");
      }
      const double value = params["value"].get<double>();
      std::cout << "Parsing MyIC from json" << '\n';
      std::cout << "Applying MyIC with value " << value << '\n';
      for (auto &v : field.vec()) v = value;
    };

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
      "saveat": 10.0,
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

    auto domain = ui::from_json<Domain>(settings);
    auto time = ui::from_json<Time>(settings);
    sim::stacks::SpectralCPUStack stack(std::move(domain), rank, nproc);
    for (const auto &ic : settings["initial_conditions"]) {
      const auto type = ic.at("type").get<std::string>();
      catalog.at(type)(stack.u(), ic);
    }

    pfc::sim::run(time, [&](double) {
      if (rank == 0) std::cout << "step()" << '\n';
    });

    MPI_Finalize();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    MPI_Finalize();
    return 1;
  }
}
