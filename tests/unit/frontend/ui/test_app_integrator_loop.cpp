// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <string>

#include <catch2/catch_test_macros.hpp>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_simulation_session.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using json = nlohmann::json;

namespace {

json minimal_spectral_settings() {
  json settings;
  settings["method"] = "spectral";
  settings["backend"] = "cpu";
  settings["Lx"] = 4;
  settings["Ly"] = 4;
  settings["Lz"] = 4;
  settings["dx"] = 0.25;
  settings["dy"] = 0.25;
  settings["dz"] = 0.25;
  settings["origin"] = "center";
  settings["t0"] = 0.0;
  settings["t1"] = 0.05;
  settings["dt"] = 0.01;
  settings["saveat"] = 0.05;
  settings["decomposition"]["fft"] = "cpu";
  settings["decomposition"]["pencil"] = std::vector<int>{1, 1, 1};
  settings["plan_options"]["heffte_options"] = json{};
  return settings;
}

} // namespace

TEST_CASE("JSON spectral session drives SimulationDriver without Model",
          "[ui][unit][session]") {
  int rank_id = 0;
  int num_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  auto session = pfc::ui::make_simulation_session<pfc::sim::stacks::SpectralCPUStack>(
      minimal_spectral_settings(), rank_id, num_ranks, MPI_COMM_WORLD);
  REQUIRE(std::string(session.stack_name()) == "SpectralCPUStack");

  int steps = 0;
  pfc::sim::run(session.time(), [&](double) { ++steps; });
  REQUIRE(steps > 0);
  REQUIRE(pfc::time::done(session.time()));
}
