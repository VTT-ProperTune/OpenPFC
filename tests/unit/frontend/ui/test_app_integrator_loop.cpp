// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <mpi.h>

#include <openpfc/frontend/ui/app_integrator_loop.hpp>
#include <openpfc/frontend/ui/spectral_simulation_session.hpp>
#include <openpfc/kernel/utils/logging.hpp>

#include <nlohmann/json.hpp>

#include <fixtures/mock_model.hpp>

using namespace pfc::ui;
using namespace pfc;
using namespace pfc::testing;

struct StreamRedirect {
  std::ostream &os;
  std::streambuf *old_buf;
  std::ostringstream captured;
  explicit StreamRedirect(std::ostream &target)
      : os(target), old_buf(target.rdbuf()) {
    os.rdbuf(captured.rdbuf());
  }
  ~StreamRedirect() { os.rdbuf(old_buf); }
};

TEST_CASE("test_app_integrator_loop_with_valid_logger", "[ui][unit]") {
  int rank_id, num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  bool rank0 = (rank_id == 0);

  // Create a real logger at Info level
  Logger test_logger(LogLevel::Info, rank_id);

  // Set up SimulatorIntegratorLoopEnv with valid app_log pointer
  SimulatorIntegratorLoopEnv env{};
  env.app_log = &test_logger;
  env.profiler = nullptr;
  env.comm = MPI_COMM_WORLD;
  env.rank_id = rank_id;
  env.rank0 = rank0;

  // Create minimal settings for SpectralSimulationSession
  nlohmann::json settings;
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
  settings["plan_options"]["heffte_options"] = nlohmann::json{};

  // Create a session with MockModel and actually call the function
  SpectralSimulationSession<MockModel> session(settings, MPI_COMM_WORLD, rank_id, num_ranks);

  // Capture std::clog to verify logging occurs
  StreamRedirect redirect(std::clog);

  // Call the function with a valid logger
  auto timings = run_simulator_time_integration_loop(session, env);

  std::string log_output = redirect.captured.str();

  // Verify the function executed without crash
  REQUIRE(timings.steps_completed > 0);

  // Verify logging occurred when app_log is non-null
  if (rank0) {
    REQUIRE_FALSE(log_output.empty());
    REQUIRE(log_output.find("Step") != std::string::npos);
    REQUIRE(log_output.find("[INFO]") != std::string::npos);
  }
}

TEST_CASE("test_app_integrator_loop_with_null_logger", "[ui][unit]") {
  int rank_id, num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  bool rank0 = (rank_id == 0);

  // Set up SimulatorIntegratorLoopEnv with null app_log (default)
  SimulatorIntegratorLoopEnv env{};
  env.profiler = nullptr;
  env.comm = MPI_COMM_WORLD;
  env.rank_id = rank_id;
  env.rank0 = rank0;

  // Verify that the default state has null app_log
  REQUIRE(env.app_log == nullptr);

  // Create minimal settings for SpectralSimulationSession
  nlohmann::json settings;
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
  settings["plan_options"]["heffte_options"] = nlohmann::json{};

  // Create a session with MockModel
  SpectralSimulationSession<MockModel> session(settings, MPI_COMM_WORLD, rank_id, num_ranks);

  // Capture std::clog to verify silent execution
  StreamRedirect redirect(std::clog);

  // Call the function with null app_log (should not crash)
  auto timings = run_simulator_time_integration_loop(session, env);

  std::string log_output = redirect.captured.str();

  // Verify the function executed without crash
  REQUIRE(timings.steps_completed > 0);

  // Verify silent execution when app_log is nullptr
  REQUIRE(log_output.empty());
}
