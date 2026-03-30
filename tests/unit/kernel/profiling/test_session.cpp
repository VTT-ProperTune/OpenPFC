// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/metric_catalog.hpp>
#include <openpfc/kernel/profiling/names.hpp>
#include <openpfc/kernel/profiling/region_scope.hpp>
#include <openpfc/kernel/profiling/session.hpp>
#include <sstream>
#include <string>
#include <thread>

using pfc::profiling::ProfilingContextScope;
using pfc::profiling::ProfilingMetricCatalog;
using pfc::profiling::ProfilingSession;
using pfc::profiling::ProfilingTimedScope;

TEST_CASE("ProfilingSession single rank JSON export v2", "[profiling]") {
  ProfilingSession s(ProfilingMetricCatalog::with_defaults_and_extras({}));
  s.begin_step_frame(7, 0);
  {
    ProfilingContextScope scope(&s);
    pfc::profiling::record_time(pfc::profiling::kProfilingRegionGradient, 0.002);
  }
  s.end_step_frame(0.42, 0.15, 1000u, 2000u, 3000u);

  const auto tmp =
      std::filesystem::temp_directory_path() / "openpfc_profiling_test.json";
  const std::string path = tmp.string();

  pfc::profiling::ProfilingExportOptions opt;
  opt.write_json = true;
  opt.json_path = path;

  s.finalize_and_export(MPI_COMM_WORLD, opt);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0)
    return;

  REQUIRE(std::filesystem::exists(tmp));
  std::ifstream in(path);
  nlohmann::json j;
  in >> j;
  REQUIRE(j["schema_version"] == 2);
  REQUIRE(j["n_frames"] == 1);
  REQUIRE(j["frames"][0]["wall_step"] == Catch::Approx(0.42));
  REQUIRE(j["frames"][0]["regions"]["fft"]["inclusive"] == Catch::Approx(0.15));
  REQUIRE(j["frames"][0]["regions"]["gradient"]["inclusive"] ==
          Catch::Approx(0.002));
  std::filesystem::remove(tmp);
}

TEST_CASE("ProfilingSession single rank CSV export v2", "[profiling]") {
  ProfilingSession s(ProfilingMetricCatalog::with_defaults_and_extras({}));
  s.begin_step_frame(3, 0);
  {
    ProfilingContextScope scope(&s);
    pfc::profiling::record_time(pfc::profiling::kProfilingRegionCommunication,
                                0.001);
  }
  s.end_step_frame(0.5, 0.2, 0u, 0u, 0u);

  const auto tmp =
      std::filesystem::temp_directory_path() / "openpfc_profiling_test.csv";
  const std::string path = tmp.string();

  pfc::profiling::ProfilingExportOptions opt;
  opt.write_json = false;
  opt.write_csv = true;
  opt.csv_path = path;

  s.finalize_and_export(MPI_COMM_WORLD, opt);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0)
    return;

  REQUIRE(std::filesystem::exists(tmp));
  std::ifstream in(path);
  std::string header;
  REQUIRE(std::getline(in, header));
  REQUIRE(header.find("communication_inclusive") != std::string::npos);
  std::string row;
  REQUIRE(std::getline(in, row));
  std::istringstream rs(row);
  std::string cell;
  std::vector<double> nums;
  while (std::getline(rs, cell, ',')) {
    nums.push_back(std::stod(cell));
  }
  const std::size_t K = 3;
  REQUIRE(nums.size() == static_cast<std::size_t>(6 + 2 * K));
  REQUIRE(nums[0] == Catch::Approx(3.0));
  REQUIRE(nums[2] == Catch::Approx(0.5));
  std::filesystem::remove(tmp);
}

TEST_CASE("ProfilingSession nested timed scopes exclusive", "[profiling]") {
  ProfilingMetricCatalog cat =
      ProfilingMetricCatalog::with_defaults_and_extras({"outer/inner"});
  ProfilingSession s(std::move(cat));
  s.begin_step_frame(1, 0);
  {
    ProfilingContextScope ctx(&s);
    ProfilingTimedScope o("outer");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    {
      ProfilingTimedScope i("outer/inner");
      std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  s.end_step_frame(1.0, 0.0, 0u, 0u, 0u);

  const auto tmp =
      std::filesystem::temp_directory_path() / "openpfc_profiling_nested.json";
  pfc::profiling::ProfilingExportOptions opt;
  opt.write_json = true;
  opt.json_path = tmp.string();
  s.finalize_and_export(MPI_COMM_WORLD, opt);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0)
    return;

  std::ifstream in(tmp.string());
  nlohmann::json j;
  in >> j;
  const double inner_inc =
      j["frames"][0]["regions"]["outer"]["inner"]["inclusive"].get<double>();
  const double outer_inc =
      j["frames"][0]["regions"]["outer"]["inclusive"].get<double>();
  const double outer_exc =
      j["frames"][0]["regions"]["outer"]["exclusive"].get<double>();
  REQUIRE(inner_inc > 0.005);
  REQUIRE(outer_inc >= inner_inc);
  REQUIRE(outer_exc >= 0.0);
  REQUIRE(outer_exc <= outer_inc);
  REQUIRE(outer_inc - outer_exc == Catch::Approx(inner_inc).margin(1e-3));
  std::filesystem::remove(tmp);
}

TEST_CASE("ProfilingSession MPI gather two ranks", "[profiling][MPI]") {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2)
    return;

  ProfilingSession s(ProfilingMetricCatalog::with_defaults_and_extras({}));
  s.begin_step_frame(100, rank);
  s.end_step_frame(0.1 * static_cast<double>(rank + 1), 0.05, 0u, 0u, 0u);

  const auto tmp =
      std::filesystem::temp_directory_path() / "openpfc_profiling_mpi2.json";
  pfc::profiling::ProfilingExportOptions opt;
  opt.write_json = true;
  opt.json_path = tmp.string();

  s.finalize_and_export(MPI_COMM_WORLD, opt);

  if (rank == 0) {
    std::ifstream in(tmp.string());
    nlohmann::json j;
    in >> j;
    REQUIRE(j["n_frames"] == 2);
    REQUIRE(j["schema_version"] == 2);
    std::filesystem::remove(tmp);
  }
}
