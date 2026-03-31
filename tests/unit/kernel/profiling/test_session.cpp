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
#include <openpfc/kernel/profiling/openpfc_frame_metrics.hpp>
#include <openpfc/kernel/profiling/region_scope.hpp>
#include <openpfc/kernel/profiling/session.hpp>
#include <string>
#include <thread>

using pfc::profiling::openpfc_begin_frame_with_step_and_rank;
using pfc::profiling::openpfc_end_frame_with_fft_region_wall_and_memory;
using pfc::profiling::ProfilingContextScope;
using pfc::profiling::ProfilingMetricCatalog;
using pfc::profiling::ProfilingSession;
using pfc::profiling::ProfilingTimedScope;

TEST_CASE("ProfilingSession single rank JSON export", "[profiling]") {
  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if (mpi_size != 1) return;

  ProfilingSession s(ProfilingMetricCatalog::with_defaults_and_extras({}),
                     ProfilingSession::openpfc_default_frame_metrics());
  openpfc_begin_frame_with_step_and_rank(s, 7, 0);
  {
    ProfilingContextScope scope(&s);
    pfc::profiling::record_time(pfc::profiling::kProfilingRegionGradient, 0.002);
  }
  openpfc_end_frame_with_fft_region_wall_and_memory(s, 0.42, 0.15, 1000u, 2000u,
                                                    3000u);

  const auto tmp =
      std::filesystem::temp_directory_path() / "openpfc_profiling_test.json";
  const std::string path = tmp.string();

  pfc::profiling::ProfilingExportOptions opt;
  opt.write_json = true;
  opt.json_path = path;

  s.finalize_and_export(MPI_COMM_WORLD, opt);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) return;

  REQUIRE(std::filesystem::exists(tmp));
  std::ifstream in(path);
  nlohmann::json j;
  in >> j;
  REQUIRE(j["schema_version"] == 2);
  REQUIRE(j["total_frames"] == 1);
  REQUIRE(j["ranks"].is_array());
  REQUIRE(j["ranks"][0]["mpi_rank"] == 0);
  REQUIRE(j["ranks"][0]["n_frames"] == 1);
  REQUIRE(j["frame_metric_names"].size() ==
          ProfilingSession::openpfc_default_frame_metrics().size());
  REQUIRE(j["region_paths"].is_array());
  const auto &sc = j["ranks"][0]["frames"][0]["scalars"];
  REQUIRE(sc[2].get<double>() == Catch::Approx(0.42));
  REQUIRE(j["ranks"][0]["frames"][0]["regions"]["fft"]["inclusive"] ==
          Catch::Approx(0.15));
  REQUIRE(j["ranks"][0]["frames"][0]["regions"]["gradient"]["inclusive"] ==
          Catch::Approx(0.002));
  std::filesystem::remove(tmp);
}

TEST_CASE("ProfilingSession nested timed scopes exclusive", "[profiling]") {
  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if (mpi_size != 1) return;

  ProfilingMetricCatalog cat =
      ProfilingMetricCatalog::with_defaults_and_extras({"outer/inner"});
  ProfilingSession s(std::move(cat),
                     ProfilingSession::openpfc_default_frame_metrics());
  openpfc_begin_frame_with_step_and_rank(s, 1, 0);
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
  openpfc_end_frame_with_fft_region_wall_and_memory(s, 1.0, 0.0, 0u, 0u, 0u);

  const auto tmp =
      std::filesystem::temp_directory_path() / "openpfc_profiling_nested.json";
  pfc::profiling::ProfilingExportOptions opt;
  opt.write_json = true;
  opt.json_path = tmp.string();
  s.finalize_and_export(MPI_COMM_WORLD, opt);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) return;

  std::ifstream in(tmp.string());
  nlohmann::json j;
  in >> j;
  const auto inner_inc =
      j["ranks"][0]["frames"][0]["regions"]["outer"]["inner"]["inclusive"]
          .get<double>();
  const auto outer_inc =
      j["ranks"][0]["frames"][0]["regions"]["outer"]["inclusive"].get<double>();
  const auto outer_exc =
      j["ranks"][0]["frames"][0]["regions"]["outer"]["exclusive"].get<double>();
  REQUIRE(inner_inc > 0.005);
  REQUIRE(outer_inc >= inner_inc);
  REQUIRE(outer_exc >= 0.0);
  REQUIRE(outer_exc <= outer_inc);
  REQUIRE(outer_inc - outer_exc == Catch::Approx(inner_inc).margin(1e-3));
  std::filesystem::remove(tmp);
}

TEST_CASE("ProfilingSession custom frame metric names", "[profiling]") {
  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if (mpi_size != 1) return;

  ProfilingSession s(ProfilingMetricCatalog::with_defaults_and_extras({}),
                     {"latency_ms"});
  s.begin_frame();
  s.set_frame_metric("latency_ms", 12.5);
  {
    ProfilingContextScope scope(&s);
    pfc::profiling::record_time(pfc::profiling::kProfilingRegionGradient, 0.01);
  }
  s.end_frame();

  const auto tmp = std::filesystem::temp_directory_path() /
                   "openpfc_profiling_custom_metrics.json";
  pfc::profiling::ProfilingExportOptions opt;
  opt.write_json = true;
  opt.json_path = tmp.string();
  s.finalize_and_export(MPI_COMM_WORLD, opt);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) return;

  std::ifstream in(tmp.string());
  nlohmann::json j;
  in >> j;
  REQUIRE(j["frame_metric_names"].size() == 1);
  REQUIRE(j["frame_metric_names"][0] == "latency_ms");
  REQUIRE(j["ranks"][0]["frames"][0]["scalars"][0].get<double>() ==
          Catch::Approx(12.5));
  std::filesystem::remove(tmp);
}

TEST_CASE("ProfilingSession MPI gather two ranks", "[profiling][MPI]") {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) return;

  ProfilingSession s(ProfilingMetricCatalog::with_defaults_and_extras({}),
                     ProfilingSession::openpfc_default_frame_metrics());
  openpfc_begin_frame_with_step_and_rank(s, 100, rank);
  openpfc_end_frame_with_fft_region_wall_and_memory(
      s, 0.1 * static_cast<double>(rank + 1), 0.05, 0u, 0u, 0u);

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
    REQUIRE(j["total_frames"] == 2);
    REQUIRE(j["schema_version"] == 2);
    REQUIRE(j["ranks"].size() == static_cast<std::size_t>(size));
    REQUIRE(j["ranks"][0]["n_frames"] == 1);
    REQUIRE(j["ranks"][1]["n_frames"] == 1);
    REQUIRE(j["ranks"][0]["mpi_rank"] == 0);
    REQUIRE(j["ranks"][1]["mpi_rank"] == 1);
    std::filesystem::remove(tmp);
  }
}
