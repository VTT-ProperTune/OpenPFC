// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/metric_catalog.hpp>
#include <openpfc/kernel/profiling/names.hpp>
#include <openpfc/kernel/profiling/openpfc_frame_metrics.hpp>
#include <openpfc/kernel/profiling/region_scope.hpp>
#include <openpfc/kernel/profiling/session.hpp>
#include <openpfc/kernel/profiling/timer_report.hpp>
#include <sstream>
#include <string>

using pfc::profiling::openpfc_begin_frame_with_step_and_rank;
using pfc::profiling::openpfc_end_frame_with_fft_region_wall_and_memory;
using pfc::profiling::print_profiling_timer;
using pfc::profiling::ProfilingContextScope;
using pfc::profiling::ProfilingMetricCatalog;
using pfc::profiling::ProfilingPrintOptions;
using pfc::profiling::ProfilingSession;

TEST_CASE("print_profiling_timer aggregates two frames", "[profiling]") {
  ProfilingSession s(ProfilingMetricCatalog::with_defaults_and_extras({}),
                     ProfilingSession::openpfc_default_frame_metrics());

  openpfc_begin_frame_with_step_and_rank(s, 0, 0);
  {
    ProfilingContextScope scope(&s);
    pfc::profiling::record_time(pfc::profiling::kProfilingRegionGradient, 0.05);
  }
  openpfc_end_frame_with_fft_region_wall_and_memory(s, 1.0, 0.1, 0u, 0u, 0u);

  openpfc_begin_frame_with_step_and_rank(s, 1, 0);
  {
    ProfilingContextScope scope(&s);
    pfc::profiling::record_time(pfc::profiling::kProfilingRegionCommunication, 0.02);
  }
  openpfc_end_frame_with_fft_region_wall_and_memory(s, 2.0, 0.3, 0u, 0u, 0u);

  std::ostringstream oss;
  ProfilingPrintOptions opts;
  opts.title = "Test report";
  opts.ascii_lines = true;
  print_profiling_timer(oss, s, opts);
  const std::string out = oss.str();

  REQUIRE(out.find("Test report") != std::string::npos);
  REQUIRE(out.find("Section") != std::string::npos);
  REQUIRE(out.find("ncalls") != std::string::npos);
  REQUIRE(out.find("%tot") != std::string::npos);
  REQUIRE(out.find("fft") != std::string::npos);
  REQUIRE(out.find("communication") != std::string::npos);
  REQUIRE(out.find("gradient") != std::string::npos);
  REQUIRE(out.find("Tot wall (sum steps):") != std::string::npos);
  REQUIRE(out.find("400.000 ms") != std::string::npos);
}

TEST_CASE("print_profiling_timer uses current_session when in scope",
          "[profiling]") {
  ProfilingSession s(ProfilingMetricCatalog::with_defaults_and_extras({}),
                     ProfilingSession::openpfc_default_frame_metrics());
  ProfilingContextScope scope(&s);
  openpfc_begin_frame_with_step_and_rank(s, 0, 0);
  openpfc_end_frame_with_fft_region_wall_and_memory(s, 0.5, 0.2, 0u, 0u, 0u);

  std::ostringstream oss;
  print_profiling_timer(oss, ProfilingPrintOptions{});
  const std::string out = oss.str();

  REQUIRE(out.find("OpenPFC profiling") != std::string::npos);
  REQUIRE(out.find("fft") != std::string::npos);
}

TEST_CASE("print_profiling_timer MPI aggregate mean across ranks",
          "[profiling][MPI]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) return;

  ProfilingSession s(ProfilingMetricCatalog::with_defaults_and_extras({}),
                     ProfilingSession::openpfc_default_frame_metrics());

  if (rank == 0) {
    openpfc_begin_frame_with_step_and_rank(s, 0, 0);
    {
      ProfilingContextScope scope(&s);
      pfc::profiling::record_time(pfc::profiling::kProfilingRegionGradient, 0.10);
    }
    openpfc_end_frame_with_fft_region_wall_and_memory(s, 1.0, 0.1, 0u, 0u, 0u);
  } else {
    openpfc_begin_frame_with_step_and_rank(s, 0, 1);
    {
      ProfilingContextScope scope(&s);
      pfc::profiling::record_time(pfc::profiling::kProfilingRegionGradient, 0.20);
    }
    openpfc_end_frame_with_fft_region_wall_and_memory(s, 1.0, 0.1, 0u, 0u, 0u);
  }

  std::ostringstream oss;
  ProfilingPrintOptions opts;
  opts.title = "MPI agg test";
  opts.ascii_lines = true;
  opts.mpi_aggregate_stdout = true;
  opts.mpi_aggregate_stat = "mean";
  print_profiling_timer(oss, MPI_COMM_WORLD, s, opts);

  if (rank != 0) return;
  const std::string out = oss.str();
  REQUIRE(out.find("MPI agg test") != std::string::npos);
  REQUIRE(out.find("gradient") != std::string::npos);
  // mean(0.10, 0.20) = 0.15 s -> 150.000 ms in time column
  REQUIRE(out.find("150.000 ms") != std::string::npos);
}
