// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/metric_catalog.hpp>
#include <openpfc/kernel/profiling/names.hpp>
#include <openpfc/kernel/profiling/region_scope.hpp>
#include <openpfc/kernel/profiling/session.hpp>
#include <openpfc/kernel/profiling/timer_report.hpp>
#include <sstream>
#include <string>

using pfc::profiling::ProfilingContextScope;
using pfc::profiling::ProfilingMetricCatalog;
using pfc::profiling::ProfilingPrintOptions;
using pfc::profiling::ProfilingSession;
using pfc::profiling::print_profiling_timer;

TEST_CASE("print_profiling_timer aggregates two frames", "[profiling]") {
  ProfilingSession s(ProfilingMetricCatalog::with_defaults_and_extras({}));

  s.begin_step_frame(0, 0);
  {
    ProfilingContextScope scope(&s);
    pfc::profiling::record_time(pfc::profiling::kProfilingRegionGradient, 0.05);
  }
  s.end_step_frame(1.0, 0.1, 0u, 0u, 0u);

  s.begin_step_frame(1, 0);
  {
    ProfilingContextScope scope(&s);
    pfc::profiling::record_time(pfc::profiling::kProfilingRegionCommunication, 0.02);
  }
  s.end_step_frame(2.0, 0.3, 0u, 0u, 0u);

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

TEST_CASE("print_profiling_timer uses current_session when in scope", "[profiling]") {
  ProfilingSession s(ProfilingMetricCatalog::with_defaults_and_extras({}));
  ProfilingContextScope scope(&s);
  s.begin_step_frame(0, 0);
  s.end_step_frame(0.5, 0.2, 0u, 0u, 0u);

  std::ostringstream oss;
  print_profiling_timer(oss, ProfilingPrintOptions{});
  const std::string out = oss.str();

  REQUIRE(out.find("OpenPFC profiling") != std::string::npos);
  REQUIRE(out.find("fft") != std::string::npos);
}
