// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/profile_scope_macro.hpp>
#include <openpfc/kernel/profiling/session.hpp>

using pfc::profiling::ProfilingContextScope;
using pfc::profiling::ProfilingManualScope;
using pfc::profiling::ProfilingMetricCatalog;
using pfc::profiling::ProfilingSession;
using pfc::profiling::ProfilingTimedScope;

TEST_CASE("ensure_path empty catalog then timed scope", "[profiling]") {
  ProfilingSession s(ProfilingMetricCatalog{});
  ProfilingContextScope ctx(&s);
  s.begin_frame();
  {
    ProfilingTimedScope z("nest/inner");
    (void)z;
  }
  s.end_frame();
  REQUIRE(s.catalog().size() >= 2);
  REQUIRE(s.num_frames() == 1);
}

TEST_CASE("OPENPFC_PROFILE macro registers path", "[profiling]") {
  ProfilingSession s(ProfilingMetricCatalog{});
  ProfilingContextScope ctx(&s);
  s.begin_frame();
  OPENPFC_PROFILE("macro_block") { REQUIRE(true); }
  s.end_frame();
  std::size_t ix = 0;
  REQUIRE(s.catalog().try_index("macro_block", ix));
}

TEST_CASE("ProfilingManualScope stop and restart", "[profiling]") {
  ProfilingSession s(ProfilingMetricCatalog{});
  ProfilingContextScope ctx(&s);
  s.begin_frame();
  ProfilingManualScope timer;
  timer.start("phase_a");
  timer.stop();
  timer.restart("phase_b");
  // destructor pops phase_b
  s.end_frame();
  REQUIRE(s.num_frames() == 1);
  std::size_t ia = 0, ib = 0;
  REQUIRE(s.catalog().try_index("phase_a", ia));
  REQUIRE(s.catalog().try_index("phase_b", ib));
}

TEST_CASE("ProfilingManualScope move assigns sequential regions", "[profiling]") {
  ProfilingSession s(ProfilingMetricCatalog{});
  ProfilingContextScope ctx(&s);
  s.begin_frame();
  ProfilingManualScope t("first");
  t.stop();
  t = ProfilingManualScope("second");
  s.end_frame();
  std::size_t i1 = 0, i2 = 0;
  REQUIRE(s.catalog().try_index("first", i1));
  REQUIRE(s.catalog().try_index("second", i2));
  REQUIRE(s.num_frames() == 1);
}
