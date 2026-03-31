// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file profiling_timer_report.cpp
 * @brief Demo: auto-registered paths, OPENPFC_PROFILE macro, measured step wall
 *
 * No up-front catalog list: paths are created on first use (TimerOutputs-style).
 * OPENPFC_PROFILE("path") { … } expands to a uniquely named ProfilingTimedScope.
 * ProfilingManualScope supports stop()/restart(path) without nested braces.
 *
 * Run: ./profiling_timer_report
 */

#include <chrono>
#include <iostream>
#include <openpfc/kernel/profiling/profiling.hpp>

namespace {
using namespace std::chrono_literals;

void spin(std::chrono::milliseconds d) {
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - t0 < d) {
  }
}
} // namespace

int main() {
  using pfc::profiling::print_profiling_timer;
  using pfc::profiling::ProfilingContextScope;
  using pfc::profiling::ProfilingManualScope;
  using pfc::profiling::ProfilingMetricCatalog;
  using pfc::profiling::ProfilingPrintOptions;
  using pfc::profiling::ProfilingSession;

  ProfilingSession session(ProfilingMetricCatalog{});
  session.reset_report_clock();

  const int n_steps = 4;
  for (int step = 0; step < n_steps; ++step) {
    ProfilingContextScope ctx(&session);
    session.begin_step_frame(step, 0);

    OPENPFC_PROFILE("demo_nest") {
      spin(2ms);
      OPENPFC_PROFILE("demo_nest/level_a") { spin(3ms); }
      for (int k = 0; k < 2; ++k) {
        OPENPFC_PROFILE("demo_nest/level_b") { spin(2ms); }
      }
    }

    OPENPFC_PROFILE("linear_solve") { spin(4ms + std::chrono::milliseconds(step)); }

    OPENPFC_PROFILE("assemble") {
      spin(2ms);
      OPENPFC_PROFILE("assemble/inner") { spin(5ms); }
    }

    {
      ProfilingManualScope timer;
      timer.restart("sequential/first");
      spin(1ms);
      timer.restart("sequential/second");
      spin(1ms);
    }

    session.add_recorded_time("manual_chunk",
                              0.0005 * (1.0 + 0.2 * static_cast<double>(step)));

    session.end_step_frame(0u, 0u, 0u);
  }

  ProfilingPrintOptions opts;
  opts.title = "OpenPFC — TimerOutputs-style report (demo)";
  opts.ascii_lines = true;
  opts.sort_by_time = true;
  opts.show_exclusive_column = true;

  std::cout
      << "Committed " << n_steps
      << " frames: paths auto-registered; OPENPFC_PROFILE uses unique locals.\n";
  print_profiling_timer(std::cout, session, opts);

  std::cout << "\n--- print_profiling_timer(os) with active session ---\n";
  {
    ProfilingContextScope again(&session);
    ProfilingPrintOptions via_ctx;
    via_ctx.title = "OpenPFC profiling (active session)";
    via_ctx.ascii_lines = true;
    print_profiling_timer(std::cout, via_ctx);
  }

  return 0;
}
