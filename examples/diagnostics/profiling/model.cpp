// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>
#include <vector>

int main() {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    namespace profiling = pfc::profiling;
    constexpr double dt = .01;
    constexpr int steps = 4, n = 4096;
    std::vector<double> u(n, 1), du(n);
    auto metrics = profiling::ProfilingSession::openpfc_default_frame_metrics();
    metrics.push_back("rms");
    profiling::ProfilingSession session(profiling::ProfilingMetricCatalog{},
                                        metrics);
    session.reset_report_clock();
    // start snippet scopes
    for (int step = 0; step < steps; ++step) {
      profiling::ProfilingContextScope context(&session);
      profiling::openpfc_begin_frame_with_step_and_rank(session, step, comm.rank());
      OPENPFC_PROFILE("step") {
        OPENPFC_PROFILE("step/rhs") {
          for (int i = 0; i < n; ++i) du[i] = -u[i];
        }
        OPENPFC_PROFILE("step/update") {
          for (int i = 0; i < n; ++i) u[i] += dt * du[i];
        }
      }
      {
        profiling::ProfilingManualScope timer;
        timer.restart("diagnostics/norm");
        double norm2 = 0;
        for (double value : u) norm2 += value * value;
        session.set_frame_metric("rms", std::sqrt(norm2 / n));
        timer.stop();
      }
      profiling::openpfc_end_frame_memory_only_wall_from_clock(session, 0u, 0u, 0u);
    }
    // end snippet scopes
    std::filesystem::create_directories("results");
    // start snippet export
    profiling::ProfilingExportOptions output;
    output.json_path = "results/profile.json";
    output.run_id = "teaching";
    output.export_metadata = {{"equation", "u_t=-u"}, {"dt", dt}};
    session.finalize_and_export(comm, output); // Every rank participates.
    profiling::ProfilingPrintOptions format;
    format.ascii_lines = true;
    format.show_exclusive_column = true;
    format.mpi_aggregate_stdout = true;
    profiling::print_profiling_timer(std::cout, comm, session, format);
    // end snippet export
    const double exact = std::pow(1 - dt, steps);
    double error = 0;
    for (double value : u) error = std::max(error, std::abs(value - exact));
    if (comm.rank() == 0)
      std::cout << "frames=" << session.num_frames() << " recurrence_error=" << error
                << '\n';
    return error < 1e-14 ? 0 : 1;
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
