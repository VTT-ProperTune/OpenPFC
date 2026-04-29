// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file app_integrator_loop.hpp
 * @brief Time integration loop used by JSON `App` (profiling + step + ETA logs)
 */

#ifndef PFC_UI_APP_INTEGRATOR_LOOP_HPP
#define PFC_UI_APP_INTEGRATOR_LOOP_HPP

#include <cstdint>
#include <sstream>

#include <mpi.h>
#include <openpfc/frontend/ui/spectral_simulation_session.hpp>
#include <openpfc/frontend/utils/timeleft.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/utils/logging.hpp>

namespace pfc::ui {

/** @brief Aggregates wall-clock samples accumulated over
 * `run_simulator_time_integration_loop` */
struct IntegratorTimings {
  double total_step_time = 0.0;
  double total_fft_time = 0.0;
  int steps_completed = 0;
};

/**
 * @brief Run `Simulator` integrator API until `Time::done()`, with optional
 * profiling
 */
template <class ConcreteModel>
IntegratorTimings run_simulator_time_integration_loop(
    SpectralSimulationSession<ConcreteModel> &session, MPI_Comm comm, int rank_id,
    bool rank0, pfc::profiling::ProfilingSession *profiler,
    bool profiler_memory_samples, const pfc::Logger &app_lg) {
  IntegratorTimings out{};
  // Exponential moving average of step wall time for ETA (after a short warm-up).
  double avg_step_ema = 0.0;

  while (!session.time().done()) {
    session.fft().reset_fft_time();
    session.simulator().begin_integrator_step();
    const double barrier_step_s = pfc::profiling::measure_barriered(comm, [&] {
      if (profiler) {
        pfc::profiling::openpfc_begin_frame_with_step_and_rank(
            *profiler, session.time().get_increment(), rank_id);
      }
      if (profiler) {
        pfc::profiling::ProfilingContextScope scope(profiler);
        step(session.simulator(), session.model());
      } else {
        step(session.simulator(), session.model());
      }
    });
    const double fft_meter_s = session.fft().get_fft_time();

    std::uint64_t rss = 0;
    std::uint64_t model_mem = 0;
    std::uint64_t fft_mem = 0;
    if (profiler && profiler_memory_samples) {
      rss = pfc::profiling::try_read_process_rss_bytes();
      model_mem = session.model().get_allocated_memory_bytes();
      fft_mem = session.fft().get_allocated_memory_bytes();
    }
    if (profiler) {
      profiler->assign_recorded_time("fft", fft_meter_s);
      pfc::profiling::openpfc_end_frame_step_wall_and_memory(
          *profiler, barrier_step_s, rss, model_mem, fft_mem);
    }

    const double steptime =
        pfc::profiling::reduce_max_to_root(comm, barrier_step_s, 0);
    const double fft_time = pfc::profiling::reduce_max_to_root(comm, fft_meter_s, 0);

    session.simulator().end_integrator_step();

    if (out.steps_completed > 3) {
      avg_step_ema = 0.01 * steptime + 0.99 * avg_step_ema;
    } else {
      avg_step_ema = steptime;
    }
    const int increment = session.time().get_increment();
    const double t = session.time().get_current();
    const double t1 = session.time().get_t1();
    const double eta_i = (t1 - t) / session.time().get_dt();
    const double eta_t = eta_i * avg_step_ema;
    const double other_time = steptime - fft_time;
    if (rank0) {
      std::ostringstream steposs;
      steposs << "Step " << increment << " done in " << steptime << " s ("
              << fft_time << " s FFT, " << other_time
              << " s other). Simulation time: " << t << " / " << t1 << " ("
              << (t / t1 * 100) << " % done). ETA: " << pfc::utils::TimeLeft(eta_t);
      pfc::log_info(app_lg, steposs.str());
    }

    out.total_step_time += steptime;
    out.total_fft_time += fft_time;
    out.steps_completed += 1;
  }

  if (out.steps_completed > 0) {
    const double avg_steptime = out.total_step_time / out.steps_completed;
    const double avg_fft_time = out.total_fft_time / out.steps_completed;
    const double avg_oth_time = avg_steptime - avg_fft_time;
    const double p_fft = avg_fft_time / avg_steptime * 100.0;
    const double p_oth = avg_oth_time / avg_steptime * 100.0;
    if (rank0) {
      std::ostringstream sumoss;
      sumoss << "Simulated " << out.steps_completed << " steps. Average times:\n"
             << "Step time:  " << avg_steptime << " s\n"
             << "FFT time:   " << avg_fft_time << " s / " << p_fft << " %\n"
             << "Other time: " << avg_oth_time << " s / " << p_oth << " %";
      pfc::log_info(app_lg, sumoss.str());
    }
  } else if (rank0) {
    pfc::log_info(
        app_lg,
        "No complete timesteps were executed; skipping average timing summary.");
  }

  return out;
}

} // namespace pfc::ui

#endif // PFC_UI_APP_INTEGRATOR_LOOP_HPP
