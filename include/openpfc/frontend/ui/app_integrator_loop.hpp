// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file app_integrator_loop.hpp
 * @brief Time integration loop used by JSON `App` (profiling + step + ETA logs)
 */

#ifndef PFC_UI_APP_INTEGRATOR_LOOP_HPP
#define PFC_UI_APP_INTEGRATOR_LOOP_HPP

#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>

#include <mpi.h>
#include <openpfc/frontend/ui/spectral_simulation_session.hpp>
#include <openpfc/frontend/utils/timeleft.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <openpfc/kernel/utils/logging.hpp>

namespace pfc::ui {

/** @brief Aggregates wall-clock samples accumulated over
 * `run_simulator_time_integration_loop` */
struct IntegratorTimings {
  double total_step_time = 0.0;
  double total_fft_time = 0.0;
  int steps_completed = 0;
};

/** @brief MPI rank, logging, and profiling knobs for the spectral integrator loop */
struct SimulatorIntegratorLoopEnv {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank_id = 0;
  bool rank0 = false;
  pfc::profiling::ProfilingSession *profiler = nullptr;
  bool profiler_memory_samples = false;
  const pfc::Logger *app_log = nullptr;
};

/**
 * @brief Run `Simulator` integrator API until `Time::done()`, with optional
 * profiling
 */
template <class ConcreteModel>
IntegratorTimings run_simulator_time_integration_loop(
    SpectralSimulationSession<ConcreteModel> &session,
    const SimulatorIntegratorLoopEnv &env) {
  const pfc::Logger &app_lg = *env.app_log;
  IntegratorTimings out{};
  // Exponential moving average of step wall time for ETA (after a short warm-up).
  double avg_step_ema = 0.0;

  while (!pfc::time::done(time(session))) {
    pfc::fft::reset_fft_time(fft(session));
    pfc::begin_integrator_step(simulator(session));
    const double barrier_step_s = pfc::profiling::measure_barriered(env.comm, [&] {
      std::optional<pfc::profiling::ProfilingContextScope> profile_ctx;
      if (env.profiler) {
        pfc::profiling::openpfc_begin_frame_with_step_and_rank(
            *env.profiler, pfc::time::increment(time(session)), env.rank_id);
        profile_ctx.emplace(env.profiler);
      }
      step(simulator(session), model(session));
    });
    const double fft_meter_s = pfc::fft::get_fft_time(fft(session));

    std::uint64_t rss = 0;
    std::uint64_t model_mem = 0;
    std::uint64_t fft_mem = 0;
    if (env.profiler && env.profiler_memory_samples) {
      rss = pfc::profiling::try_read_process_rss_bytes();
      model_mem = model(session).get_allocated_memory_bytes();
      fft_mem = fft(session).get_allocated_memory_bytes();
    }
    if (env.profiler) {
      env.profiler->assign_recorded_time("fft", fft_meter_s);
      pfc::profiling::openpfc_end_frame_step_wall_and_memory(
          *env.profiler, barrier_step_s, rss, model_mem, fft_mem);
    }

    const double steptime =
        pfc::profiling::reduce_max_to_root(env.comm, barrier_step_s, 0);
    const double fft_time =
        pfc::profiling::reduce_max_to_root(env.comm, fft_meter_s, 0);

    pfc::end_integrator_step(simulator(session));

    if (out.steps_completed > 3) {
      avg_step_ema = 0.01 * steptime + 0.99 * avg_step_ema;
    } else {
      avg_step_ema = steptime;
    }
    const int increment = pfc::time::increment(time(session));
    const double t = pfc::time::current(time(session));
    const double t1 = pfc::time::t1(time(session));
    const double dt = pfc::time::dt(time(session));
    double eta_t = 0.0;
    if (dt > 0.0 && std::isfinite(dt) && t1 > t) {
      const double eta_i = (t1 - t) / dt;
      if (std::isfinite(eta_i)) {
        eta_t = eta_i * avg_step_ema;
      }
    }
    const double other_time = steptime - fft_time;
    if (env.rank0) {
      std::ostringstream steposs;
      steposs << "Step " << increment << " done in " << steptime << " s ("
              << fft_time << " s FFT, " << other_time
              << " s other). Simulation time: " << t << " / " << t1;
      if (t1 > 0.0 && std::isfinite(t) && std::isfinite(t1)) {
        steposs << " (" << (t / t1 * 100.0) << " % done)";
      } else {
        steposs << " (progress n/a: end time must be > 0 for %)";
      }
      steposs << ". ETA: ";
      if (dt > 0.0 && t1 > t && std::isfinite(eta_t) && eta_t >= 0.0) {
        steposs << pfc::utils::TimeLeft(eta_t);
      } else {
        steposs << "n/a";
      }
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
    if (env.rank0) {
      std::ostringstream sumoss;
      sumoss << "Simulated " << out.steps_completed << " steps. Average times:\n"
             << "Step time:  " << avg_steptime << " s\n"
             << "FFT time:   " << avg_fft_time << " s / " << p_fft << " %\n"
             << "Other time: " << avg_oth_time << " s / " << p_oth << " %";
      pfc::log_info(app_lg, sumoss.str());
    }
  } else if (env.rank0) {
    pfc::log_info(
        app_lg,
        "No complete timesteps were executed; skipping average timing summary.");
  }

  return out;
}

/**
 * @brief Legacy overload: packs @p comm / rank / profiler into @ref
 *        SimulatorIntegratorLoopEnv.
 */
template <class ConcreteModel>
IntegratorTimings run_simulator_time_integration_loop(
    SpectralSimulationSession<ConcreteModel> &session, MPI_Comm comm, int rank_id,
    bool rank0, pfc::profiling::ProfilingSession *profiler,
    bool profiler_memory_samples, const pfc::Logger &app_lg) {
  return run_simulator_time_integration_loop(
      session, SimulatorIntegratorLoopEnv{comm, rank_id, rank0, profiler,
                                          profiler_memory_samples, &app_lg});
}

} // namespace pfc::ui

#endif // PFC_UI_APP_INTEGRATOR_LOOP_HPP
