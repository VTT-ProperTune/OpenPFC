// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/app.hpp
 * @brief Application class for running simulations from JSON/TOML configuration
 *
 * @details
 * Orchestrates loading settings, building a `SpectralSimulationSession`, wiring
 * the simulator from JSON, running the time loop, and optional profiling export.
 * Detailed steps live in private methods so `main()` reads as a high-level script.
 *
 * @see docs/performance_profiling.md for profiling configuration
 */

#ifndef PFC_UI_APP_HPP
#define PFC_UI_APP_HPP

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/app_integrator_loop.hpp>
#include <openpfc/frontend/ui/app_profiling.hpp>
#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/frontend/ui/json_helpers.hpp>
#include <openpfc/frontend/ui/settings_loader.hpp>
#include <openpfc/frontend/ui/spectral_simulation_session.hpp>
#include <openpfc/frontend/utils/memory_reporter.hpp>
#include <openpfc/frontend/utils/nancheck.hpp>
#include <openpfc/kernel/utils/logging.hpp>
#include <openpfc/openpfc_minimal.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pfc::ui {

/** @brief Prefix for structured application log lines */
inline constexpr std::string_view k_app_log_tag = "[app] ";

/**
 * @brief JSON-driven MPI application entry point for a concrete physics model
 */
template <class ConcreteModel> class App {
private:
  MPI_Comm m_comm;
  MPI_Worker m_worker;
  bool rank0;
  json m_settings;
  AppProfilingController m_profiling;

  [[nodiscard]] json read_settings(int argc, char **argv) {
    const pfc::Logger lg{pfc::LogLevel::Info, m_worker.get_rank()};
    if (argc <= 1) {
      if (rank0) {
        pfc::log_error(lg, std::string(k_app_log_tag) +
                               "Configuration file required. Usage: " +
                               std::string(argv[0]) + " <config.json|config.toml>");
      }
      throw std::runtime_error("OpenPFC App: configuration file required (pass path "
                               "to JSON or TOML as first argument)");
    }

    const std::filesystem::path file(argv[1]);
    const auto ext = file.extension().string();

    if (rank0) {
      if (ext == ".toml") {
        pfc::log_info(lg, std::string(k_app_log_tag) +
                              "Reading TOML configuration from " + file.string());
      } else if (ext == ".json") {
        pfc::log_info(lg, std::string(k_app_log_tag) +
                              "Reading JSON configuration from " + file.string());
      }
    }

    try {
      return load_settings_file(file);
    } catch (const std::exception &err) {
      if (rank0) {
        pfc::log_error(lg, std::string(k_app_log_tag) +
                               "Failed to load settings: " + err.what());
      }
      throw std::runtime_error(
          std::string("OpenPFC App: failed to load settings: ") + err.what());
    }
  }

  void log_gpu_awareness_hints_([[maybe_unused]] const pfc::Logger &app_lg) const {
#if defined(OpenPFC_ENABLE_HIP) && defined(OpenPFC_MPI_HIP_AWARE)
    if (rank0) {
      pfc::log_info(app_lg,
                    std::string(k_app_log_tag) +
                        "OpenPFC: GPU-aware MPI (HIP) is enabled at compile time.");
      const char *gpu_env = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
      if (gpu_env == nullptr || std::string(gpu_env) != "1") {
        pfc::log_warning(
            app_lg,
            std::string(k_app_log_tag) +
                "MPICH_GPU_SUPPORT_ENABLED is not set to 1; Cray MPICH may not "
                "accept device pointers (set export MPICH_GPU_SUPPORT_ENABLED=1 "
                "in your job).");
      }
    }
#endif
#if defined(OpenPFC_ENABLE_CUDA) && defined(OpenPFC_MPI_CUDA_AWARE)
    if (rank0) {
      pfc::log_info(app_lg,
                    std::string(k_app_log_tag) +
                        "OpenPFC: GPU-aware MPI (CUDA) is enabled at compile time.");
    }
#endif
  }

  void log_effective_configuration_(const pfc::Logger &app_lg) const {
    if (rank0) {
      pfc::log_info(app_lg,
                    std::string(k_app_log_tag) + "Effective configuration (JSON):");
      pfc::log_info(app_lg, m_settings.dump(4));
    }
  }

  [[nodiscard]] std::unique_ptr<SpectralSimulationSession<ConcreteModel>>
  build_spectral_session_(int rank_id) const {
    return SpectralSimulationSession<ConcreteModel>::assemble(
        m_settings, m_comm, rank_id, m_worker.get_num_ranks());
  }

  void
  log_world_summary_(const pfc::Logger &app_lg,
                     const SpectralSimulationSession<ConcreteModel> &session) const {
    if (rank0) {
      std::ostringstream woss;
      woss << session.world();
      pfc::log_info(app_lg, std::string(k_app_log_tag) + "World: " + woss.str());
    }
  }

  void apply_model_params_from_settings_(
      SpectralSimulationSession<ConcreteModel> &session) const {
    if (m_settings.contains("model") && m_settings["model"].contains("params")) {
      from_json(m_settings["model"]["params"], session.model());
    }
  }

  void configure_profiling_(int rank_id) {
    m_profiling.configure_from_root_settings(m_settings, rank_id, rank0);
  }

  void initialize_model_(const pfc::Logger &app_lg,
                         SpectralSimulationSession<ConcreteModel> &session) const {
    if (rank0) {
      pfc::log_info(app_lg, std::string(k_app_log_tag) + "Initializing model...");
    }
    session.model().initialize(session.time().get_dt());
  }

  void report_memory_usage_(
      int rank_id, const SpectralSimulationSession<ConcreteModel> &session) const {
    const size_t model_mem = session.model().get_allocated_memory_bytes();
    const size_t fft_mem = session.fft().get_allocated_memory_bytes();
    const pfc::utils::MemoryUsage usage{model_mem, fft_mem};
    const pfc::Logger logger{pfc::LogLevel::Info, rank_id};
    pfc::utils::report_memory_usage(usage, session.world(), logger, m_comm);
  }

  void wire_simulator_and_log_run_start_(
      const pfc::Logger &app_lg, int rank_id,
      SpectralSimulationSession<ConcreteModel> &session) {
    session.wire_simulator_from_settings(m_settings, rank_id, rank0);
    if (rank0) {
      pfc::log_info(app_lg,
                    std::string(k_app_log_tag) +
                        "Starting time integration (Simulator integrator API)");
    }
  }

  void run_time_integration_(const pfc::Logger &app_lg, int rank_id,
                             SpectralSimulationSession<ConcreteModel> &session) {
    (void)run_simulator_time_integration_loop(session, m_comm, rank_id, rank0,
                                              m_profiling.session(),
                                              m_profiling.memory_samples(), app_lg);
  }

  void finalize_profiling_export_(const pfc::Logger &app_lg) {
    m_profiling.finalize_and_export_if_active(m_settings, m_comm, rank0, app_lg);
  }

public:
  App(int argc, char **argv, MPI_Comm comm = MPI_COMM_WORLD)
      : m_comm(comm), m_worker(MPI_Worker(argc, argv, comm)),
        rank0(m_worker.get_rank() == 0), m_settings(read_settings(argc, argv)) {}

  App(json settings, MPI_Comm comm = MPI_COMM_WORLD)
      : m_comm(comm), m_worker(MPI_Worker(0, nullptr, comm)),
        rank0(m_worker.get_rank() == 0), m_settings(std::move(settings)) {}

  [[nodiscard]] bool create_results_dir(const std::string &output) {
    return ensure_results_parent_dir_for_writer(output, m_worker.get_rank());
  }

  [[nodiscard]] int main() {
    const int rank_id = m_worker.get_rank();
    // So `from_json` diagnostics (FFT backend, HeFFTe options) use the same rank
    // prefix as app logs.
    set_from_json_log_rank(rank_id);
    pfc::utils::set_default_nan_check_mpi_comm(m_comm);
    const pfc::Logger app_lg{pfc::LogLevel::Info, rank_id};

    log_gpu_awareness_hints_(app_lg);
    log_effective_configuration_(app_lg);

    auto session = build_spectral_session_(rank_id);
    log_world_summary_(app_lg, *session);

    apply_model_params_from_settings_(*session);
    configure_profiling_(rank_id);

    initialize_model_(app_lg, *session);
    report_memory_usage_(rank_id, *session);

    wire_simulator_and_log_run_start_(app_lg, rank_id, *session);
    run_time_integration_(app_lg, rank_id, *session);
    finalize_profiling_export_(app_lg);

    return 0;
  }
};

} // namespace pfc::ui

#endif // PFC_UI_APP_HPP
