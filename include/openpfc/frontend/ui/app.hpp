// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/app.hpp
 * @brief Application class for running simulations from JSON/TOML configuration
 *
 * @details
 * This header provides the App template class that orchestrates the entire
 * simulation workflow from configuration files. It handles:
 * - Reading JSON/TOML configuration files
 * - Creating and configuring the simulation world
 * - Setting up initial and boundary conditions
 * - Running the simulation loop
 * - Writing results
 * - Performance timing and reporting via `AppProfilingController` (see
 * docs/performance_profiling.md)
 *
 * @author OpenPFC Development Team
 * @date 2025
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
#include <openpfc/frontend/utils/logging.hpp>
#include <openpfc/frontend/utils/memory_reporter.hpp>
#include <openpfc/openpfc_minimal.hpp>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pfc::ui {

/**
 * @brief The main json-based application
 *
 */
template <class ConcreteModel> class App {
private:
  MPI_Comm m_comm;
  MPI_Worker m_worker;
  bool rank0;
  json m_settings;
  AppProfilingController m_profiling;

  // read settings from file (JSON or TOML format)
  json read_settings(int argc, char **argv) {
    const pfc::Logger lg{pfc::LogLevel::Info, m_worker.get_rank()};
    if (argc <= 1) {
      if (rank0) {
        pfc::log_error(lg, std::string("Configuration file required. Usage: ") +
                               argv[0] + " <config.json|config.toml>");
      }
      throw std::runtime_error("OpenPFC App: configuration file required (pass path "
                               "to JSON or TOML as first argument)");
    }

    std::filesystem::path file(argv[1]);
    auto ext = file.extension().string();

    if (rank0) {
      if (ext == ".toml") {
        pfc::log_info(lg, std::string("Reading TOML configuration from ") +
                              file.string());
      } else if (ext == ".json") {
        pfc::log_info(lg, std::string("Reading JSON configuration from ") +
                              file.string());
      }
    }

    try {
      return load_settings_file(file);
    } catch (const std::exception &err) {
      if (rank0) {
        pfc::log_error(lg, std::string("Failed to load settings: ") + err.what());
      }
      throw std::runtime_error(
          std::string("OpenPFC App: failed to load settings: ") + err.what());
    }
  }

public:
  App(int argc, char **argv, MPI_Comm comm = MPI_COMM_WORLD)
      : m_comm(comm), m_worker(MPI_Worker(argc, argv, comm)),
        rank0(m_worker.get_rank() == 0), m_settings(read_settings(argc, argv)) {}

  App(json settings, MPI_Comm comm = MPI_COMM_WORLD)
      : m_comm(comm), m_worker(MPI_Worker(0, nullptr, comm)),
        rank0(m_worker.get_rank() == 0), m_settings(std::move(settings)) {}

  bool create_results_dir(const std::string &output) {
    return ensure_results_parent_dir_for_writer(output, m_worker.get_rank());
  }

  int main() {
    const int rank_id = m_worker.get_rank();
    const pfc::Logger app_lg{pfc::LogLevel::Info, rank_id};
#if defined(OpenPFC_ENABLE_HIP) && defined(OpenPFC_MPI_HIP_AWARE)
    if (rank0) {
      pfc::log_info(app_lg,
                    "OpenPFC: GPU-aware MPI (HIP) is enabled at compile time.");
      const char *gpu_env = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
      if (gpu_env == nullptr || std::string(gpu_env) != "1") {
        pfc::log_warning(
            app_lg, "MPICH_GPU_SUPPORT_ENABLED is not set to 1; Cray MPICH may not "
                    "accept device pointers (set export MPICH_GPU_SUPPORT_ENABLED=1 "
                    "in your job).");
      }
    }
#endif
#if defined(OpenPFC_ENABLE_CUDA) && defined(OpenPFC_MPI_CUDA_AWARE)
    if (rank0) {
      pfc::log_info(app_lg,
                    "OpenPFC: GPU-aware MPI (CUDA) is enabled at compile time.");
    }
#endif
    if (rank0) {
      pfc::log_info(app_lg, "Reading configuration from json file:");
      pfc::log_info(app_lg, m_settings.dump(4));
    }

    auto session = SpectralSimulationSession<ConcreteModel>::assemble(
        m_settings, m_comm, rank_id, m_worker.get_num_ranks());
    if (rank0) {
      std::ostringstream woss;
      woss << session->world();
      pfc::log_info(app_lg, std::string("World: ") + woss.str());
    }

    if (m_settings.contains("model") && m_settings["model"].contains("params")) {
      from_json(m_settings["model"]["params"], session->model());
    }
    m_profiling.configure_from_root_settings(m_settings, rank_id, rank0);

    if (rank0) {
      pfc::log_info(app_lg, "Initializing model...");
    }
    session->model().initialize(session->time().get_dt());

    // Report memory usage
    {
      size_t model_mem = session->model().get_allocated_memory_bytes();
      size_t fft_mem = session->fft().get_allocated_memory_bytes();
      pfc::utils::MemoryUsage usage{model_mem, fft_mem};
      pfc::Logger logger{pfc::LogLevel::Info, rank_id};
      pfc::utils::report_memory_usage(usage, session->world(), logger, m_comm);
    }

    session->wire_simulator_from_settings(m_settings, m_comm, rank_id, rank0);

    if (rank0) {
      pfc::log_info(app_lg, "Starting time integration (Simulator integrator API)");
    }

    (void)run_simulator_time_integration_loop(*session, m_comm, rank_id, rank0,
                                              m_profiling.session(),
                                              m_profiling.memory_samples(), app_lg);

    m_profiling.finalize_and_export_if_active(m_settings, m_comm, rank0, app_lg);

    return 0;
  }
};

} // namespace pfc::ui

#endif // PFC_UI_APP_HPP
