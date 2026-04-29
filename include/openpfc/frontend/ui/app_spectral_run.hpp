// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file app_spectral_run.hpp
 * @brief JSON spectral `App` run pipeline (session build → wire → integrate)
 *
 * @details
 * `SpectralJsonAppRun` holds the collaborators needed after settings are loaded
 * and logging hooks are configured. It keeps `App` focused on construction,
 * settings I/O, and optional field-modifier catalog injection.
 *
 * @see app.hpp
 * @see docs/app_pipeline.md
 */

#ifndef PFC_UI_APP_SPECTRAL_RUN_HPP
#define PFC_UI_APP_SPECTRAL_RUN_HPP

#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <string_view>

#include <mpi.h>

#include <openpfc/frontend/ui/app_integrator_loop.hpp>
#include <openpfc/frontend/ui/app_profiling.hpp>
#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/frontend/ui/spectral_simulation_session.hpp>
#include <openpfc/frontend/utils/memory_reporter.hpp>
#include <openpfc/kernel/mpi/worker.hpp>
#include <openpfc/kernel/utils/logging.hpp>

namespace pfc::ui {

namespace detail {

inline constexpr std::string_view k_spectral_app_log_tag = "[app] ";

} // namespace detail

/**
 * @brief Executes the default spectral JSON pipeline for a concrete model
 */
template <class ConcreteModel> class SpectralJsonAppRun {
public:
  SpectralJsonAppRun(const nlohmann::json &settings, MPI_Comm comm,
                     const MPI_Worker &worker, bool rank0,
                     AppProfilingController &profiling,
                     const FieldModifierCatalog *modifier_catalog_override)
      : m_settings(settings), m_comm(comm), m_worker(worker), m_rank0(rank0),
        m_profiling(profiling),
        m_modifier_catalog_override(modifier_catalog_override) {}

  [[nodiscard]] int execute(const pfc::Logger &app_lg) {
    const int rank_id = m_worker.get_rank();

    auto session = build_session_(rank_id);
    log_world_summary_(app_lg, *session);

    apply_model_params_(*session);
    configure_profiling_(rank_id);

    initialize_model_(app_lg, *session);
    report_memory_usage_(rank_id, *session);

    wire_simulator_and_log_run_start_(app_lg, rank_id, *session);
    run_time_integration_(app_lg, rank_id, *session);
    finalize_profiling_export_(app_lg);

    return 0;
  }

private:
  const nlohmann::json &m_settings;
  MPI_Comm m_comm;
  const MPI_Worker &m_worker;
  bool m_rank0;
  AppProfilingController &m_profiling;
  const FieldModifierCatalog *m_modifier_catalog_override;

  [[nodiscard]] std::unique_ptr<SpectralSimulationSession<ConcreteModel>>
  build_session_(int rank_id) const {
    return SpectralSimulationSession<ConcreteModel>::assemble(
        m_settings, m_comm, rank_id, m_worker.get_num_ranks());
  }

  void
  log_world_summary_(const pfc::Logger &app_lg,
                     const SpectralSimulationSession<ConcreteModel> &session) const {
    if (m_rank0) {
      std::ostringstream woss;
      woss << session.world();
      pfc::log_info(app_lg, std::string(detail::k_spectral_app_log_tag) +
                                "World: " + woss.str());
    }
  }

  void apply_model_params_(SpectralSimulationSession<ConcreteModel> &session) const {
    if (m_settings.contains("model") && m_settings["model"].contains("params")) {
      from_json(m_settings["model"]["params"], session.model());
    }
  }

  void configure_profiling_(int rank_id) {
    m_profiling.configure_from_root_settings(m_settings, rank_id, m_rank0);
  }

  void initialize_model_(const pfc::Logger &app_lg,
                         SpectralSimulationSession<ConcreteModel> &session) const {
    if (m_rank0) {
      pfc::log_info(app_lg, std::string(detail::k_spectral_app_log_tag) +
                                "Initializing model...");
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
    const FieldModifierCatalog &catalog = m_modifier_catalog_override
                                              ? *m_modifier_catalog_override
                                              : default_field_modifier_catalog();
    session.wire_simulator_from_settings(m_settings, rank_id, m_rank0, catalog);
    if (m_rank0) {
      pfc::log_info(app_lg,
                    std::string(detail::k_spectral_app_log_tag) +
                        "Starting time integration (Simulator integrator API)");
    }
  }

  void run_time_integration_(const pfc::Logger &app_lg, int rank_id,
                             SpectralSimulationSession<ConcreteModel> &session) {
    (void)run_simulator_time_integration_loop(session, m_comm, rank_id, m_rank0,
                                              m_profiling.session(),
                                              m_profiling.memory_samples(), app_lg);
  }

  void finalize_profiling_export_(const pfc::Logger &app_lg) {
    m_profiling.finalize_and_export_if_active(m_settings, m_comm, m_rank0, app_lg);
  }
};

} // namespace pfc::ui

#endif // PFC_UI_APP_SPECTRAL_RUN_HPP
