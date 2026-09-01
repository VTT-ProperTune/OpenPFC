// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file tungsten_etd_profile.hpp
 * @brief Optional JSON `profiling` block for tungsten ETD sessions.
 *
 * Same `AppProfilingController` + barriered `wall_step` as Gen-1 `App`, so
 * `scripts/compare_perf_baseline.py` can compare 0.2 vs stored Gen-1 JSON.
 */

#include <optional>
#include <utility>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/app_profiling.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>
#include <openpfc/kernel/utils/logging.hpp>

namespace tungsten {

class EtdProfileEnv {
public:
  EtdProfileEnv(nlohmann::json settings, int rank, MPI_Comm comm)
      : m_settings(std::move(settings)), m_comm(comm), m_rank(rank),
        m_logger{pfc::LogLevel::Info, rank} {
    m_controller.configure_from_root_settings(m_settings, m_rank, m_rank == 0);
  }

  template <class Step>
  void timed_step(int increment, pfc::fft::IFFTQueries &fft, Step &&step) {
    auto *prof = m_controller.session();
    if (prof == nullptr) {
      step();
      return;
    }
    pfc::fft::reset_fft_time(fft);
    const double wall = pfc::profiling::measure_barriered(m_comm, [&] {
      std::optional<pfc::profiling::ProfilingContextScope> ctx;
      pfc::profiling::openpfc_begin_frame_with_step_and_rank(*prof, increment,
                                                             m_rank);
      ctx.emplace(prof);
      step();
    });
    const double fft_s = pfc::fft::get_fft_time(fft);
    pfc::profiling::openpfc_end_frame_with_fft_region_wall_and_memory(
        *prof, wall, fft_s, 0, 0, 0);
  }

  void finalize() {
    m_controller.finalize_and_export_if_active(m_settings, m_comm, m_rank == 0,
                                               m_logger);
  }

  [[nodiscard]] bool enabled() const noexcept { return m_controller.enabled(); }

private:
  nlohmann::json m_settings;
  MPI_Comm m_comm;
  int m_rank;
  pfc::Logger m_logger;
  pfc::ui::AppProfilingController m_controller;
};

} // namespace tungsten
