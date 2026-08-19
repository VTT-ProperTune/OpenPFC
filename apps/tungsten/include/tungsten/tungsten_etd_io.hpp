// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file tungsten_etd_io.hpp
 * @brief JSON `fields[]` binary writers for tungsten ETD sessions.
 *
 * Uses `Time::do_save()` like Gen-1 `Simulator`. Only the real `psi` field
 * is written; `psiMF` entries are skipped (mean-field is a work array).
 */

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/results_writer_catalog.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace tungsten {

class TungstenEtdWriters {
public:
  void configure(const nlohmann::json &settings, const pfc::Domain &domain,
                 const pfc::Box3i &inbox, MPI_Comm comm, int rank) {
    m_comm = comm;
    m_rank = rank;
    if (!settings.contains("fields") || !settings["fields"].is_array()) {
      return;
    }
    auto &catalog = pfc::ui::default_results_writer_catalog();
    const auto gs = pfc::domain::get_size(domain);
    const std::array<int, 3> global{gs[0], gs[1], gs[2]};
    const std::array<int, 3> local{inbox.size[0], inbox.size[1], inbox.size[2]};
    const std::array<int, 3> offset{inbox.low[0], inbox.low[1], inbox.low[2]};
    for (const auto &field : settings["fields"]) {
      const std::string name = field.value("name", std::string{});
      if (name != "psi") {
        continue;
      }
      const std::string data = field.at("data").get<std::string>();
      std::string writer_type = "binary";
      if (field.contains("writer") && field["writer"].is_string()) {
        writer_type = field["writer"].get<std::string>();
      }
      if (m_rank == 0) {
        std::filesystem::path dir(data);
        if (dir.has_filename()) {
          dir = dir.parent_path();
        }
        if (!dir.empty()) {
          std::filesystem::create_directories(dir);
        }
      }
      MPI_Barrier(m_comm);
      auto writer_opt = catalog.try_create(writer_type, data, m_comm);
      if (!writer_opt) {
        continue;
      }
      (*writer_opt)->set_domain(global, local, offset);
      m_psi_writer = std::move(*writer_opt);
    }
  }

  void maybe_write(const pfc::Time &time, const std::vector<double> &psi) {
    if (!m_psi_writer || !pfc::time::do_save(time)) {
      return;
    }
    m_psi_writer->write(m_counter, psi);
    ++m_counter;
  }

  [[nodiscard]] int dumps() const noexcept { return m_counter; }
  [[nodiscard]] bool enabled() const noexcept {
    return static_cast<bool>(m_psi_writer);
  }

private:
  MPI_Comm m_comm{MPI_COMM_WORLD};
  int m_rank{0};
  int m_counter{0};
  std::unique_ptr<pfc::ResultsWriter> m_psi_writer;
};

} // namespace tungsten
