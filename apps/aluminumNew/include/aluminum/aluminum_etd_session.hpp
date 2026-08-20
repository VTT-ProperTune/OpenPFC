// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file aluminum_etd_session.hpp
 * @brief JSON-driven CPU session: stack + AluminumPhysics + moving-frame ETD.
 *
 * M9 A/B driver. Gen-1 `aluminumNew` (`App<Aluminum>`) stays. This session
 * owns `SpectralCPUStack`, `SimulationState`, and
 * `MovingFrameMeanFieldETDSystem` — no model-owned FFT.
 */

#include <memory>
#include <optional>
#include <utility>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <aluminum/aluminum_etd_io.hpp>
#include <aluminum/aluminum_field_modifiers.hpp>
#include <aluminum/aluminum_physics.hpp>
#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/moving_frame_mean_field_etd.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace aluminum {

class AluminumETDSession {
public:
  AluminumETDSession(const AluminumETDSession &) = delete;
  AluminumETDSession &operator=(const AluminumETDSession &) = delete;
  AluminumETDSession(AluminumETDSession &&) = delete;
  AluminumETDSession &operator=(AluminumETDSession &&) = delete;

  AluminumETDSession(const nlohmann::json &settings, int rank, int nproc,
                     MPI_Comm comm = MPI_COMM_WORLD)
      : m_domain(pfc::ui::from_json<pfc::Domain>(settings)),
        m_time(pfc::ui::from_json<pfc::Time>(settings)),
        m_stack(m_domain, rank, nproc, comm) {
    AluminumPhysics<> phys;
    phys.domain = m_domain;
    phys.box = m_stack.fft().get_inbox_bounds();
    if (settings.contains("model") && settings["model"].contains("params")) {
      apply_aluminum_json(settings["model"]["params"], phys.params);
    }
    phys.declare_fields(m_state);
    auto &psi = m_state.get_field<double>("psi");
    apply_ics_from_json(settings, psi.domain(), psi.box(), psi.data(), psi.size());
    m_bc = parse_fixed_bc(settings);
    m_writers.configure(settings, m_domain, m_stack.fft().get_inbox_bounds(), comm,
                        rank);
    pfc::sim::MovingFrameMeanFieldETDOptions opt{};
    opt.comm = comm;
    m_sys =
        std::make_unique<pfc::sim::MovingFrameMeanFieldETDSystem<AluminumPhysics<>>>(
            std::move(phys), m_stack.fft(), m_state, pfc::time::dt(m_time),
            std::move(opt));
  }

  void run() {
    pfc::sim::SimulationDriver driver(m_time, &m_state);
    driver.run([&](double t) { m_sys->step(t); },
               [&](pfc::Time &) { apply_fixed_bc(); },
               [&](pfc::Time &) { apply_fixed_bc(); },
               [&](const pfc::Time &tm) { m_writers.maybe_write(tm, psi().vec()); });
  }

  [[nodiscard]] pfc::data::Field<double> &psi() {
    return m_state.get_field<double>("psi");
  }
  [[nodiscard]] const pfc::data::Field<double> &psi() const {
    return m_state.get_field<double>("psi");
  }
  [[nodiscard]] const pfc::Time &time() const noexcept { return m_time; }
  [[nodiscard]] pfc::sim::stacks::SpectralCPUStack &stack() noexcept {
    return m_stack;
  }
  [[nodiscard]] double last_free_energy() const { return m_sys->last_free_energy(); }
  [[nodiscard]] double last_free_energy_sum() const {
    return m_sys->last_free_energy_sum();
  }
  [[nodiscard]] int dumps() const noexcept { return m_writers.dumps(); }

private:
  void apply_fixed_bc() {
    if (!m_bc) {
      return;
    }
    auto &psi = m_state.get_field<double>("psi");
    aluminum::apply_fixed_bc(psi.domain(), psi.box(), psi.data(), *m_bc);
    psi.note_host_write();
  }

  pfc::Domain m_domain{};
  pfc::Time m_time;
  pfc::sim::stacks::SpectralCPUStack m_stack;
  pfc::SimulationState m_state;
  std::optional<FixedBc> m_bc{};
  AluminumETDWriters m_writers{};
  std::unique_ptr<pfc::sim::MovingFrameMeanFieldETDSystem<AluminumPhysics<>>> m_sys;
};

} // namespace aluminum
