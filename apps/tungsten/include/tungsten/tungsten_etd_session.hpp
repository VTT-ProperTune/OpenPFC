// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file tungsten_etd_session.hpp
 * @brief JSON-driven CPU session: stack + TungstenPhysics + mean-field ETD.
 *
 * @details
 * M8 A/B driver. Gen-1 `tungsten` (`App<Tungsten>`) stays. This session owns
 * `SpectralCPUStack`, `SimulationState`, and `SpectralMeanFieldETDSystem` —
 * no model-owned FFT. Initial conditions and fixed BCs are applied on the
 * `Field` (same formulas as Gen-1 `Constant` / `SingleSeed` / `FixedBC`).
 * Binary `psi` dumps follow `Time::do_save()` when JSON `fields` is set.
 */

#include <memory>
#include <optional>
#include <utility>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_mean_field_etd.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <tungsten/tungsten_etd_io.hpp>
#include <tungsten/tungsten_field_modifiers.hpp>
#include <tungsten/tungsten_physics.hpp>

namespace tungsten {

class TungstenETDSession {
public:
  TungstenETDSession(const TungstenETDSession &) = delete;
  TungstenETDSession &operator=(const TungstenETDSession &) = delete;
  TungstenETDSession(TungstenETDSession &&) = delete;
  TungstenETDSession &operator=(TungstenETDSession &&) = delete;

  TungstenETDSession(const nlohmann::json &settings, int rank, int nproc,
                     MPI_Comm comm = MPI_COMM_WORLD)
      : m_domain(pfc::ui::from_json<pfc::Domain>(settings)),
        m_time(pfc::ui::from_json<pfc::Time>(settings)),
        m_stack(m_domain, rank, nproc, comm) {
    TungstenPhysics<> phys;
    phys.domain = m_domain;
    phys.box = m_stack.fft().get_inbox_bounds();
    if (settings.contains("model") && settings["model"].contains("params")) {
      apply_tungsten_json(settings["model"]["params"], phys.params);
    }
    phys.declare_fields(m_state);
    auto &psi = m_state.get_field<double>("psi");
    apply_ics_from_json(settings, psi.domain(), psi.box(), psi.data(),
                        psi.size());
    m_bc = parse_fixed_bc(settings);
    m_writers.configure(settings, m_domain, m_stack.fft().get_inbox_bounds(),
                        comm, rank);
    m_sys = std::make_unique<
        pfc::sim::SpectralMeanFieldETDSystem<TungstenPhysics<>>>(
        std::move(phys), m_stack.fft(), m_state, pfc::time::dt(m_time));
  }

  void run() {
    while (!pfc::time::done(m_time)) {
      if (pfc::time::increment(m_time) == 0) {
        apply_fixed_bc();
        m_writers.maybe_write(m_time, psi().vec());
      }
      pfc::time::next(m_time);
      apply_fixed_bc();
      m_sys->step(pfc::time::current(m_time));
      m_writers.maybe_write(m_time, psi().vec());
    }
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
  [[nodiscard]] int dumps() const noexcept { return m_writers.dumps(); }

private:
  void apply_fixed_bc() {
    if (!m_bc) {
      return;
    }
    auto &psi = m_state.get_field<double>("psi");
    tungsten::apply_fixed_bc(psi.domain(), psi.box(), psi.data(), *m_bc);
    psi.note_host_write();
  }

  pfc::Domain m_domain{};
  pfc::Time m_time;
  pfc::sim::stacks::SpectralCPUStack m_stack;
  pfc::SimulationState m_state;
  std::optional<FixedBc> m_bc{};
  TungstenETDWriters m_writers{};
  std::unique_ptr<pfc::sim::SpectralMeanFieldETDSystem<TungstenPhysics<>>>
      m_sys;
};

} // namespace tungsten
