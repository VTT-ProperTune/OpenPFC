// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file tungsten_etd_session.hpp
 * @brief JSON-driven CPU session: stack + TungstenPhysics + mean-field ETD.
 *
 * @details
 * Production `tungsten` driver. This session owns
 * `SpectralCPUStack`, `SimulationState`, and `SpectralMeanFieldETDSystem` —
 * no model-owned FFT. Initial conditions and fixed BCs are applied on the
 * `Field` (same formulas as Gen-1 `Constant` / `SingleSeed` / `FixedBC`).
 * Binary / VTK `psi` dumps follow `Time::do_save()` when JSON `fields` is set.
 * Optional JSON `profiling` uses the same `wall_step` exporter as Gen-1 `App`.
 */

#include <memory>
#include <optional>
#include <utility>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/frontend/ui/json_checkpoint.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/simulation/checkpoint_service.hpp>
#include <openpfc/kernel/simulation/session_stack_factory.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_mean_field_etd.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <tungsten/tungsten_etd_io.hpp>
#include <tungsten/tungsten_etd_profile.hpp>
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
        m_stack(pfc::sim::make_spectral_cpu_stack(
            pfc::ui::from_json<pfc::sim::SessionSelection>(settings), m_domain, rank,
            nproc, comm)),
        m_ckpt(pfc::ui::make_checkpoint_service(settings, comm)),
        m_profile(settings, rank, comm) {
    TungstenPhysics<> phys;
    phys.domain = m_domain;
    phys.box = m_stack.fft().get_inbox_bounds();
    if (settings.contains("model") && settings["model"].contains("params")) {
      apply_tungsten_json(settings["model"]["params"], phys.params);
    }
    phys.declare_fields(m_state);
    auto &psi = m_state.get_field<double>("psi");
    apply_ics_from_json(settings, psi.domain(), psi.box(), psi.data(), psi.size());
    m_bc = parse_fixed_bc(settings);
    m_moving = parse_moving_bc(settings, comm);
    m_writers.configure(settings, m_domain, m_stack.fft().get_inbox_bounds(), comm,
                        rank);
    m_sys =
        std::make_unique<pfc::sim::SpectralMeanFieldETDSystem<TungstenPhysics<>>>(
            std::move(phys), m_stack.fft(), m_state, pfc::time::dt(m_time));
    m_ckpt.restore_from_config(m_state, m_time);
  }

  void run() {
    pfc::sim::SimulationDriver driver(m_time, &m_state);
    driver.run(
        [&](double t) {
          m_profile.timed_step(pfc::time::increment(m_time), m_stack.fft(),
                               [&] { m_sys->step(t); });
          m_ckpt.maybe_save(m_state, m_time);
        },
        [&](pfc::Time &) { apply_bcs(); }, [&](pfc::Time &) { apply_bcs(); },
        [&](const pfc::Time &tm) { m_writers.maybe_write(tm, psi().vec()); });
    m_profile.finalize();
  }

  void step_physics() { m_sys->step(pfc::time::current(m_time)); }

  [[nodiscard]] pfc::data::Field<double> &psi() {
    return m_state.get_field<double>("psi");
  }
  [[nodiscard]] const pfc::data::Field<double> &psi() const {
    return m_state.get_field<double>("psi");
  }
  [[nodiscard]] pfc::Time &time() noexcept { return m_time; }
  [[nodiscard]] const pfc::Time &time() const noexcept { return m_time; }
  [[nodiscard]] pfc::SimulationState &state() noexcept { return m_state; }
  [[nodiscard]] const pfc::SimulationState &state() const noexcept {
    return m_state;
  }
  [[nodiscard]] pfc::sim::stacks::SpectralCPUStack &stack() noexcept {
    return m_stack;
  }
  [[nodiscard]] pfc::fft::IFFTQueries &fft() noexcept { return m_stack.fft(); }
  [[nodiscard]] int dumps() const noexcept { return m_writers.dumps(); }

private:
  void apply_bcs() {
    auto &psi = m_state.get_field<double>("psi");
    if (m_bc) {
      tungsten::apply_fixed_bc(psi.domain(), psi.box(), psi.data(), *m_bc);
    }
    if (m_moving) {
      tungsten::apply_moving_bc(psi.domain(), psi.box(), psi.data(), *m_moving);
    }
    if (m_bc || m_moving) {
      psi.note_host_write();
    }
  }

  pfc::Domain m_domain{};
  pfc::Time m_time;
  pfc::sim::stacks::SpectralCPUStack m_stack;
  pfc::SimulationState m_state;
  pfc::sim::CheckpointService m_ckpt;
  std::optional<FixedBc> m_bc{};
  std::optional<MovingBc> m_moving{};
  TungstenETDWriters m_writers{};
  EtdProfileEnv m_profile;
  std::unique_ptr<pfc::sim::SpectralMeanFieldETDSystem<TungstenPhysics<>>> m_sys;
};

} // namespace tungsten
