// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file tungsten_etd_session.hpp
 * @brief JSON-driven CPU session: stack + TungstenPhysics + mean-field ETD.
 *
 * @details
 * M8 A/B driver. Gen-1 `tungsten` (`App<Tungsten>`) stays. This session owns
 * `SpectralCpuStack`, `SimulationState`, and `SpectralMeanFieldEtdSystem` —
 * no model-owned FFT. Initial conditions and fixed BCs are applied on the
 * `Field` (same formulas as Gen-1 `Constant` / `SingleSeed` / `FixedBC`).
 * Result writers are not wired yet.
 */

#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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
#include <tungsten/tungsten_physics.hpp>

namespace tungsten {

class TungstenEtdSession {
public:
  TungstenEtdSession(const TungstenEtdSession &) = delete;
  TungstenEtdSession &operator=(const TungstenEtdSession &) = delete;
  TungstenEtdSession(TungstenEtdSession &&) = delete;
  TungstenEtdSession &operator=(TungstenEtdSession &&) = delete;

  TungstenEtdSession(const nlohmann::json &settings, int rank, int nproc,
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
    apply_initial_conditions(settings, m_state.get_field<double>("psi"));
    parse_fixed_bc(settings);
    m_sys = std::make_unique<
        pfc::sim::SpectralMeanFieldEtdSystem<TungstenPhysics<>>>(
        std::move(phys), m_stack.fft(), m_state, pfc::time::dt(m_time));
  }

  void run() {
    while (!pfc::time::done(m_time)) {
      if (pfc::time::increment(m_time) == 0) {
        apply_fixed_bc();
      }
      pfc::time::next(m_time);
      apply_fixed_bc();
      m_sys->step(pfc::time::current(m_time));
    }
  }

  [[nodiscard]] pfc::data::Field<double> &psi() {
    return m_state.get_field<double>("psi");
  }
  [[nodiscard]] const pfc::data::Field<double> &psi() const {
    return m_state.get_field<double>("psi");
  }
  [[nodiscard]] const pfc::Time &time() const noexcept { return m_time; }
  [[nodiscard]] pfc::sim::stacks::SpectralCpuStack &stack() noexcept {
    return m_stack;
  }

private:
  static void apply_initial_conditions(const nlohmann::json &settings,
                                       pfc::data::Field<double> &psi) {
    if (!settings.contains("initial_conditions") ||
        !settings["initial_conditions"].is_array()) {
      return;
    }
    for (const auto &ic : settings["initial_conditions"]) {
      const std::string type = ic.value("type", std::string{});
      if (type == "constant") {
        const double n0 = ic.at("n0").get<double>();
        psi.apply([n0](double, double, double) { return n0; });
      } else if (type == "single_seed") {
        const double amp = ic.at("amp_eq").get<double>();
        const double rho = ic.at("rho_seed").get<double>();
        apply_single_seed(psi, amp, rho);
      } else {
        throw std::invalid_argument(
            "TungstenEtdSession: unsupported initial condition type '" + type +
            "'");
      }
    }
  }

  static void apply_single_seed(pfc::data::Field<double> &psi, double amp,
                                double rho) {
    const double s = 1.0 / std::sqrt(2.0);
    const double q[6][3] = {{s, s, 0}, {s, 0, s}, {0, s, s},
                            {s, 0, -s}, {s, -s, 0}, {0, s, -s}};
    const double r2 = 64.0 * 64.0;
    psi.apply([=](double x, double y, double z) {
      if (x * x + y * y + z * z >= r2) {
        return 0.0;
      }
      double u = rho;
      for (int qi = 0; qi < 6; ++qi) {
        u += 2.0 * amp * std::cos(q[qi][0] * x + q[qi][1] * y + q[qi][2] * z);
      }
      return u;
    });
  }

  void parse_fixed_bc(const nlohmann::json &settings) {
    if (!settings.contains("boundary_conditions") ||
        !settings["boundary_conditions"].is_array()) {
      return;
    }
    for (const auto &bc : settings["boundary_conditions"]) {
      if (bc.value("type", std::string{}) != "fixed") {
        throw std::invalid_argument(
            "TungstenEtdSession: unsupported boundary condition type '" +
            bc.value("type", std::string{}) + "'");
      }
      m_rho_low = bc.at("rho_low").get<double>();
      m_rho_high = bc.at("rho_high").get<double>();
    }
  }

  void apply_fixed_bc() {
    if (!m_rho_low || !m_rho_high) {
      return;
    }
    auto &psi = m_state.get_field<double>("psi");
    const auto n = pfc::domain::get_size(psi.domain());
    const auto dx = pfc::domain::get_spacing(psi.domain());
    const double xwidth = 20.0;
    const double alpha = 1.0;
    const double xpos =
        static_cast<double>(n[0]) * dx[0] - xwidth;
    const auto local = psi.local_size();
    const int nx = local[0];
    const int ny = local[1];
    const int nz = local[2];
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const double x = psi.coords(i, j, k)[0];
          if (std::abs(x - xpos) < xwidth) {
            const double S = 1.0 / (1.0 + std::exp(-alpha * (x - xpos)));
            psi(i, j, k) = (*m_rho_low * S) + (*m_rho_high * (1.0 - S));
          }
        }
      }
    }
    psi.note_host_write();
  }

  pfc::Domain m_domain{};
  pfc::Time m_time;
  pfc::sim::stacks::SpectralCpuStack m_stack;
  pfc::SimulationState m_state;
  std::optional<double> m_rho_low{};
  std::optional<double> m_rho_high{};
  std::unique_ptr<pfc::sim::SpectralMeanFieldEtdSystem<TungstenPhysics<>>>
      m_sys;
};

} // namespace tungsten
