// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file json_fd_session.hpp
 * @brief JSON → `SimulationSession<FDCPUStack>` + composed RK stepper (M10).
 *
 * @details
 * FD session wiring: `method`/`backend`/`fd_order` plus Time build an
 * `FDCPUStack`. `timestepping.integrator.method` (and optional
 * `simulator.integrator.method` overlay) selects a registered RK composer.
 * IMEX/ETD tokens fail closed — those need `compose_imex_euler` /
 * `compose_etd1`. The default RHS is periodic heat, \(\partial_t u =
 * \Delta u\), via `stack.du()`.
 */

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_integrator_method.hpp>
#include <openpfc/frontend/ui/from_json_simulation_session.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/integrator_method.hpp>
#include <openpfc/kernel/simulation/steppers/method_composition.hpp>

namespace pfc::ui {

struct FdJsonHeatGrads {
  double xx{};
  double yy{};
  double zz{};
};

inline void overlay_simulator_integrator_method(pfc::Time &time,
                                                const nlohmann::json &settings) {
  if (!settings.contains("simulator") || !settings["simulator"].is_object()) {
    return;
  }
  const auto &j = settings["simulator"];
  if (j.contains("integrator") && j["integrator"].is_object() &&
      j["integrator"].contains("method")) {
    time.set_method(from_json<pfc::sim::steppers::RKIntegratorMethod>(
        j["integrator"]["method"]));
  }
}

/// Fill a periodic sine along x (Laplacian eigenmode) on an FD CPU stack field.
inline void fill_fd_sine_x(pfc::data::Field<double> &u) {
  const int nx = u.global_size()[0];
  const double k = 2.0 * std::numbers::pi / static_cast<double>(nx);
  u.apply([k](double x, double, double) { return std::sin(k * x); });
}

/**
 * @brief Compose RK from `session.time().method()` and step `stack.u()`.
 * @return Number of physics steps taken.
 */
inline int step_fd_cpu_session(
    pfc::sim::SimulationSession<pfc::sim::stacks::FDCPUStack> &session) {
  const auto method = session.time().method();
  if (!pfc::sim::steppers::is_runge_kutta(method)) {
    throw std::invalid_argument(
        std::string("JSON FD CPU session requires an RK integrator; \"") +
        pfc::sim::steppers::to_string(method) +
        "\" needs compose_etd1 / compose_imex_euler");
  }

  auto &stack = session.stack();
  auto &u = stack.u();
  auto du = stack.du<FdJsonHeatGrads>();
  auto rhs = [&u, &du](double t, std::vector<double> &field,
                       std::vector<double> &d) {
    auto &uv = u.vec();
    if (std::addressof(field) != std::addressof(uv)) {
      uv = field;
    }
    du.apply([](const FdJsonHeatGrads &g) { return g.xx + g.yy + g.zz; }, t);
    if (d.size() != du.size()) {
      d.resize(du.size());
    }
    std::copy(du.data(), du.data() + du.size(), d.begin());
  };

  pfc::sim::steppers::IntegratorComposeConfig cfg{
      .dt = pfc::time::dt(session.time()), .requires_adaptive = false};
  auto composition =
      pfc::sim::steppers::compose_scalar(method, cfg, u.size(), std::move(rhs));
  int steps = 0;
  session.run([&](double t) {
    composition.stepper.step(t, u);
    ++steps;
  });
  return steps;
}

/**
 * @brief JSON → FD CPU session, sine IC, composed RK heat step.
 * @return Number of physics steps taken.
 */
inline int run_fd_cpu_json_session(const nlohmann::json &settings, int rank,
                                   int nproc, MPI_Comm comm = MPI_COMM_WORLD) {
  auto session = make_simulation_session<pfc::sim::stacks::FDCPUStack>(
      settings, rank, nproc, comm);
  overlay_simulator_integrator_method(session.time(), settings);
  fill_fd_sine_x(session.stack().u());
  return step_fd_cpu_session(session);
}

} // namespace pfc::ui
