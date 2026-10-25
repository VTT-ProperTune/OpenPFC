// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/adaptive_controller.hpp>
#include <openpfc/kernel/simulation/etd_step_doubling.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>
#include <openpfc/kernel/simulation/time.hpp>

/** \example 24_adaptive_etd.cpp
 *
 * ETD1 step doubling. A pure heat mode is checked against
 * \f$\exp(-k^2 t)\f$. The same driver then integrates
 * \f$\partial_t \psi = \nabla^2 \psi - \psi^3\f$ with the memoryless
 * controller. Printed step counts are one trajectory, not a claim that
 * the adaptive step is faster.
 *
 * Run: `mpirun -np 1 ./24_adaptive_etd`
 */

namespace {

struct Heat {
  pfc::Domain domain{};
  pfc::Box3i box{};
  struct Zero {
    double nonlinearity(const pfc::sim::SpectralCell &) const { return 0.0; }
  };
  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<double>(state, "psi", domain, box, 0);
  }
  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    return k_laplacian;
  }
  [[nodiscard]] Zero pointwise() const { return {}; }
};

struct CubicReaction {
  pfc::Domain domain{};
  pfc::Box3i box{};
  struct Cubic {
    double nonlinearity(const pfc::sim::SpectralCell &cell) const {
      return -cell.psi * cell.psi * cell.psi;
    }
  };
  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<double>(state, "psi", domain, box, 0);
  }
  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    return k_laplacian;
  }
  [[nodiscard]] Cubic pointwise() const { return {}; }
};

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  constexpr int n = 8;
  const auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  const double k2 = std::pow(2.0 * pfc::pi / static_cast<double>(n), 2);

  Heat heat;
  heat.domain = domain;
  heat.box = fft.get_inbox_bounds();
  pfc::SimulationState heat_state;
  heat.declare_fields(heat_state);
  constexpr double heat_dt = 0.05;
  pfc::sim::SpectralETDSystem<Heat> heat_sys(heat, fft, heat_state, heat_dt);
  auto &heat_psi = heat_state.get_field<double>("psi");
  heat_psi.apply([&](double x, double, double) {
    return std::cos(2.0 * pfc::pi * x / static_cast<double>(n));
  });
  const std::vector<double> heat0 = heat_psi.vec();
  pfc::sim::ETD1StepDoubling<Heat> heat_doubling(heat_sys);
  double heat_t = 0.0;
  double heat_diff = 0.0;
  for (int step = 0; step < 8; ++step) {
    heat_doubling.attempt(heat_t, heat_dt);
    for (double v : heat_doubling.raw_difference()) {
      heat_diff = std::max(heat_diff, std::abs(v));
    }
    heat_doubling.commit();
    heat_t += heat_dt;
  }
  const double heat_factor = std::exp(-k2 * heat_t);
  double heat_error = 0.0;
  for (std::size_t i = 0; i < heat_psi.vec().size(); ++i) {
    heat_error =
        std::max(heat_error, std::abs(heat_psi.vec()[i] - heat_factor * heat0[i]));
  }
  std::cout << "linear t=" << heat_t << " error=" << heat_error
            << " max_raw_difference=" << heat_diff << "\n";

  CubicReaction physics;
  physics.domain = domain;
  physics.box = fft.get_inbox_bounds();
  pfc::SimulationState state;
  physics.declare_fields(state);
  constexpr double dt0 = 0.01;
  pfc::sim::SpectralETDSystem<CubicReaction> sys(physics, fft, state, dt0);
  state.get_field<double>("psi").apply([&](double x, double, double) {
    return 0.2 * std::cos(2.0 * pfc::pi * x / static_cast<double>(n));
  });

  pfc::sim::AdaptiveControlConfig cfg;
  cfg.mode = pfc::sim::AdaptiveControlMode::adaptive;
  cfg.atol = 1e-8;
  cfg.rtol = 1e-3;
  cfg.min_dt = 1e-6;
  cfg.max_dt = 0.05;
  cfg.max_sequential_rejections = 30;
  pfc::sim::AdaptiveTimeController controller(cfg, /*error_order=*/2);
  pfc::Time time({0.0, 0.1, dt0}, 0.0);
  pfc::sim::ETD1StepDoubling<CubicReaction> doubling(sys);

  int nonlinear = 0;
  int transforms = 0;
  std::vector<double> accepted_dt;
  const auto t0 = std::chrono::steady_clock::now();
  int guard = 0;
  while (!time.done() && guard < 2000) {
    time.begin_attempt(time.get_dt());
    const double dt = time.get_attempted_dt();
    doubling.attempt(time.get_accepted_time(), dt);
    const auto decision = controller.decide_from_embedded_error(
        dt, doubling.raw_difference(), doubling.saved_state(),
        doubling.half_steps());
    if (decision.accepted) {
      doubling.commit();
      accepted_dt.push_back(dt);
    } else {
      doubling.reject();
    }
    nonlinear += doubling.nonlinear_count();
    transforms += doubling.transform_count();
    controller.apply(time, decision);
    ++guard;
  }
  const double wall_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
  std::cout << "adaptive t=" << time.get_accepted_time()
            << " accepted=" << controller.accepted_count()
            << " rejected=" << controller.rejected_count()
            << " nonlinear=" << nonlinear << " transforms=" << transforms
            << " wall_ms=" << wall_ms << "\n";
  std::cout << "adaptive dt";
  for (double h : accepted_dt) {
    std::cout << ' ' << h;
  }
  std::cout << "\n";

  MPI_Finalize();
  const bool ok = heat_error < 1e-8 && heat_diff < 1e-8 && time.done();
  return ok ? 0 : 1;
}
