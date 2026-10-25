// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/simulation/adaptive_controller.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/embedded_rk.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>
#include <openpfc/kernel/simulation/time.hpp>

/** \example 21_adaptive_stepping.cpp
 *
 * Adaptive Bogacki–Shampine 3(2) on the scalar ODE
 * \f$y' = -k(t)\, y\f$ with a stiff transient (\f$k=50\f$ for \f$t<0.15\f$,
 * then \f$k=1\f$). The exact value at \f$t=0.5\f$ is \f$\exp(-7.85)\f$.
 *
 * The run compares the memoryless controller with the PI.3.4 controller on
 * the same tolerance. The embedded error is scaled by
 * \f$\max(|y|, |y_{\mathrm{candidate}}|)\f$. The printed step counts are a
 * comparison on this ODE, not a claim that PI is better in general.
 *
 * Run: `mpirun -np 1 ./21_adaptive_stepping`
 */

namespace {

struct PiecewiseStiffRhs {
  void operator()(double t, std::vector<double> &u, std::vector<double> &du) const {
    const double k = (t < 0.15) ? 50.0 : 1.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = -k * u[i];
    }
  }
};

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  using pfc::Time;
  using pfc::sim::AdaptiveControlConfig;
  using pfc::sim::AdaptiveControlMode;
  using pfc::sim::AdaptiveTimeController;
  using pfc::sim::StepController;
  using pfc::sim::steppers::commit_step_attempt;
  using pfc::sim::steppers::EmbeddedRKStepper;
  using pfc::sim::steppers::make_embedded_rk23;

  const auto run = [](StepController kind, const char *name) {
    AdaptiveControlConfig cfg;
    cfg.mode = AdaptiveControlMode::adaptive;
    cfg.controller = kind;
    cfg.atol = 1e-5;
    cfg.rtol = 1e-5;
    cfg.safety_factor = 0.9;
    cfg.growth_max = 2.0;
    cfg.shrink_max = 0.5;
    cfg.min_dt = 1e-6;
    cfg.max_dt = 0.1;
    cfg.max_sequential_rejections = 30;

    AdaptiveTimeController controller(cfg, /*error_order=*/3);
    EmbeddedRKStepper stepper(1, make_embedded_rk23<double>(), PiecewiseStiffRhs{});
    Time time({0.0, 0.5, 0.05}, 0.0);
    std::vector<double> u{1.0};
    std::vector<double> accepted_dt;
    unsigned rhs = 0;
    int steps = 0;
    while (!time.done() && steps < 800) {
      time.begin_attempt(time.get_dt());
      const double dt = time.get_attempted_dt();
      const auto attempt = stepper.attempt(time.get_accepted_time(), dt, u);
      rhs += stepper.last_rhs_evals();
      const auto decision = controller.decide_from_embedded_error(
          dt, stepper.error(), u, stepper.u_high());
      if (decision.accepted) {
        commit_step_attempt(u, attempt);
        accepted_dt.push_back(dt);
      }
      controller.apply(time, decision);
      ++steps;
    }

    const double exact = std::exp(-7.85);
    std::cout << name << " t=" << time.get_accepted_time() << " y=" << u[0]
              << " error=" << std::abs(u[0] - exact)
              << " accepted=" << controller.accepted_count()
              << " rejected=" << controller.rejected_count() << " rhs=" << rhs
              << "\n";
    std::cout << name << " accepted_dt";
    for (double h : accepted_dt) {
      std::cout << ' ' << h;
    }
    std::cout << '\n';
    return time.done();
  };

  const bool memoryless_done = run(StepController::memoryless, "memoryless");
  const bool pi_done = run(StepController::pi, "pi");

  MPI_Finalize();
  return (memoryless_done && pi_done) ? 0 : 1;
}
