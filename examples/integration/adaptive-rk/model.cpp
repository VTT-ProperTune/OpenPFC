// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <iostream>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/adaptive_controller.hpp>
#include <openpfc/kernel/simulation/steppers/embedded_rk.hpp>

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    const auto input = lesson::input(argc, argv);
    const double tolerance = input.value("tolerance", 1e-6);
    const double dt0 = input.value("dt", .05), end = input.value("end", .5);
    const double transition = input.value("transition", .15);
    lesson::positive(tolerance, "tolerance");
    lesson::positive(dt0, "dt");
    lesson::positive(end, "end");
    if (!(transition > 0 && transition < end) || end > 2)
      throw std::invalid_argument("Require 0 < transition < end <= 2");
    auto output =
        lesson::csv("attempts_rank_" + std::to_string(comm.rank()) + ".csv");
    output << "controller,attempt,t0,dt,accepted,metric,time,y,exact,rhs\n";
    for (bool pi : {false, true}) {
      pfc::sim::AdaptiveControlConfig policy;
      policy.mode = pfc::sim::AdaptiveControlMode::adaptive;
      policy.controller =
          pi ? pfc::sim::StepController::pi : pfc::sim::StepController::memoryless;
      policy.atol = policy.rtol = tolerance;
      policy.min_dt = 1e-10;
      policy.max_dt = .1;
      policy.max_sequential_rejections = 40;
      pfc::sim::AdaptiveTimeController controller(policy, 3);
      pfc::Time time({0, end, dt0}, 0);
      std::vector<double> y{1};
      double rate = 50;
      auto rhs = [&](double, const auto &u, auto &du) { du[0] = -rate * u[0]; };
      pfc::sim::steppers::EmbeddedRKStepper stepper(
          1, pfc::sim::steppers::make_embedded_rk23<double>(), rhs);
      int attempts = 0;
      // start snippet adaptive
      while (!time.done()) {
        if (++attempts > 10000) throw std::runtime_error("Attempt limit reached");
        const double t0 = time.get_accepted_time();
        double proposed = time.get_dt();
        if (t0 < transition) proposed = std::min(proposed, transition - t0);
        rate = t0 < transition ? 50 : 1; // One smooth phase per attempted interval.
        time.begin_attempt(proposed);
        const double dt = time.get_attempted_dt();
        const auto candidate = stepper.attempt(t0, dt, y);
        const auto decision = controller.decide_from_embedded_error(
            dt, stepper.error(), y, stepper.u_high(), comm);
        if (decision.accepted) pfc::sim::steppers::commit_step_attempt(y, candidate);
        controller.apply(time, decision);
        // end snippet adaptive
        const double t = time.get_accepted_time();
        const double exact =
            std::exp(-50 * std::min(t, transition) - std::max(0.0, t - transition));
        output << (pi ? "pi" : "memoryless") << ',' << attempts << ',' << t0 << ','
               << dt << ',' << decision.accepted << ',' << decision.metric << ','
               << t << ',' << y[0] << ',' << exact << ',' << stepper.last_rhs_evals()
               << '\n';
      }
      if (comm.rank() == 0)
        std::cout << (pi ? "pi" : "memoryless")
                  << " accepted=" << controller.accepted_count()
                  << " rejected=" << controller.rejected_count() << '\n';
    }
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
