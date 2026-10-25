// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <iostream>
#include <numbers>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/adaptive_controller.hpp>
#include <openpfc/kernel/simulation/etd_step_doubling.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

// start snippet physics
struct ReactionDiffusion {
  pfc::Domain domain;
  pfc::Box3i box;
  double cubic;
  struct Local {
    double cubic;
    double nonlinearity(const pfc::sim::SpectralCell &cell) const {
      return -cubic * cell.psi * cell.psi * cell.psi;
    }
  };
  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<double>(state, "psi", domain, box, 0);
  }
  double linear_symbol(double laplacian) const { return laplacian; }
  Local pointwise() const { return {cubic}; }
};
// end snippet physics

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    const auto input = lesson::input(argc, argv);
    const int n = input.value("n", 16);
    const double tolerance = input.value("tolerance", 1e-6);
    const double dt0 = input.value("dt", .04), end = input.value("end", .2);
    lesson::positive(tolerance, "tolerance");
    lesson::positive(dt0, "dt");
    lesson::positive(end, "end");
    if (n < 8 || n > 64 || n % 2 || end > 2)
      throw std::invalid_argument("Use even n in [8,64] and end <= 2");
    const auto domain =
        pfc::domain::create(pfc::GridSize{{n, 4, 4}}, pfc::PhysicalOrigin{{0, 0, 0}},
                            pfc::GridSpacing{{2 * std::numbers::pi / n, 1, 1}});
    pfc::sim::stacks::SpectralCPUStack stack(domain, comm.rank(), comm.size());
    auto attempts =
        lesson::csv("attempts_rank_" + std::to_string(comm.rank()) + ".csv");
    auto fields = lesson::csv("fields_rank_" + std::to_string(comm.rank()) + ".csv");
    attempts << "case,attempt,t0,dt,accepted,metric,time,rollback_error,transforms,"
                "nonlinear\n";
    fields << "case,i,j,k,x,u\n";
    for (const std::string name : {"heat", "uniform", "reaction"}) {
      ReactionDiffusion physics{domain, stack.u().box(), name == "heat" ? 0.0 : 1.0};
      pfc::SimulationState state;
      physics.declare_fields(state);
      auto &u = state.get_field<double>("psi");
      u.apply([&](double x, double, double) {
        return name == "uniform" ? .4 : .4 * std::cos(x);
      });
      pfc::sim::SpectralETDSystem system(physics, stack.fft(), state, dt0);
      pfc::sim::ETD1StepDoubling doubling(system);
      pfc::sim::AdaptiveControlConfig policy;
      policy.mode = pfc::sim::AdaptiveControlMode::adaptive;
      policy.atol = policy.rtol = tolerance;
      policy.min_dt = 1e-10;
      policy.max_dt = .1;
      policy.max_sequential_rejections = 40;
      pfc::sim::AdaptiveTimeController controller(policy, 2);
      pfc::Time time({0, end, dt0}, 0);
      int count = 0;
      // start snippet doubling
      while (!time.done()) {
        if (++count > 10000) throw std::runtime_error("Attempt limit reached");
        const double t0 = time.get_accepted_time();
        time.begin_attempt(time.get_dt());
        const double dt = time.get_attempted_dt();
        doubling.attempt(t0, dt);
        const auto decision = controller.decide_from_embedded_error(
            dt, doubling.raw_difference(), doubling.saved_state(),
            doubling.half_steps(), comm);
        double rollback = 0;
        if (decision.accepted) {
          doubling.commit();
        } else {
          doubling.reject();
          for (std::size_t i = 0; i < u.vec().size(); ++i)
            rollback =
                std::max(rollback, std::abs(u.vec()[i] - doubling.saved_state()[i]));
        }
        controller.apply(time, decision);
        // end snippet doubling
        attempts << name << ',' << count << ',' << t0 << ',' << dt << ','
                 << decision.accepted << ',' << decision.metric << ','
                 << time.get_current() << ',' << rollback << ','
                 << doubling.transform_count() << ',' << doubling.nonlinear_count()
                 << '\n';
      }
      u.for_each_owned([&](int i, int j, int k) {
        const auto low = u.box().low;
        const double x = (i + low[0]) * domain.spacing[0];
        fields << name << ',' << i + low[0] << ',' << j + low[1] << ',' << k + low[2]
               << ',' << x << ',' << u(i, j, k) << '\n';
      });
    }
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
