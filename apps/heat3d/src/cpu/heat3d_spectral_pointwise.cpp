// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_spectral_pointwise.cpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — explicit
 *        Euler with point-wise RHS over materialized spectral derivatives.
 *
 * @details
 * Per-method binary in the heat3d quartet
 * (`heat3d_fd`, `heat3d_fd_manual`, `heat3d_spectral`,
 * `heat3d_spectral_pointwise`). This driver uses
 * `pfc::field::SpectralGradient<heat3d::HeatGrads>` to materialize
 * the second derivatives (1 fwd + 3 inv FFTs/step) and then drives an
 * explicit Euler step with the **same** `HeatModel::rhs(t, HeatGrads&)`
 * that the FD path uses — same physics, different gradient evaluator.
 */

#include <cstdlib>
#include <mpi.h>

#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/common/mpi_timer.hpp>

#include <heat3d/cli.hpp>
#include <heat3d/heat_model.hpp>
#include <heat3d/reporting.hpp>

using namespace pfc;
using heat3d::HeatModel;
using heat3d::RunConfig;

namespace {

void run_spectral_pointwise(const RunConfig &cfg, int rank, int nproc) {
  HeatModel model;

  sim::stacks::SpectralCpuStack stack(
      GridSize({cfg.N, cfg.N, cfg.N}), PhysicalOrigin({0.0, 0.0, 0.0}),
      GridSpacing({1.0, 1.0, 1.0}), rank, nproc, MPI_COMM_WORLD);
  stack.u().apply(model.initial_condition);

  auto grad = field::create<heat3d::HeatGrads>(stack.u(), stack.fft());
  auto stepper = sim::steppers::create(stack.u(), grad, model, cfg.dt);

  runtime::MpiTimer timer{MPI_COMM_WORLD};
  runtime::tic(timer);
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step)
    t = stepper.step(t, stack.u().vec());
  const double max_elapsed = runtime::toc(timer);

  heat3d::report(rank, nproc, cfg, "spectral_pw", "", max_elapsed,
                 "(point-wise spectral RHS, explicit Euler)",
                 [&stack](auto &&cb) { stack.u().for_each_owned(cb); });
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg =
            heat3d::parse_spectral_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) return EXIT_FAILURE;
        run_spectral_pointwise(*cfg, rank, nproc);
        return EXIT_SUCCESS;
      });
}
