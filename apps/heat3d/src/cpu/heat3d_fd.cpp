// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_fd.cpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — compact
 *        finite-difference driver.
 *
 * @details
 * Per-method binary in the heat3d quartet
 * (`heat3d_fd`, `heat3d_fd_manual`, `heat3d_spectral`,
 * `heat3d_spectral_pointwise`). This one is the **compact** FD driver:
 * the time loop is three lines that compose `pfc::sim::stacks::FdCpuStack`
 * with `pfc::field::FdGradient<heat3d::HeatGrads>` and
 * `pfc::sim::steppers::EulerStepper`. Use `heat3d_fd_manual` for the
 * laboratory-style equivalent that exposes the stencil and comm overlap.
 *
 * Layout, halo policy and FD stencils are all hidden inside `FdCpuStack`
 * and the kernel `field/fd_apply.hpp` primitives — see
 * `apps/heat3d/README.md` for the side-by-side comparison.
 */

#include <cstdlib>
#include <mpi.h>

#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
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

void run_fd(const RunConfig &cfg, int rank, int nproc) {
  HeatModel model;
  model.D = cfg.D;

  sim::stacks::FdCpuStack stack(
      GridSize({cfg.N, cfg.N, cfg.N}), PhysicalOrigin({0.0, 0.0, 0.0}),
      GridSpacing({1.0, 1.0, 1.0}), cfg.fd_order, rank, nproc, MPI_COMM_WORLD);
  stack.u().apply(model.initial_condition);

  auto grad = field::create<heat3d::HeatGrads>(stack.u(), cfg.fd_order);
  auto stepper = sim::steppers::create(stack.u(), grad, model, cfg.dt);

  runtime::MpiTimer timer{MPI_COMM_WORLD};
  runtime::tic(timer);
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step) {
    stack.exchange_halos();
    t = stepper.step(t, stack.u().vec());
  }
  const double max_elapsed = runtime::toc(timer);

  heat3d::report(rank, nproc, cfg, "fd", heat3d::fd_extra_metadata(cfg), max_elapsed,
                 "(periodic domain; error dominated by boundaries for localized IC)",
                 [&stack](auto &&cb) { stack.u().for_each_interior(cb); });
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg = heat3d::parse_fd_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) return EXIT_FAILURE;
        run_fd(*cfg, rank, nproc);
        return EXIT_SUCCESS;
      });
}
