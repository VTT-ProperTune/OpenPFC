// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_spectral.cpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — implicit
 *        Euler in Fourier space.
 *
 * @details
 * Per-method binary in the heat3d quartet
 * (`heat3d_fd`, `heat3d_fd_manual`, `heat3d_spectral`,
 * `heat3d_spectral_pointwise`). The spectral propagator
 * (`heat3d::SpectralHeatPropagator`) takes 2 FFTs/step (forward,
 * apply diagonal multiplier in k-space, inverse). No FD halo, no
 * stencil — the laboratory-style variant lives in `heat3d_fd_manual`.
 */

#include <cstdlib>
#include <mpi.h>

#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/common/mpi_timer.hpp>

#include <heat3d/cli.hpp>
#include <heat3d/heat_model.hpp>
#include <heat3d/reporting.hpp>
#include <heat3d/spectral_heat_propagator.hpp>

using namespace pfc;
using heat3d::HeatModel;
using heat3d::RunConfig;

namespace {

void run_spectral(const RunConfig &cfg, int rank, int nproc) {
  HeatModel model;
  model.D = cfg.D;

  sim::stacks::SpectralCpuStack stack(
      GridSize({cfg.N, cfg.N, cfg.N}), PhysicalOrigin({0.0, 0.0, 0.0}),
      GridSpacing({1.0, 1.0, 1.0}), rank, nproc, MPI_COMM_WORLD);
  stack.u().apply(model.initial_condition);

  heat3d::SpectralHeatPropagator prop(stack.fft(), stack.u(), model.D, cfg.dt);

  runtime::MpiTimer timer{MPI_COMM_WORLD};
  runtime::tic(timer);
  for (int step = 0; step < cfg.n_steps; ++step) prop.step(stack.u());
  const double max_elapsed = runtime::toc(timer);

  heat3d::report(rank, nproc, cfg, "spectral", "", max_elapsed,
                 "(periodic spectral vs infinite-domain reference)",
                 [&stack](auto &&cb) { stack.u().for_each_owned(cb); });
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg =
            heat3d::parse_spectral_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) return EXIT_FAILURE;
        run_spectral(*cfg, rank, nproc);
        return EXIT_SUCCESS;
      });
}
