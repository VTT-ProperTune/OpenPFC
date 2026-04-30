// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d.cpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — application driver.
 *
 * @details
 * Pure orchestration. Every reusable concern has been factored out:
 *
 * | Concern                              | Lives in |
 * |--------------------------------------|-------------------------------------------------------|
 * | Physics model (parameters + RHS)     | `heat3d/heat_model.hpp` | | Spectral
 * implicit-Euler propagator   | `heat3d/spectral_heat_propagator.hpp` | |
 * Command-line parsing                 | `heat3d/cli.hpp` | | Analytic reference +
 * L2 reporting    | `heat3d/reporting.hpp`                                | | FD
 * periodic CPU stack                | `pfc::sim::stacks::FdCpuStack` | | Spectral
 * CPU stack                   | `pfc::sim::stacks::SpectralCpuStack` | | Point-wise
 * gradient evaluators       | `pfc::field::FdGradient`, `SpectralGradient` | |
 * Per-cell driver loop                 | `pfc::sim::for_each_interior` | |
 * Explicit-Euler ODE stepper           | `pfc::sim::steppers::EulerStepper` | |
 * Halo-aware local field bundle        | `pfc::field::LocalField<T>` | | MPI timing
 * helper                    | `pfc::runtime::MpiTimer` + `tic`/`toc` | | MPI app
 * entry-point wrapper          | `pfc::runtime::mpi_main` | | Single-rank
 * CPU-affinity rescue      | `pfc::runtime::reset_cpu_affinity_if_single_mpi_rank` |
 *
 * Three solvers wire the same `HeatModel` to different OpenPFC pieces:
 *  - `fd`          — explicit Euler with even-order central FD Laplacian
 *                    (`pfc::field::FdGradient`) and a separated face-halo
 *                    exchange driven by `FdCpuStack::exchange_halos`.
 *  - `spectral`    — implicit Euler in Fourier space (2 FFTs/step) via
 *                    `heat3d::SpectralHeatPropagator`.
 *  - `spectral_pw` — explicit Euler with a point-wise RHS over materialized
 *                    second-derivative fields (`pfc::field::SpectralGradient`,
 *                    1 fwd + 3 inv FFTs/step). Same `HeatModel::rhs` as `fd`.
 */

#include <cstdlib>
#include <mpi.h>

#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/common/mpi_timer.hpp>

#include <heat3d/cli.hpp>
#include <heat3d/heat_model.hpp>
#include <heat3d/reporting.hpp>
#include <heat3d/spectral_heat_propagator.hpp>

using namespace pfc;
using heat3d::HeatModel;
using heat3d::Method;
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

void run_spectral_pointwise(const RunConfig &cfg, int rank, int nproc) {
  HeatModel model;
  model.D = cfg.D;

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
        const auto cfg = heat3d::parse_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) return EXIT_FAILURE;
        switch (cfg->method) {
        case Method::Fd: run_fd(*cfg, rank, nproc); break;
        case Method::Spectral: run_spectral(*cfg, rank, nproc); break;
        case Method::SpectralPointwise:
          run_spectral_pointwise(*cfg, rank, nproc);
          break;
        }
        return EXIT_SUCCESS;
      });
}
