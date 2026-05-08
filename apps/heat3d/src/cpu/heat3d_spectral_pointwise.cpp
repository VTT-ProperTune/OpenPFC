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
 * `heat3d_spectral_pointwise`). This driver is the **spectral twin** of
 * `heat3d_fd`: the user-facing time loop is identical save for the stack
 * type, and `pfc::field::SpectralGradient<HeatGrads>` (1 fwd + 3 inv FFTs
 * per step) replaces `FdGradient<HeatGrads>`. The same `HeatModel::rhs`
 * is applied cell-by-cell through `pfc::sim::DuField<G, ...>`:
 *
 *     auto& u  = stack.u();
 *     auto  du = stack.du<HeatGrads>();   // spectral evaluator wired in
 *
 *     u.apply(model.initial_condition);
 *
 *     for (int step = 0; step < cfg.n_steps; ++step) {
 *       du.apply([](const HeatGrads& g) { return kD * (g.xx + g.yy + g.zz); });
 *       u += cfg.dt * du;                 // explicit Euler, on the page
 *       t  += cfg.dt;
 *     }
 *
 * For the implicit Euler in Fourier space (1 fwd + 1 inv FFT, diagonal
 * multiplier), see `heat3d_spectral`.
 */

#include <cstdlib>
#include <mpi.h>

#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/common/mpi_timer.hpp>

#include <heat3d/cli.hpp>
#include <heat3d/heat_model.hpp>
#include <heat3d/reporting.hpp>

using namespace pfc;
using heat3d::HeatGrads;
using heat3d::HeatModel;
using heat3d::RunConfig;

namespace {

void run_spectral_pointwise(const RunConfig &cfg, int rank, int nproc) {
  HeatModel model;

  sim::stacks::SpectralCpuStack stack(
      GridSize({cfg.N, cfg.N, cfg.N}), PhysicalOrigin({0.0, 0.0, 0.0}),
      GridSpacing({1.0, 1.0, 1.0}), rank, nproc, MPI_COMM_WORLD);

  auto &u = stack.u();
  auto du = stack.du<HeatGrads>();

  u.apply(model.initial_condition);

  runtime::MpiTimer timer{MPI_COMM_WORLD};
  runtime::tic(timer);
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step) {
    du.apply([](const HeatGrads &g) { return heat3d::kD * (g.xx + g.yy + g.zz); });
    u += cfg.dt * du;
    t += cfg.dt;
  }
  const double max_elapsed = runtime::toc(timer);
  (void)t;

  heat3d::report(rank, nproc, cfg, "spectral_pw", "", max_elapsed,
                 "(point-wise spectral RHS, explicit Euler)",
                 [&u](auto &&cb) { u.for_each_owned(cb); });
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
