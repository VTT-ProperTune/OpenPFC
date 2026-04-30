// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_fd_manual.cpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — laboratory-style
 *        finite-difference driver.
 *
 * @details
 * This is the **"laboratory, not fortress"** counterpart to `heat3d_fd`.
 * Where `heat3d_fd` composes `FdCpuStack + FdGradient + EulerStepper`
 * in three lines and hides the physics inside the kernel, this driver
 * exposes the stencil, the comm/compute overlap, and the explicit
 * Euler update directly in `main`. Only the cumbersome plumbing —
 * decomposition, MPI face exchange, and linear-index arithmetic —
 * stays hidden behind:
 *
 *  - `pfc::field::PaddedBrick<double>` — single contiguous buffer with
 *    `u(i, j, k)` valid for `i,j,k in [-hw, n+hw)`. No edge overwrite,
 *    no separate face vectors.
 *  - `pfc::PaddedHaloExchanger<double>` — non-blocking
 *    `start_halo_exchange()` / `finish_halo_exchange()` pair on the
 *    same buffer.
 *  - `pfc::field::for_each_inner / for_each_border / for_each_owned` —
 *    `(int i, int j, int k)` triple-yielding lambda iterators.
 *  - `pfc::runtime::tic(timer, "label") / toc(timer, "label")` —
 *    collective-free per-section timers; `print_timing_summary` does
 *    one allreduce-max at the end to report the slowest rank.
 *
 * The hot loop literally reads:
 *
 *     halo.start_halo_exchange(u.data(), u.size());
 *     for_each_inner(u, hw, [&](int i, int j, int k) { ... stencil ... });
 *     halo.finish_halo_exchange();
 *     for_each_border(u, hw, [&](int i, int j, int k) { ... same stencil ... });
 *     for_each_owned(u, [&](int i, int j, int k) {
 *       u(i, j, k) += cfg.dt * du(i, j, k);
 *     });
 *
 * The stencil is the textbook second-order central seven-point
 * Laplacian (halo width 1). `HeatModel::rhs` is reused unchanged so
 * the physics still lives in the model; the driver just *spells out*
 * how the model is wired to the data.
 */

#include <cstdlib>
#include <mpi.h>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/common/mpi_timer.hpp>

#include <heat3d/cli.hpp>
#include <heat3d/heat_grads.hpp>
#include <heat3d/heat_model.hpp>
#include <heat3d/reporting.hpp>

using namespace pfc;
using heat3d::HeatGrads;
using heat3d::HeatModel;
using heat3d::RunConfig;

namespace {

void run_fd_manual(const RunConfig &cfg, int rank, int nproc) {
  // 1. Physics: initial condition + RHS live in HeatModel; the diffusion
  //    coefficient itself is hard-pinned via `heat3d::kD` in heat_model.hpp.
  HeatModel model;

  // 2. Hidden plumbing: world geometry + decomposition. The single
  //    `decomposition::create(world, nproc)` call auto-picks the
  //    rank grid; the manual driver does not need to spell it out.
  const auto world =
      world::create(GridSize({cfg.N, cfg.N, cfg.N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                    GridSpacing({1.0, 1.0, 1.0}));
  const auto decomp = decomposition::create(world, nproc);

  // 3. Two halo-padded buffers: `u` (state) and `du` (RHS). Both
  //    cover the local owned core plus a 1-cell ghost ring on every
  //    side, all in one contiguous `std::vector<double>`.
  const int hw = 1; // second-order central Laplacian -> stencil radius 1
  field::PaddedBrick<double> u(decomp, rank, hw);
  field::PaddedBrick<double> du(decomp, rank, hw);

  // 4. Hidden plumbing: in-place non-blocking halo exchanger.
  PaddedHaloExchanger<double> halo(decomp, rank, hw, MPI_COMM_WORLD);

  // 5. Initial condition: physicist-friendly `(x, y, z) -> u(x, y, z)`,
  //    fills only the owned core. `apply` does the index loop for us.
  u.apply(model.initial_condition);

  // 6. Per-cell stencil (textbook 2nd-order central 7-point Laplacian).
  //    Pulled into a lambda so the inner-region and border loops can
  //    share the same code (and so the reader can see the physics
  //    once, in one place).
  auto stencil_step = [&](int i, int j, int k) {
    HeatGrads g{};
    g.xx = u(i + 1, j, k) - 2.0 * u(i, j, k) + u(i - 1, j, k);
    g.yy = u(i, j + 1, k) - 2.0 * u(i, j, k) + u(i, j - 1, k);
    g.zz = u(i, j, k + 1) - 2.0 * u(i, j, k) + u(i, j, k - 1);
    du(i, j, k) = model.rhs(0.0, g);
  };

  // 7. Time loop. Top-level timer brackets the loop; per-section
  //    timers break each step into named slices reported at the end.
  runtime::MpiTimer timer{MPI_COMM_WORLD};
  runtime::tic(timer);
  for (int step = 0; step < cfg.n_steps; ++step) {

    // Start non-blocking halo exchange — overlaps with inner work.
    halo.start_halo_exchange(u.data(), u.size());

    // Inner cells: stencil only reads owned cells, no halo dependency.
    runtime::tic(timer, "inner");
    field::for_each_inner_omp(u, hw, stencil_step);
    runtime::toc(timer, "inner");

    // Wait for neighbour data to land in the halo ring.
    runtime::tic(timer, "halo_wait");
    halo.finish_halo_exchange();
    runtime::toc(timer, "halo_wait");

    // Border cells: same stencil, now safely reaches into the halo.
    runtime::tic(timer, "border");
    field::for_each_border(u, hw, stencil_step);
    runtime::toc(timer, "border");

    // Explicit Euler over the full owned region: u <- u + dt * du.
    runtime::tic(timer, "euler");
    field::for_each_owned_omp(
        u, [&](int i, int j, int k) { u(i, j, k) += cfg.dt * du(i, j, k); });
    runtime::toc(timer, "euler");
  }
  const double max_elapsed = runtime::toc(timer);

  // Per-section timing breakdown (rank 0 only; collective-safe).
  runtime::print_timing_summary(timer, /*print_rank=*/0);

  // Reporting: bridge a `for_each_inner`-style visitor to the shared
  // `heat3d::report` API which expects `cb(x, y, z, value)`. We
  // report over the *interior* (skipping the outermost owned layer)
  // so the L2 number is comparable with `heat3d_fd`'s reporting,
  // which uses `LocalField::for_each_interior` (also `[hw, n-hw)`).
  heat3d::report(rank, nproc, cfg, "fd_manual",
                 "manual stencil, padded brick, non-blocking halos", max_elapsed,
                 "(periodic; manual loop, interior L2)", [&u, hw](auto &&cb) {
                   field::for_each_inner(u, hw, [&](int i, int j, int k) {
                     const auto p = u.global_coords(i, j, k);
                     cb(p[0], p[1], p[2], u(i, j, k));
                   });
                 });
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg =
            heat3d::parse_spectral_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) return EXIT_FAILURE;
        run_fd_manual(*cfg, rank, nproc);
        return EXIT_SUCCESS;
      });
}
