// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_fd_scratch.cpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — version 0,
 *        from-scratch driver. The very bottom of the model hierarchy.
 *
 * @details
 * This is the **most primitive** of the heat3d binaries. Where the other
 * four progressively pull plumbing back into reusable OpenPFC components,
 * this one shows what every line costs when you write the heat solver
 * by hand: the only OpenPFC piece in the hot loop is the halo exchange.
 *
 * Compared to its siblings:
 *
 *  - `heat3d_fd_manual` uses `HeatModel::rhs`, `HeatGrads`, the
 *    `for_each_inner / for_each_border / for_each_owned` lambda
 *    iterators, and `start_halo_exchange / finish_halo_exchange` for
 *    comm/compute overlap.
 *  - **`heat3d_fd_scratch`** uses **none** of those. It writes the
 *    physics inline (`u_t = D nabla^2 u` with `D = heat3d::kD`),
 *    initialises with `exp(-r^2/(4 D))` directly, runs three nested
 *    `for (k) for (j) for (i)` loops over `[0, n)` with manual padded
 *    linear indexing, accesses the data as `u_ptr[lin]` /
 *    `u_ptr[lin + sx]` / etc., and stores the per-step Laplacian in a
 *    plain `std::vector<double>` of size `nx*ny*nz` with **no halo**
 *    (matching `examples/15_finite_difference_heat.cpp`).
 *
 * The OpenPFC pieces still used (because rebuilding them would just
 * duplicate the design) are:
 *
 *  - `pfc::world::create` + `pfc::decomposition::create` for the
 *    geometry (so `mpirun -n 4` actually distributes the brick).
 *  - `pfc::field::PaddedBrick<double>` as the storage container — but
 *    the driver immediately drops to `u.data()` and computes its own
 *    `lin` by hand.
 *  - `pfc::PaddedHaloExchanger<double>::exchange_halos` (blocking; no
 *    overlap, deliberately) for periodic halo updates.
 *
 * That's it. Read this driver first if you want to understand what the
 * higher-level versions are abstracting over.
 */

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/common/mpi_timer.hpp>

#include <heat3d/cli.hpp>
#include <heat3d/heat_model.hpp>
#include <heat3d/reporting.hpp>

namespace {

using heat3d::RunConfig;

void run_fd_scratch(const RunConfig &cfg, int rank, int nproc) {
  // 1. Geometry: world + decomposition + padded storage. The PaddedBrick
  //    constructor allocates the contiguous (nx+2)*(ny+2)*(nz+2) buffer
  //    and remembers (lower, origin, spacing); it does *not* fill data.
  const auto world = pfc::world::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                                        pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                        pfc::GridSpacing({1.0, 1.0, 1.0}));
  const auto decomp = pfc::decomposition::create(world, nproc);
  const int hw = 1; // 2nd-order central stencil -> halo width 1
  pfc::field::PaddedBrick<double> u(decomp, rank, hw);
  pfc::PaddedHaloExchanger<double> halo(decomp, rank, hw, MPI_COMM_WORLD);

  // 2. Pull every quantity the manual driver hides inside `for_each_*`
  //    out into local variables, so the indexing arithmetic is visible.
  const int nx = u.nx();
  const int ny = u.ny();
  const int nz = u.nz();
  const int nxp = u.nx_padded();
  const int nyp = u.ny_padded();
  const auto lower = u.lower_global();
  const auto origin = u.origin();
  const auto dx = u.spacing();

  // Strides in the padded buffer (row-major, x fastest).
  const std::size_t sx = 1;
  const std::size_t sy = static_cast<std::size_t>(nxp);
  const std::size_t sz =
      static_cast<std::size_t>(nxp) * static_cast<std::size_t>(nyp);
  // Per-axis 1/dx^2 for the central 2nd-derivative kernel.
  const double inv_dx2_x = 1.0 / (dx[0] * dx[0]);
  const double inv_dx2_y = 1.0 / (dx[1] * dx[1]);
  const double inv_dx2_z = 1.0 / (dx[2] * dx[2]);
  // Raw pointer into the padded buffer; everything below is u_ptr[lin].
  double *const u_ptr = u.data();

  // 3. Initial condition: u(x, y, z, 0) = exp(-r^2 / (4 D)). Written
  //    inline, no model.initial_condition, no PaddedBrick::apply.
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const double x = origin[0] + (lower[0] + i) * dx[0];
        const double y = origin[1] + (lower[1] + j) * dx[1];
        const double z = origin[2] + (lower[2] + k) * dx[2];
        const double r2 = x * x + y * y + z * z;
        const std::size_t lin = (i + hw) * sx + (j + hw) * sy + (k + hw) * sz;
        u_ptr[lin] = std::exp(-r2 / (4.0 * heat3d::kD));
      }
    }
  }

  // 4. Per-step aux for the Laplacian. No halo, contiguous nx*ny*nz
  //    plain vector — different (and visibly different) flat indexing
  //    `lin_lap = i + j*nx + k*nx*ny` from the padded `u`, illustrating
  //    why padded layouts have a real cost benefit for stencils.
  std::vector<double> lap(static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz),
                          0.0);

  // 5. Time loop. One MpiTimer brackets the whole thing; no per-stage
  //    breakdown — the driver is meant to be read top-to-bottom.
  pfc::runtime::MpiTimer timer{MPI_COMM_WORLD};
  pfc::runtime::tic(timer);
  for (int step = 0; step < cfg.n_steps; ++step) {

    // The single OpenPFC call inside the hot loop: refresh the halo
    // ring of `u` from the six neighbour ranks (periodic in 1-rank).
    halo.exchange_halos(u_ptr, u.size());

    // Build the Laplacian: textbook 7-point central stencil, written
    // out via stride arithmetic. The key step: `lin + sx`, `lin - sx`
    // etc. give the six neighbours in the padded buffer.
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const std::size_t lin = (i + hw) * sx + (j + hw) * sy + (k + hw) * sz;
          const double c = u_ptr[lin];
          const double l =
              (u_ptr[lin + sx] - 2.0 * c + u_ptr[lin - sx]) * inv_dx2_x +
              (u_ptr[lin + sy] - 2.0 * c + u_ptr[lin - sy]) * inv_dx2_y +
              (u_ptr[lin + sz] - 2.0 * c + u_ptr[lin - sz]) * inv_dx2_z;
          lap[static_cast<std::size_t>(i) +
              static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
              static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                  static_cast<std::size_t>(ny)] = l;
        }
      }
    }

    // Explicit Euler update: u <- u + dt * D * lap. Two indexing
    // schemes side by side: padded `lin` for `u`, plain `lin_lap`
    // for `lap`.
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const std::size_t lin = (i + hw) * sx + (j + hw) * sy + (k + hw) * sz;
          const std::size_t lin_lap =
              static_cast<std::size_t>(i) +
              static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
              static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                  static_cast<std::size_t>(ny);
          u_ptr[lin] += cfg.dt * heat3d::kD * lap[lin_lap];
        }
      }
    }
  }
  const double max_elapsed = pfc::runtime::toc(timer);

  // 6. Report. The L2 visitor walks the same interior region the other
  //    drivers report on (skipping the outermost owned layer) using the
  //    same hand-written stride arithmetic, so the L2 number is
  //    directly comparable with `heat3d_fd` and `heat3d_fd_manual`.
  heat3d::report(
      rank, nproc, cfg, "fd_scratch",
      "from-scratch stencil, raw pointer arithmetic, plain Lap aux", max_elapsed,
      "(periodic; bare triple loop, interior L2)", [&](auto &&cb) {
        for (int k = hw; k < nz - hw; ++k) {
          for (int j = hw; j < ny - hw; ++j) {
            for (int i = hw; i < nx - hw; ++i) {
              const std::size_t lin = (i + hw) * sx + (j + hw) * sy + (k + hw) * sz;
              const double x = origin[0] + (lower[0] + i) * dx[0];
              const double y = origin[1] + (lower[1] + j) * dx[1];
              const double z = origin[2] + (lower[2] + k) * dx[2];
              cb(x, y, z, u_ptr[lin]);
            }
          }
        }
      });
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg =
            heat3d::parse_spectral_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) return EXIT_FAILURE;
        run_fd_scratch(*cfg, rank, nproc);
        return EXIT_SUCCESS;
      });
}
