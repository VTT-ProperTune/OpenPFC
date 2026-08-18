// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_fd_manual.cpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — laboratory-style
 *        finite-difference driver.
 *
 * @details
 * This is the **"laboratory, not fortress"** counterpart to `heat3d_fd`.
 * Where `heat3d_fd` composes `Field + HaloExchange +
 * FDGradient + for_each` as three visible primitives (halo, gradient,
 * sweep) in `main`, this driver instead exposes a different decomposition:
 * a single overlapped halo exchange (start interior / finish border) and a
 * raw 2nd-order central stencil written line-by-line, with the explicit
 * Euler update inlined. Only the cumbersome plumbing —
 * decomposition, MPI face exchange, and linear-index arithmetic —
 * stays hidden behind:
 *
 *  - `pfc::data::Field<double, pfc::HostSpace>` — single contiguous buffer with
 *    `u(i, j, k)` valid for `i,j,k in [-hw, n+hw)`. No edge overwrite,
 *    no separate face vectors.
 *  - `pfc::comm::HaloExchange<HostSpace>` — non-blocking
 *    `start()` / `finish()` pair on the bound field.
 *  - `pfc::data::Field::for_each_interior / for_each_owned` —
 *    interior and owned cell iterators (with coordinate/value signatures).
 *  - `pfc::runtime::tic(timer, "label") / toc(timer, "label")` —
 *    collective-free per-section timers; `print_timing_summary` does
 *    one allreduce-max at the end to report the slowest rank.
 *
 * The hot loop literally reads:
 *
 *     halo.start();
 *     // Manual inner loops over [hw, n-hw) on each axis
 *     // ... stencil reads u(i ± 1, j, k), etc. ...
 *     halo.finish();
 *     // Manual border loops over slabs of thickness hw
 *     // ... same stencil can now reach into the halo ...
 *     u.for_each_owned([&](int i, int j, int k, double& u_val) {
 *       u_val += cfg.dt * du(i, j, k);
 *     });
 *
 * The stencil is the textbook second-order central seven-point
 * Laplacian (halo width 1). `HeatModel::rhs` is reused unchanged so
 * the physics still lives in the model; the driver just *spells out*
 * how the model is wired to the data.
 */

#include <cstdlib>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
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

void run_fd_manual(const RunConfig &cfg, int rank, int nproc) {
  // 1. Physics: initial condition + RHS live in HeatModel; the diffusion
  //    coefficient itself is hard-pinned via `heat3d::kD` in heat_model.hpp.
  HeatModel model;

  // 2. Hidden plumbing: domain geometry + decomposition. The single
  //    `decomposition::create(domain, nproc)` call auto-picks the
  //    rank grid; the manual driver does not need to spell it out.
  const auto domain =
      pfc::domain::create(GridSize({cfg.N, cfg.N, cfg.N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                          GridSpacing({1.0, 1.0, 1.0}));
  const auto decomp = decomposition::create(domain, nproc);

  // 3. Two halo-padded buffers: `u` (state) and `du` (RHS). Both
  //    cover the local owned core plus a 1-cell ghost ring on every
  //    side, all in one contiguous buffer.
  const int hw = 1; // second-order central Laplacian -> stencil radius 1
  const auto owned_box = pfc::decomposition::local_box(decomp, rank);
  pfc::data::Field<double, pfc::HostSpace> u(domain, owned_box, hw);
  pfc::data::Field<double, pfc::HostSpace> du(domain, owned_box, hw);

  // 4. Hidden plumbing: in-place non-blocking halo exchanger.
  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);

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
    // Using the bound exchanger API for brevity.
    halo.start();

    // Inner cells: stencil only reads owned cells, no halo dependency.
    // Inner region is [hw, nx-hw) x [hw, ny-hw) x [hw, nz-hw).
    runtime::tic(timer, "inner");
    {
      const int nx = u.local_size()[0];
      const int ny = u.local_size()[1];
      const int nz = u.local_size()[2];
      const int imin = hw, imax = nx - hw;
      const int jmin = hw, jmax = ny - hw;
      const int kmin = hw, kmax = nz - hw;
      if (imin < imax && jmin < jmax && kmin < kmax) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = kmin; k < kmax; ++k) {
          for (int j = jmin; j < jmax; ++j) {
            for (int i = imin; i < imax; ++i) {
              stencil_step(i, j, k);
            }
          }
        }
      }
    }
    runtime::toc(timer, "inner");

    // Wait for neighbour data to land in the halo ring.
    runtime::tic(timer, "halo_wait");
    halo.finish();
    runtime::toc(timer, "halo_wait");

    // Border cells: same stencil, now safely reaches into the halo.
    runtime::tic(timer, "border");
    {
      const int nx = u.local_size()[0];
      const int ny = u.local_size()[1];
      const int nz = u.local_size()[2];
      // If the domain is too small for an inner region, all owned cells are border.
      if (nx <= 2 * hw || ny <= 2 * hw || nz <= 2 * hw) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k) {
          for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
              stencil_step(i, j, k);
            }
          }
        }
      } else {
        // Six face slabs of thickness hw (each cell visited exactly once).
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k) {
          for (int j = 0; j < ny; ++j) {
            // Left/right x-faces (full y/z extents).
            for (int i = 0; i < hw; ++i) stencil_step(i, j, k);
            for (int i = nx - hw; i < nx; ++i) stencil_step(i, j, k);
          }
        }
#pragma omp parallel for schedule(static)
        for (int k = 0; k < nz; ++k) {
          // Top/bottom y-faces (excluding x-faces already done).
          for (int j = 0; j < hw; ++j) {
            for (int i = hw; i < nx - hw; ++i) stencil_step(i, j, k);
          }
          for (int j = ny - hw; j < ny; ++j) {
            for (int i = hw; i < nx - hw; ++i) stencil_step(i, j, k);
          }
        }
#pragma omp parallel for schedule(static)
        // Front/back z-faces (excluding x and y faces already done).
        for (int k = 0; k < hw; ++k) {
          for (int j = hw; j < ny - hw; ++j) {
            for (int i = hw; i < nx - hw; ++i) stencil_step(i, j, k);
          }
        }
#pragma omp parallel for schedule(static)
        for (int k = nz - hw; k < nz; ++k) {
          for (int j = hw; j < ny - hw; ++j) {
            for (int i = hw; i < nx - hw; ++i) stencil_step(i, j, k);
          }
        }
      }
    }
    runtime::toc(timer, "border");

    // Explicit Euler over the full owned region: u <- u + dt * du.
    runtime::tic(timer, "euler");
    u.for_each_owned([&](int i, int j, int k) { u(i, j, k) += cfg.dt * du(i, j, k); });
    runtime::toc(timer, "euler");
  }
  const double max_elapsed = runtime::toc(timer);

  // Per-section timing breakdown (rank 0 only; collective-safe).
  runtime::print_timing_summary(timer, /*print_rank=*/0);

  // Reporting: bridge a `for_each_interior`-style visitor to the shared
  // `heat3d::report` API which expects `cb(x, y, z, value)`. We
  // report over the *interior* (skipping the outermost owned layer)
  // so the L2 number is computed over the interior domain
  // matching the design documented in reporting.hpp.
  heat3d::report(rank, nproc, cfg, "fd_manual",
                 "manual stencil, field, non-blocking halos", max_elapsed,
                 "(periodic; manual loop, interior L2)", [&](auto &&cb) {
                   // Field's for_each_interior directly provides (x, y, z, value) signatures.
                   u.for_each_interior(
                       [&](double x, double y, double z, const double &u_val) {
                         cb(x, y, z, u_val);
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
