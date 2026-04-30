// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file wave2d_fd.cpp
 * @brief 2D wave — `PaddedBrick` + runtime even-order central Laplacian
 *        (`EvenCentralD2View`) + same y-physical BC path as `wave2d_fd_manual`.
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/fd_apply.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/common/mpi_timer.hpp>

#include <wave2d/cli.hpp>
#include <wave2d/reporting.hpp>
#include <wave2d/wave_boundary.hpp>
#include <wave2d/wave_model.hpp>

using namespace pfc;
using wave2d::RunConfig;
using wave2d::WaveModel;
using wave2d::YBoundaryKind;

namespace {

int run_fd(const RunConfig &cfg, int rank, int nproc) {
  WaveModel model;
  model.inv_dx2 = 1.0;
  model.inv_dy2 = 1.0;

  pfc::field::fd::EvenCentralD2View stencil{};
  if (!pfc::field::fd::lookup_even_central_d2(cfg.fd_order, &stencil)) {
    if (rank == 0) {
      std::cerr << "wave2d_fd: internal error: fd_order lookup failed\n";
    }
    return EXIT_FAILURE;
  }

  const int hw = stencil.half_width;
  const auto world =
      world::create(GridSize({cfg.Nx, cfg.Ny, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                    GridSpacing({1.0, 1.0, 1.0}));
  const auto decomp = decomposition::create(world, nproc);

  field::PaddedBrick<double> u(decomp, rank, hw);
  field::PaddedBrick<double> v(decomp, rank, hw);
  field::PaddedBrick<double> lap(decomp, rank, hw);

  PaddedHaloExchanger<double> halo_u(decomp, rank, hw, MPI_COMM_WORLD, 0);

  const double inv_den = 1.0 / static_cast<double>(stencil.denom);
  const double sx = model.inv_dx2 * inv_den;
  const double sy = model.inv_dy2 * inv_den;

  const std::ptrdiff_t sy_stride = static_cast<std::ptrdiff_t>(u.nx_padded());
  const std::ptrdiff_t sz_stride = static_cast<std::ptrdiff_t>(u.nx_padded()) *
                                   static_cast<std::ptrdiff_t>(u.ny_padded());

  const double xc = 0.5 * static_cast<double>(cfg.Nx - 1);
  const double yc = 0.5 * static_cast<double>(cfg.Ny - 1);
  const double sigma = 0.12 * static_cast<double>(std::min(cfg.Nx, cfg.Ny));

  u.apply([&](double x, double y, double /*z*/) {
    const double dx = x - xc;
    const double dy = y - yc;
    return std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  });
  v.apply([](double, double, double) { return 0.0; });

  halo_u.exchange_halos(u.data(), u.size());
  wave2d::fill_y_physical_ghosts_padded(u, cfg.y_bc, cfg.Ny,
                                        static_cast<double>(cfg.u_wall));
  if (cfg.y_bc == YBoundaryKind::Dirichlet) {
    wave2d::enforce_dirichlet_y_walls_owned(u, v, cfg.Ny,
                                            static_cast<double>(cfg.u_wall));
  }

  auto stencil_lap = [&](int i, int j, int k) {
    const double *core = u.data();
    const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(u.idx(i, j, k));
    const double dxx =
        pfc::field::fd::apply_d2_along<0>(stencil, core, c,
                                          /*sx=*/1, sy_stride, sz_stride);
    const double dyy =
        pfc::field::fd::apply_d2_along<1>(stencil, core, c,
                                          /*sx=*/1, sy_stride, sz_stride);
    lap(i, j, k) = sx * dxx + sy * dyy;
  };

  runtime::MpiTimer timer{MPI_COMM_WORLD};
  runtime::tic(timer);
  for (int step = 0; step < cfg.n_steps; ++step) {
    halo_u.exchange_halos(u.data(), u.size());
    wave2d::fill_y_physical_ghosts_padded(u, cfg.y_bc, cfg.Ny,
                                          static_cast<double>(cfg.u_wall));
    field::for_each_owned(u, [&](int i, int j, int k) { stencil_lap(i, j, k); });

    field::for_each_owned(u, [&](int i, int j, int k) {
      const double v0 = v(i, j, k);
      const double l = lap(i, j, k);
      u(i, j, k) += cfg.dt * v0;
      v(i, j, k) += cfg.dt * wave2d::kC * wave2d::kC * l;
    });

    if (cfg.y_bc == YBoundaryKind::Dirichlet) {
      wave2d::enforce_dirichlet_y_walls_owned(u, v, cfg.Ny,
                                              static_cast<double>(cfg.u_wall));
    }
  }
  const double max_elapsed = runtime::toc(timer);

  const int skip = hw;
  wave2d::report(rank, nproc, cfg, "fd", wave2d::fd_extra_metadata(cfg), max_elapsed,
                 "(runtime FD order; y physical BC; interior RMS u)",
                 [&](auto &&cb) {
                   field::for_each_inner(u, skip, [&](int i, int j, int k) {
                     const auto p = u.global_coords(i, j, k);
                     cb(p[0], p[1], p[2], u(i, j, k));
                   });
                 });
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg = wave2d::parse_fd_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) return EXIT_FAILURE;
        return run_fd(*cfg, rank, nproc);
      });
}
