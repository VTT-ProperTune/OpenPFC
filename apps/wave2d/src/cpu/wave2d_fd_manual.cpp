// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file wave2d_fd_manual.cpp
 * @brief 2D wave equation — manual 5-point Laplacian on `PaddedBrick` + periodic
 *        halos in x,z and physical y-boundary ghosts (Dirichlet or Neumann).
 */

#include <cmath>
#include <cstdlib>
#include <memory>
#include <mpi.h>

#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/kernel/data/model_types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/common/mpi_timer.hpp>

#include <wave2d/cli.hpp>
#include <wave2d/reporting.hpp>
#include <wave2d/vtk_snapshot.hpp>
#include <wave2d/wave_boundary.hpp>
#include <wave2d/wave_model.hpp>

using namespace pfc;
using wave2d::RunConfig;
using wave2d::WaveModel;
using wave2d::YBoundaryKind;

namespace {

void run_fd_manual(const RunConfig &cfg, int rank, int nproc) {
  WaveModel model;
  model.inv_dx2 = 1.0;
  model.inv_dy2 = 1.0;

  const auto world =
      world::create(GridSize({cfg.Nx, cfg.Ny, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                    GridSpacing({1.0, 1.0, 1.0}));
  const auto decomp = decomposition::create(world, nproc);

  constexpr int hw = 1;
  field::PaddedBrick<double> u(decomp, rank, hw);
  field::PaddedBrick<double> v(decomp, rank, hw);
  field::PaddedBrick<double> lap(decomp, rank, hw);

  PaddedHaloExchanger<double> halo_u(decomp, rank, hw, MPI_COMM_WORLD, 0);

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

  pfc::RealField vtk_buf;
  std::unique_ptr<pfc::VTKWriter> vtk_writer;
  if (!cfg.vtk_pattern.empty()) {
    vtk_writer = std::make_unique<pfc::VTKWriter>(cfg.vtk_pattern);
    wave2d::vtk_configure_writer(*vtk_writer, u);
    wave2d::mkdir_vtk_parent_rank0(cfg.vtk_pattern, rank);
    wave2d::vtk_write_increment(*vtk_writer, 0, u, vtk_buf);
  }

  auto stencil_lap = [&](int i, int j, int k) {
    const double lxx = u(i + 1, j, k) - 2.0 * u(i, j, k) + u(i - 1, j, k);
    const double lyy = u(i, j + 1, k) - 2.0 * u(i, j, k) + u(i, j - 1, k);
    lap(i, j, k) = model.inv_dx2 * lxx + model.inv_dy2 * lyy;
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

    if (vtk_writer && (step + 1) % cfg.vtk_every == 0) {
      wave2d::vtk_write_increment(*vtk_writer, step + 1, u, vtk_buf);
    }
  }
  const double max_elapsed = runtime::toc(timer);

  const int skip = hw;
  wave2d::report(rank, nproc, cfg, "fd_manual", "manual 5-point + padded halos",
                 max_elapsed, "(y physical BC; interior RMS u)", [&](auto &&cb) {
                   field::for_each_inner(u, skip, [&](int i, int j, int k) {
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
            wave2d::parse_manual_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) return EXIT_FAILURE;
        run_fd_manual(*cfg, rank, nproc);
        return EXIT_SUCCESS;
      });
}
