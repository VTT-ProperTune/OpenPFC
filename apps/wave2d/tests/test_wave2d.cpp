// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/comm_sparse_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/stage_preparation.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/fd_apply.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/checkpoint_service.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/time.hpp>

#include <wave2d/cli.hpp>
#include <wave2d/reporting.hpp>
#include <wave2d/wave_boundary.hpp>
#include <wave2d/wave_model.hpp>
#include <wave2d/wave_step_separated.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace pfc;

TEST_CASE("wave2d::kC is 1.0", "[wave2d]") {
  REQUIRE_THAT(wave2d::kC, WithinAbs(1.0, 1e-15));
}

TEST_CASE("WaveModel::rhs basic", "[wave2d][WaveModel]") {
  wave2d::WaveModel m;
  m.inv_dx2 = 1.0;
  m.inv_dy2 = 1.0;
  wave2d::WaveLaplacian lap{.lxx = 0.0, .lyy = 0.0};
  auto inc = m.rhs(0.0, 2.5, lap);
  REQUIRE_THAT(inc.du, WithinAbs(2.5, 1e-15));
  REQUIRE_THAT(inc.dv, WithinAbs(0.0, 1e-15));
  lap.lxx = 1.0;
  lap.lyy = -0.5;
  inc = m.rhs(0.0, 0.0, lap);
  REQUIRE_THAT(inc.dv, WithinAbs(0.5, 1e-15));
}

TEST_CASE("parse_y_bc", "[wave2d][cli]") {
  REQUIRE(wave2d::parse_y_bc("dirichlet"));
  REQUIRE(*wave2d::parse_y_bc("d") == wave2d::YBoundaryKind::Dirichlet);
  REQUIRE(*wave2d::parse_y_bc("neumann") == wave2d::YBoundaryKind::Neumann);
  REQUIRE_FALSE(wave2d::parse_y_bc("bogus").has_value());
}

TEST_CASE("parse_manual round-trip", "[wave2d][cli]") {
  const char *argv[] = {"wave2d_fd_manual", "32", "40", "10", "0.02",
                        "dirichlet",        "0"};
  const auto c = wave2d::parse_manual(7, const_cast<char **>(argv));
  REQUIRE(c);
  REQUIRE(c->Nx == 32);
  REQUIRE(c->Ny == 40);
  REQUIRE(c->y_bc == wave2d::YBoundaryKind::Dirichlet);
}

TEST_CASE("parse_manual VTK flags", "[wave2d][cli]") {
  const char *argv[] = {
      "wave2d_fd_manual", "32",          "40", "10", "0.02", "neumann", "--vtk",
      "out/u_%04d.vti",   "--vtk-every", "5"};
  const auto c = wave2d::parse_manual(10, const_cast<char **>(argv));
  REQUIRE(c);
  REQUIRE(c->vtk_pattern == "out/u_%04d.vti");
  REQUIRE(c->vtk_every == 5);
}

TEST_CASE("parse_fd VTK flags", "[wave2d][cli]") {
  const char *argv[] = {"wave2d_fd",  "32",          "40",      "10",
                        "0.02",       "4",           "neumann", "--vtk",
                        "r_%04d.vti", "--vtk-every", "3"};
  const auto c = wave2d::parse_fd(11, const_cast<char **>(argv));
  REQUIRE(c);
  REQUIRE(c->vtk_pattern == "r_%04d.vti");
  REQUIRE(c->vtk_every == 3);
}

TEST_CASE("fill_y_physical_ghosts_padded Dirichlet mirrors", "[wave2d][bc]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  REQUIRE(rank == 0);

  constexpr int Nx = 8;
  constexpr int Ny = 8;
  auto domain = pfc::domain::create(pfc::GridSize({Nx, Ny, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  constexpr int hw = 1;
  auto u = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  u.apply([&](double, double, double) { return 0.0; });
  u(0, 0, 0) = 1.25;
  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  halo.exchange();
  wave2d::fill_y_physical_ghosts_padded(u, wave2d::YBoundaryKind::Dirichlet, Ny,
                                        0.0);
  REQUIRE_THAT(u(0, -1, 0), WithinAbs(-1.25, 1e-12));
}

TEST_CASE("wave2d mixed BC runs on StagePreparationService", "[wave2d][bc]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  REQUIRE(rank == 0);

  constexpr int Nx = 8;
  constexpr int Ny = 8;
  auto domain = pfc::domain::create(pfc::GridSize({Nx, Ny, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  constexpr int hw = 1;
  auto u = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  u.apply([&](double, double, double) { return 0.0; });
  u(0, 0, 0) = 1.25;
  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  pfc::communication::StagePreparationService<double> prep;
  prep.bind("u", halo);
  prep.set_boundary_hook([&](std::string_view) {
    wave2d::fill_y_physical_ghosts_padded(u, wave2d::YBoundaryKind::Dirichlet, Ny,
                                          0.0);
  });
  pfc::communication::StagePreparationRequirements req{
      .needs_halo_exchange = true,
      .needs_boundary_update = true,
      .ordering = pfc::communication::BoundaryHaloOrder::HaloThenBoundary,
  };
  const std::string_view fields[] = {"u"};
  prep.prepare(req, fields);
  REQUIRE_THAT(u(0, -1, 0), WithinAbs(-1.25, 1e-12));
}

TEST_CASE("step_wave_separated_order2_cpu short vs padded manual single rank",
          "[wave2d][integration]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);

  constexpr int Nx = 16;
  constexpr int Ny = 16;
  const double dt = 0.01;
  const int n_steps = 3;

  auto domain = pfc::domain::create(pfc::GridSize({Nx, Ny, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);

  constexpr int hw = 1;
  auto u_pad = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  auto v_pad = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  auto lap_pad = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  comm::HaloExchange<HostSpace, double> halo(u_pad, decomp, rank, MPI_COMM_WORLD);

  const double xc = 0.5 * static_cast<double>(Nx - 1);
  const double yc = 0.5 * static_cast<double>(Ny - 1);
  const double sigma = 2.0;
  u_pad.apply([&](double x, double y, double) {
    const double dx = x - xc;
    const double dy = y - yc;
    return std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  });
  v_pad.apply([](double, double, double) { return 0.0; });
  halo.exchange();
  wave2d::fill_y_physical_ghosts_padded(u_pad, wave2d::YBoundaryKind::Neumann, Ny,
                                        0.0);
  wave2d::WaveModel model;
  model.inv_dx2 = 1.0;
  model.inv_dy2 = 1.0;

  for (int s = 0; s < n_steps; ++s) {
    (void)s;
    halo.exchange();
    wave2d::fill_y_physical_ghosts_padded(u_pad, wave2d::YBoundaryKind::Neumann, Ny,
                                          0.0);
    u_pad.for_each_owned([&](int i, int j, int k) {
      const double lxx =
          u_pad(i + 1, j, k) - 2.0 * u_pad(i, j, k) + u_pad(i - 1, j, k);
      const double lyy =
          u_pad(i, j + 1, k) - 2.0 * u_pad(i, j, k) + u_pad(i, j - 1, k);
      lap_pad(i, j, k) = model.inv_dx2 * lxx + model.inv_dy2 * lyy;
    });
    u_pad.for_each_owned([&](int i, int j, int k) {
      const double v0 = v_pad(i, j, k);
      const double l = lap_pad(i, j, k);
      u_pad(i, j, k) += dt * v0;
      v_pad(i, j, k) += dt * wave2d::kC * wave2d::kC * l;
    });
  }

  const auto lo = u_pad.box().low;
  const auto sz = u_pad.local_size();
  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const std::size_t nlocal = static_cast<std::size_t>(nx) *
                             static_cast<std::size_t>(ny) *
                             static_cast<std::size_t>(nz);
  std::vector<double> u_sep(nlocal);
  std::vector<double> v_sep(nlocal, 0.0);
  std::vector<double> lap_sep(nlocal);
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::size_t idx =
            static_cast<std::size_t>(i) +
            static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(k) * static_cast<std::size_t>(nx * ny);
        const double x = static_cast<double>(lo[0] + i);
        const double y = static_cast<double>(lo[1] + j);
        const double dx = x - xc;
        const double dy = y - yc;
        u_sep[idx] = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
      }
    }
  }
  auto face_halos = pfc::halo::allocate_face_halos<double>(decomp, rank, 1);
  comm::SparseExchange<HostSpace, double> exch(u_sep.data(), u_sep.size(), decomp,
                                               rank, MPI_COMM_WORLD, 1);
  for (int s = 0; s < n_steps; ++s) {
    wave2d::step_wave_separated_order2_cpu(u_sep, v_sep, lap_sep, face_halos, exch,
                                           nx, ny, nz, decomp, rank, dt,
                                           wave2d::YBoundaryKind::Neumann, Ny, 0.0);
  }

  double max_diff = 0.0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::size_t idx =
            static_cast<std::size_t>(i) +
            static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(k) * static_cast<std::size_t>(nx * ny);
        max_diff = std::max(max_diff, std::abs(u_sep[idx] - u_pad(i, j, k)));
        max_diff = std::max(max_diff, std::abs(v_sep[idx] - v_pad(i, j, k)));
      }
    }
  }
  REQUIRE(max_diff < 1e-9);
}

TEST_CASE("fill_y_physical_ghosts_padded throws on insufficient local extent",
          "[wave2d][bc]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 2) return; // This test requires exactly 2 MPI ranks

  constexpr int Nx = 8;
  constexpr int Ny = 4; // As specified in acceptance criteria id=962
  constexpr int Nz = 8;
  constexpr int hw = 3;                // As specified in acceptance criteria id=962
  const double u_wall_dirichlet = 1.5; // non-zero for Dirichlet

  auto domain = pfc::domain::create(pfc::GridSize({Nx, Ny, Nz}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 2);
  auto u = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  u.apply([&](double, double, double) { return 0.0; });

  SECTION("Dirichlet boundary") {
    REQUIRE_THROWS_AS(wave2d::fill_y_physical_ghosts_padded(
                          u, wave2d::YBoundaryKind::Dirichlet, Ny, u_wall_dirichlet),
                      std::out_of_range);
    bool caught = false;
    try {
      wave2d::fill_y_physical_ghosts_padded(u, wave2d::YBoundaryKind::Dirichlet, Ny,
                                            u_wall_dirichlet);
    } catch (const std::out_of_range &e) {
      caught = true;
      std::string msg(e.what());
      REQUIRE(msg.find("rank=") != std::string::npos);
      REQUIRE(msg.find("halo_width=") != std::string::npos);
      REQUIRE(msg.find("local_ny=") != std::string::npos);
      REQUIRE(msg.find("valid_range=[") != std::string::npos);
      REQUIRE(msg.find("mirrored_global_yp=") != std::string::npos);
      REQUIRE(msg.find("computed_local_jm=") != std::string::npos);
      REQUIRE(msg.find("Dirichlet") != std::string::npos);
    }
    REQUIRE(caught);
  }

  SECTION("Neumann boundary") {
    REQUIRE_THROWS_AS(wave2d::fill_y_physical_ghosts_padded(
                          u, wave2d::YBoundaryKind::Neumann, Ny, 0.0),
                      std::out_of_range);
    bool caught = false;
    try {
      wave2d::fill_y_physical_ghosts_padded(u, wave2d::YBoundaryKind::Neumann, Ny,
                                            0.0);
    } catch (const std::out_of_range &e) {
      caught = true;
      std::string msg(e.what());
      REQUIRE(msg.find("rank=") != std::string::npos);
      REQUIRE(msg.find("halo_width=") != std::string::npos);
      REQUIRE(msg.find("local_ny=") != std::string::npos);
      REQUIRE(msg.find("valid_range=[") != std::string::npos);
      REQUIRE(msg.find("mirrored_global_yp=") != std::string::npos);
      REQUIRE(msg.find("computed_local_jm=") != std::string::npos);
      REQUIRE(msg.find("Neumann") != std::string::npos);
    }
    REQUIRE(caught);
  }
}

TEST_CASE("wave2d CheckpointService saves and restores u and v",
          "[wave2d][checkpoint]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }

  auto domain = pfc::domain::create(pfc::GridSize({4, 4, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto u = pfc::data::field_from_subdomain<double>(decomp, 0, 0);
  auto v = pfc::data::field_from_subdomain<double>(decomp, 0, 0);
  u.apply([](double x, double y, double) { return x + 0.1 * y; });
  v.apply([](double x, double y, double) { return 2.0 * x - y; });

  pfc::SimulationState state;
  state.add_field("u", std::move(u));
  state.add_field("v", std::move(v));
  pfc::Time time({0.0, 1.0, 0.1}, 0.0);
  time.next();

  const auto ckpt_root =
      std::filesystem::temp_directory_path() / "openpfc_wave2d_ckpt";
  std::error_code ec;
  std::filesystem::remove_all(ckpt_root, ec);
  std::filesystem::create_directories(ckpt_root);
  pfc::sim::CheckpointService svc({.every = 0, .directory = ckpt_root},
                                  MPI_COMM_WORLD);
  svc.save(state, time);

  auto u2 = pfc::data::field_from_subdomain<double>(decomp, 0, 0);
  auto v2 = pfc::data::field_from_subdomain<double>(decomp, 0, 0);
  pfc::SimulationState restored;
  restored.add_field("u", std::move(u2));
  restored.add_field("v", std::move(v2));
  pfc::Time time2({0.0, 1.0, 0.1}, 0.0);
  svc.load(restored, time2, svc.step_dir(1));
  REQUIRE(time2.get_increment() == 1);

  const auto &a = state.get_field<double>("u");
  const auto &b = restored.get_field<double>("u");
  const auto &va = state.get_field<double>("v");
  const auto &vb = restored.get_field<double>("v");
  const auto sz = a.local_size();
  for (int k = 0; k < sz[2]; ++k) {
    for (int j = 0; j < sz[1]; ++j) {
      for (int i = 0; i < sz[0]; ++i) {
        REQUIRE(a(i, j, k) == b(i, j, k));
        REQUIRE(va(i, j, k) == vb(i, j, k));
      }
    }
  }
  std::filesystem::remove_all(ckpt_root, ec);
}

TEST_CASE("wave2d CPU golden matches CPU-vs-CUDA config",
          "[wave2d][cpu_golden][parity]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);

  constexpr int Nx = 24;
  constexpr int Ny = 24;
  constexpr int n_steps = 8;
  const double dt = 0.01;

  auto domain = pfc::domain::create(pfc::GridSize({Nx, Ny, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  const auto &local_box = pfc::decomposition::local_box(decomp, rank);
  auto local_size = local_box.size;
  const auto lower = local_box.low;
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const std::size_t nlocal = static_cast<std::size_t>(nx) *
                             static_cast<std::size_t>(ny) *
                             static_cast<std::size_t>(nz);

  std::vector<double> u(nlocal);
  std::vector<double> v(nlocal, 0.0);
  const double xc = 0.5 * static_cast<double>(Nx - 1);
  const double yc = 0.5 * static_cast<double>(Ny - 1);
  const double sigma = 3.0;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int gx = lower[0] + ix;
        const int gy = lower[1] + iy;
        const double dx = static_cast<double>(gx) - xc;
        const double dy = static_cast<double>(gy) - yc;
        const std::size_t idx =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
        u[idx] = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
      }
    }
  }

  std::vector<double> lap(nlocal);
  constexpr int halo_width = 1;
  auto face = pfc::halo::allocate_face_halos<double>(decomp, rank, halo_width);
  pfc::comm::SparseExchange<pfc::HostSpace, double> exch(
      u.data(), u.size(), decomp, rank, MPI_COMM_WORLD, halo_width);
  for (int s = 0; s < n_steps; ++s) {
    (void)s;
    wave2d::step_wave_separated_order2_cpu(u, v, lap, face, exch, nx, ny, nz, decomp,
                                           rank, dt, wave2d::YBoundaryKind::Neumann,
                                           Ny, 0.0);
  }

  double sum_u = 0.0;
  double sumsq_u = 0.0;
  double sum_v = 0.0;
  double sumsq_v = 0.0;
  for (std::size_t i = 0; i < nlocal; ++i) {
    sum_u += u[i];
    sumsq_u += u[i] * u[i];
    sum_v += v[i];
    sumsq_v += v[i] * v[i];
  }
  if (rank == 0) {
    std::cout << std::setprecision(17) << "CPU_GOLDEN wave2d n=" << nlocal
              << " sum_u=" << sum_u << " sumsq_u=" << sumsq_u << " sum_v=" << sum_v
              << " sumsq_v=" << sumsq_v << '\n';
  }
  REQUIRE(nlocal == 576);
  REQUIRE(std::isfinite(sum_u));
  REQUIRE(std::isfinite(sumsq_u));
  REQUIRE(std::isfinite(sum_v));
  REQUIRE(std::isfinite(sumsq_v));
  // Tohtori g0005, gcc 15.2 Debug, same config as test_wave2d_cpu_vs_cuda.
  REQUIRE_THAT(sum_u, WithinRel(56.542106624911966, 1e-10));
  REQUIRE_THAT(sumsq_u, WithinRel(28.256988690744471, 1e-10));
  REQUIRE_THAT(sum_v, WithinAbs(0.0018563899833072017, 1e-12));
  REQUIRE_THAT(sumsq_v, WithinAbs(0.0042855033181676445, 1e-12));
}

// ---------------------------------------------------------------------------
// Defect: `global_rms_u_interior` was identically 0 for every configuration.
// The interior visitor trimmed the FD half-width off *every* axis of *every*
// rank's owned box. z has one layer, so `[hw, 1 - hw)` is empty and the
// reduction summed nothing — a printed `0` that meant "no cells", not "the
// field is zero". These pin both halves of the fix: the slab's z axis is not
// trimmed, and the trim happens in global index space so the answer does not
// depend on how many ranks the run used.
// ---------------------------------------------------------------------------

TEST_CASE("interior_margin leaves a degenerate axis alone", "[wave2d][reporting]") {
  REQUIRE(wave2d::interior_margin(64, 2) == 2);
  REQUIRE(wave2d::interior_margin(5, 2) == 2);
  // 2*margin would consume the whole axis: trim nothing rather than everything.
  REQUIRE(wave2d::interior_margin(4, 2) == 0);
  REQUIRE(wave2d::interior_margin(1, 2) == 0);
  REQUIRE(wave2d::interior_margin(1, 1) == 0);
}

TEST_CASE("interior reduction is non-empty on the nz==1 slab wave2d runs",
          "[wave2d][reporting]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) return;

  constexpr int Nx = 16;
  constexpr int Ny = 16;
  constexpr int hw = 2; // fd_order 4
  auto domain = pfc::domain::create(pfc::GridSize({Nx, Ny, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto u = pfc::data::field_from_subdomain<double>(decomp, 0, hw);
  u.apply([](double, double, double) { return 1.0; });

  const auto s = wave2d::interior_stats(u, hw);
  // x and y lose a 2-cell shell at each end; the single z layer is kept.
  REQUIRE(s.count == static_cast<std::int64_t>(Nx - 2 * hw) * (Ny - 2 * hw));
  REQUIRE(s.count > 0);
  REQUIRE_THAT(std::sqrt(s.sum_sq / static_cast<double>(s.count)),
               WithinAbs(1.0, 1e-15));
}

TEST_CASE("interior reduction does not change with the rank count",
          "[wave2d][reporting]") {
  // `interior_stats` is rank-local, so a multi-rank decomposition can be
  // summed here without any MPI traffic. Trimming each rank's *owned* box
  // instead of the global one would eat a shell at every subdomain seam and
  // make this sum shrink as ranks are added.
  constexpr int Nx = 24;
  constexpr int Ny = 16;
  constexpr int hw = 2;
  auto domain = pfc::domain::create(pfc::GridSize({Nx, Ny, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  const auto seed = [](double x, double y, double) {
    return 1.0 + 0.25 * x - 0.5 * y;
  };

  auto decomp1 = pfc::decomposition::create(domain, 1);
  auto u1 = pfc::data::field_from_subdomain<double>(decomp1, 0, hw);
  u1.apply(seed);
  const auto single = wave2d::interior_stats(u1, hw);
  REQUIRE(single.count == static_cast<std::int64_t>(Nx - 2 * hw) * (Ny - 2 * hw));

  for (int ranks : {2, 4}) {
    auto decompN = pfc::decomposition::create(domain, ranks);
    wave2d::InteriorStats total;
    for (int r = 0; r < ranks; ++r) {
      auto ur = pfc::data::field_from_subdomain<double>(decompN, r, hw);
      ur.apply(seed);
      const auto s = wave2d::interior_stats(ur, hw);
      total.sum_sq += s.sum_sq;
      total.count += s.count;
    }
    INFO("ranks = " << ranks);
    REQUIRE(total.count == single.count);
    REQUIRE_THAT(total.sum_sq, WithinRel(single.sum_sq, 1e-12));
  }
}

// ---------------------------------------------------------------------------
// Defect: `wave2d_fd` advertised even fd_order 2..20 and aborted at order 4
// with "owned extents 64x64x1 cannot host halo_width=2 owned send slabs".
// The +/-Z faces of an nz==1 slab carry nothing the Laplacian reads, so the
// exchanger should not demand that z be able to host a send slab.
// ---------------------------------------------------------------------------

TEST_CASE("fd_order > 2 halo exchange constructs on an nz==1 slab",
          "[wave2d][halo]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) return;

  constexpr int Nx = 64;
  constexpr int Ny = 64;
  auto domain = pfc::domain::create(pfc::GridSize({Nx, Ny, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);

  comm::HaloExchangeOptions in_plane;
  in_plane.directions = pfc::halo::presets::Axes2D();

  // Every advertised order, not just the one that used to work.
  for (int fd_order = 2; fd_order <= 20; fd_order += 2) {
    pfc::field::fd::EvenCentralD2View stencil{};
    INFO("fd_order = " << fd_order);
    REQUIRE(pfc::field::fd::lookup_even_central_d2(fd_order, &stencil));
    auto u = pfc::data::field_from_subdomain<double>(decomp, rank,
                                                     stencil.half_width);
    REQUIRE_NOTHROW(comm::HaloExchange<HostSpace, double>(u, decomp, rank,
                                                           MPI_COMM_WORLD,
                                                           in_plane));
  }

  // Asking for +/-Z on a 1-thick z is still an error, and still says why.
  auto u4 = pfc::data::field_from_subdomain<double>(decomp, rank, 2);
  REQUIRE_THROWS_AS((comm::HaloExchange<HostSpace, double>(u4, decomp, rank,
                                                            MPI_COMM_WORLD)),
                    std::invalid_argument);
}

TEST_CASE("fd_order 4 steps the 2-D slab and reports a finite interior RMS",
          "[wave2d][integration]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) return;

  constexpr int Nx = 32;
  constexpr int Ny = 32;
  constexpr int n_steps = 20;
  constexpr double dt = 0.01;

  pfc::field::fd::EvenCentralD2View stencil{};
  REQUIRE(pfc::field::fd::lookup_even_central_d2(4, &stencil));
  const int hw = stencil.half_width;
  REQUIRE(hw == 2);

  auto domain = pfc::domain::create(pfc::GridSize({Nx, Ny, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto u = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  auto v = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  auto lap = pfc::data::field_from_subdomain<double>(decomp, rank, hw);

  comm::HaloExchangeOptions in_plane;
  in_plane.directions = pfc::halo::presets::Axes2D();
  comm::HaloExchange<HostSpace, double> halo_u(u, decomp, rank, MPI_COMM_WORLD,
                                               in_plane);

  const double xc = 0.5 * (Nx - 1);
  const double yc = 0.5 * (Ny - 1);
  const double sigma = 0.12 * std::min(Nx, Ny);
  u.apply([&](double x, double y, double) {
    const double dxc = x - xc;
    const double dyc = y - yc;
    return std::exp(-(dxc * dxc + dyc * dyc) / (2.0 * sigma * sigma));
  });
  v.apply([](double, double, double) { return 0.0; });

  const double inv_den = 1.0 / static_cast<double>(stencil.denom);
  const auto sy_stride = static_cast<std::ptrdiff_t>(u.padded_extent(0));
  const auto sz_stride = static_cast<std::ptrdiff_t>(u.padded_extent(0)) *
                         static_cast<std::ptrdiff_t>(u.padded_extent(1));

  for (int step = 0; step < n_steps; ++step) {
    halo_u.exchange();
    wave2d::fill_y_physical_ghosts_padded(u, wave2d::YBoundaryKind::Dirichlet, Ny,
                                          0.0);
    u.for_each_owned([&](int i, int j, int k) {
      const double *core = u.data();
      const auto c = static_cast<std::ptrdiff_t>(u.idx(i, j, k));
      const double dxx = pfc::field::fd::apply_d2_along<0>(stencil, core, c, 1,
                                                            sy_stride, sz_stride);
      const double dyy = pfc::field::fd::apply_d2_along<1>(stencil, core, c, 1,
                                                            sy_stride, sz_stride);
      lap(i, j, k) = inv_den * (dxx + dyy);
    });
    u.for_each_owned([&](int i, int j, int k) {
      const double v0 = v(i, j, k);
      u(i, j, k) += dt * v0;
      v(i, j, k) += dt * wave2d::kC * wave2d::kC * lap(i, j, k);
    });
    wave2d::enforce_dirichlet_y_walls_owned(u, v, Ny, 0.0);
  }

  const auto s = wave2d::interior_stats(u, hw);
  REQUIRE(s.count == static_cast<std::int64_t>(Nx - 2 * hw) * (Ny - 2 * hw));
  const double rms = std::sqrt(s.sum_sq / static_cast<double>(s.count));
  REQUIRE(std::isfinite(rms));
  REQUIRE(rms > 0.0);
}

int main(int argc, char *argv[]) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "test_wave2d: MPI_Init failed\n";
    return 1;
  }
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
