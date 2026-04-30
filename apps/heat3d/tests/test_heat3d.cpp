// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_heat3d.cpp
 * @brief Catch2 unit tests for the heat3d application's HeatModel.
 *
 * @details
 * The model is intentionally MPI/FFT-free, so the bulk of the tests work on
 * a default-constructed `heat3d::HeatModel` directly. A handful of
 * integration tests exercise the model against OpenPFC's `LocalField`,
 * `FdGradient` and explicit-Euler stepper on a small single-rank grid to
 * verify the wiring physicists actually use in `heat3d.cpp`.
 *
 * Layout follows `apps/tungsten/tests/test_tungsten.cpp` (Catch2 + custom
 * MPI-aware main).
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

#include <heat3d/cli.hpp>
#include <heat3d/heat_grads.hpp>
#include <heat3d/heat_model.hpp>
#include <heat3d/reporting.hpp>

using Catch::Matchers::WithinAbs;
using heat3d::HeatModel;

// -----------------------------------------------------------------------------
// Pure-model tests (no MPI / FFT / OpenMP).
// -----------------------------------------------------------------------------

TEST_CASE("heat3d::kD: pinned to 1.0", "[heat3d][HeatModel]") {
  REQUIRE_THAT(heat3d::kD, WithinAbs(1.0, 1e-12));
}

TEST_CASE("HeatModel: defaults", "[heat3d][HeatModel]") {
  HeatModel model;

  SECTION("default IC at the origin equals 1") {
    REQUIRE_THAT(model.initial_condition(0.0, 0.0, 0.0), WithinAbs(1.0, 1e-12));
  }

  SECTION("default IC away from origin matches exp(-r^2/(4 kD))") {
    const double r2 = 1.0 + 4.0 + 9.0; // (1, 2, 3)
    REQUIRE_THAT(model.initial_condition(1.0, 2.0, 3.0),
                 WithinAbs(std::exp(-r2 / (4.0 * heat3d::kD)), 1e-12));
  }

  SECTION("default boundary_value is empty") {
    REQUIRE_FALSE(static_cast<bool>(model.boundary_value));
  }
}

TEST_CASE("HeatModel: IC override replaces the default lambda",
          "[heat3d][HeatModel]") {
  HeatModel model;
  model.initial_condition = [](double x, double y, double z) {
    return x + y + z; // arbitrary linear field
  };

  REQUIRE_THAT(model.initial_condition(0.0, 0.0, 0.0), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(model.initial_condition(1.5, -2.0, 3.5), WithinAbs(3.0, 1e-12));
}

TEST_CASE("HeatModel: rhs = kD * (xx + yy + zz)", "[heat3d][HeatModel]") {
  HeatModel model;

  SECTION("zero Laplacian gives zero RHS") {
    heat3d::HeatGrads g{.xx = 0.0, .yy = 0.0, .zz = 0.0};
    REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(0.0, 1e-12));
  }

  SECTION("uniform unit Laplacian gives RHS = 3 * kD") {
    heat3d::HeatGrads g{.xx = 1.0, .yy = 1.0, .zz = 1.0};
    REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(3.0 * heat3d::kD, 1e-12));
  }

  SECTION("rhs sums xx + yy + zz scaled by kD") {
    heat3d::HeatGrads g{.xx = 1.0, .yy = -2.0, .zz = 0.5};
    const double lap = 1.0 - 2.0 + 0.5; // -0.5
    REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(heat3d::kD * lap, 1e-12));
  }

  SECTION("rhs ignores t") {
    heat3d::HeatGrads g{.xx = 1.0, .yy = 1.0, .zz = 1.0};
    const double expected = 3.0 * heat3d::kD;
    REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(expected, 1e-12));
    REQUIRE_THAT(model.rhs(1234.5, g), WithinAbs(expected, 1e-12));
  }
}

// -----------------------------------------------------------------------------
// Integration tests against OpenPFC primitives (single MPI rank).
//
// `pfc::sim::stacks::FdCpuStack` is the recommended bundle: World,
// Decomposition, LocalField, halo buffers, and the exchanger are constructed
// in dependency order so the exchanger's `const Decomposition&` reference
// stays valid for the lifetime of the stack. (`Decomposition` itself owns
// its `World` by value — see
// `tests/unit/kernel/decomposition/test_decomposition_lifetime.cpp` — so a
// helper that returns a Decomposition by value is also safe.)
// We use the stack here as the canonical "one statement to set up an
// FD-on-CPU solver" entry point — exactly what
// `apps/heat3d/src/cpu/heat3d.cpp` uses.
// -----------------------------------------------------------------------------

TEST_CASE("HeatModel + FdCpuStack: u.apply samples the model IC",
          "[heat3d][FdCpuStack]") {
  constexpr int N = 8;

  HeatModel model;

  // FdCpuStack's LocalField stores only owned cells (the face-halo buffer
  // lives separately), so the array indices map directly to the local
  // subworld coordinates. For nproc=1 the local subworld == global world,
  // so u(ix, iy, iz) samples the IC at global (ix, iy, iz).
  pfc::sim::stacks::FdCpuStack stack(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}), /*fd_order=*/2, /*rank=*/0, /*nproc=*/1,
      MPI_COMM_WORLD);
  stack.u().apply(model.initial_condition);

  SECTION("origin cell evaluates to exp(0) = 1") {
    REQUIRE_THAT(stack.u()(0, 0, 0), WithinAbs(1.0, 1e-12));
  }

  SECTION("owned cells match exp(-r^2/(4 kD))") {
    for (int iz = 0; iz < 4; ++iz) {
      for (int iy = 0; iy < 4; ++iy) {
        for (int ix = 0; ix < 4; ++ix) {
          const double x = static_cast<double>(ix);
          const double y = static_cast<double>(iy);
          const double z = static_cast<double>(iz);
          const double r2 = x * x + y * y + z * z;
          INFO("ix=" << ix << " iy=" << iy << " iz=" << iz);
          REQUIRE_THAT(stack.u()(ix, iy, iz),
                       WithinAbs(std::exp(-r2 / (4.0 * heat3d::kD)), 1e-12));
        }
      }
    }
  }
}

TEST_CASE("HeatModel + FdCpuStack + EulerStepper: explicit-Euler FD steps "
          "decrease the L2 norm of a Gaussian (heat dissipation)",
          "[heat3d][FdCpuStack][Euler]") {
  constexpr int N = 16;
  const int order = 2;

  pfc::sim::stacks::FdCpuStack stack(pfc::GridSize({N, N, N}),
                                     pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                     pfc::GridSpacing({1.0, 1.0, 1.0}), order,
                                     /*rank=*/0, /*nproc=*/1, MPI_COMM_WORLD);

  HeatModel model;
  stack.u().apply(model.initial_condition);

  // L2 norm of the initial state (interior only).
  double sum0 = 0.0;
  stack.u().for_each_interior(
      [&sum0](double, double, double, double v) { sum0 += v * v; });

  auto grad = pfc::field::create<heat3d::HeatGrads>(stack.u(), order);
  auto stepper = pfc::sim::steppers::create(stack.u(), grad, model,
                                            /*dt=*/1.0e-3);

  for (int step = 0; step < 5; ++step) {
    stack.exchange_halos();
    (void)stepper.step(static_cast<double>(step) * 1.0e-3, stack.u().vec());
  }

  double sum1 = 0.0;
  stack.u().for_each_interior(
      [&sum1](double, double, double, double v) { sum1 += v * v; });

  REQUIRE(sum1 < sum0);
  REQUIRE(sum1 > 0.0);
  REQUIRE(std::isfinite(sum1));
}

// -----------------------------------------------------------------------------
// Grads-type pruning: an alternative model-owned grads aggregate that drops
// `zz` (a 2D-style heat slab needs only `xx + yy`) must compile against
// `pfc::field::create<G>(...)` and the explicit-Euler stepper. This proves
// the kernel introspects member-by-member rather than assuming a fixed shape.
// -----------------------------------------------------------------------------

namespace {
struct PartialHeatGrads {
  double xx{};
  double yy{};
};

struct PartialHeatModel {
  double D = 1.0;
  inline double rhs(double, const PartialHeatGrads &g) const noexcept {
    return D * (g.xx + g.yy);
  }
};
} // namespace

TEST_CASE("FdGradient<G> + EulerStepper compile and run with a pruned grads "
          "aggregate (only xx, yy)",
          "[heat3d][FdCpuStack][prune]") {
  constexpr int N = 12;
  const int order = 2;

  pfc::sim::stacks::FdCpuStack stack(pfc::GridSize({N, N, N}),
                                     pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                     pfc::GridSpacing({1.0, 1.0, 1.0}), order,
                                     /*rank=*/0, /*nproc=*/1, MPI_COMM_WORLD);

  PartialHeatModel model;
  model.D = 1.0;
  stack.u().apply([](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / 4.0);
  });

  auto grad = pfc::field::create<PartialHeatGrads>(stack.u(), order);
  auto stepper = pfc::sim::steppers::create(stack.u(), grad, model,
                                            /*dt=*/1.0e-3);

  stack.exchange_halos();
  const double t1 = stepper.step(0.0, stack.u().vec());
  REQUIRE_THAT(t1, WithinAbs(1.0e-3, 1e-15));

  double l2sq = 0.0;
  stack.u().for_each_interior(
      [&l2sq](double, double, double, double v) { l2sq += v * v; });
  REQUIRE(std::isfinite(l2sq));
  REQUIRE(l2sq > 0.0);
}

// -----------------------------------------------------------------------------
// Laboratory-style manual FD driver: PaddedBrick + PaddedHaloExchanger +
// brick_iteration helpers. The smoke test mirrors the compact `FdCpuStack`
// test above so the two paths can be cross-checked.
// -----------------------------------------------------------------------------

TEST_CASE("Manual FD driver (PaddedBrick + PaddedHaloExchanger): smoke + L2",
          "[heat3d][fd_manual]") {
  using namespace pfc;
  constexpr int N = 16;
  const int hw = 1;
  const double dt = 1.0e-3;
  const int n_steps = 5;

  auto world = world::create(GridSize({N, N, N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, /*nproc=*/1);

  field::PaddedBrick<double> u(decomp, /*rank=*/0, hw);
  field::PaddedBrick<double> du(decomp, /*rank=*/0, hw);
  PaddedHaloExchanger<double> halo(decomp, /*rank=*/0, hw, MPI_COMM_WORLD);

  HeatModel model;
  u.apply(model.initial_condition);

  double sum0 = 0.0;
  field::for_each_inner(
      u, hw, [&](int i, int j, int k) { sum0 += u(i, j, k) * u(i, j, k); });
  REQUIRE(sum0 > 0.0);

  auto stencil_step = [&](int i, int j, int k) {
    heat3d::HeatGrads g{};
    g.xx = u(i + 1, j, k) - 2.0 * u(i, j, k) + u(i - 1, j, k);
    g.yy = u(i, j + 1, k) - 2.0 * u(i, j, k) + u(i, j - 1, k);
    g.zz = u(i, j, k + 1) - 2.0 * u(i, j, k) + u(i, j, k - 1);
    du(i, j, k) = model.rhs(0.0, g);
  };

  for (int step = 0; step < n_steps; ++step) {
    halo.start_halo_exchange(u.data(), u.size());
    field::for_each_inner(u, hw, stencil_step);
    halo.finish_halo_exchange();
    field::for_each_border(u, hw, stencil_step);
    field::for_each_owned(
        u, [&](int i, int j, int k) { u(i, j, k) += dt * du(i, j, k); });
  }

  double sum1 = 0.0;
  field::for_each_inner(u, hw, [&](int i, int j, int k) {
    REQUIRE(std::isfinite(u(i, j, k)));
    sum1 += u(i, j, k) * u(i, j, k);
  });

  REQUIRE(sum1 < sum0);
  REQUIRE(sum1 > 0.0);

  double l2sq = 0.0;
  double cnt = 0.0;
  const double t_final = static_cast<double>(n_steps) * dt;
  field::for_each_inner(u, hw, [&](int i, int j, int k) {
    const auto p = u.global_coords(i, j, k);
    const double r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    const double u_exact = heat3d::analytic_gaussian(r2, t_final, heat3d::kD);
    const double diff = u(i, j, k) - u_exact;
    l2sq += diff * diff;
    cnt += 1.0;
  });
  const double l2_rms = std::sqrt(l2sq / cnt);
  REQUIRE(std::isfinite(l2_rms));
  REQUIRE(l2_rms < 1.0e-3);
}

TEST_CASE("Manual FD driver: produces same interior L2 as compact FdCpuStack path",
          "[heat3d][fd_manual]") {
  using namespace pfc;
  constexpr int N = 16;
  const int hw = 1;
  const int order = 2;
  const double dt = 1.0e-3;
  const int n_steps = 5;

  HeatModel model;

  sim::stacks::FdCpuStack stack(GridSize({N, N, N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                                GridSpacing({1.0, 1.0, 1.0}), order, 0, 1,
                                MPI_COMM_WORLD);
  stack.u().apply(model.initial_condition);
  auto grad = field::create<heat3d::HeatGrads>(stack.u(), order);
  auto stepper = sim::steppers::create(stack.u(), grad, model, dt);
  for (int step = 0; step < n_steps; ++step) {
    stack.exchange_halos();
    (void)stepper.step(static_cast<double>(step) * dt, stack.u().vec());
  }
  double l2_compact = 0.0;
  double cnt = 0.0;
  const double t_final = static_cast<double>(n_steps) * dt;
  stack.u().for_each_interior([&](double x, double y, double z, double v) {
    const double u_exact =
        heat3d::analytic_gaussian(x * x + y * y + z * z, t_final, heat3d::kD);
    const double diff = v - u_exact;
    l2_compact += diff * diff;
    cnt += 1.0;
  });
  l2_compact = std::sqrt(l2_compact / cnt);

  auto world = world::create(GridSize({N, N, N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, 1);
  field::PaddedBrick<double> u(decomp, 0, hw);
  field::PaddedBrick<double> du(decomp, 0, hw);
  PaddedHaloExchanger<double> halo(decomp, 0, hw, MPI_COMM_WORLD);
  u.apply(model.initial_condition);

  auto stencil_step = [&](int i, int j, int k) {
    heat3d::HeatGrads g{};
    g.xx = u(i + 1, j, k) - 2.0 * u(i, j, k) + u(i - 1, j, k);
    g.yy = u(i, j + 1, k) - 2.0 * u(i, j, k) + u(i, j - 1, k);
    g.zz = u(i, j, k + 1) - 2.0 * u(i, j, k) + u(i, j, k - 1);
    du(i, j, k) = model.rhs(0.0, g);
  };

  for (int step = 0; step < n_steps; ++step) {
    halo.start_halo_exchange(u.data(), u.size());
    field::for_each_inner(u, hw, stencil_step);
    halo.finish_halo_exchange();
    field::for_each_border(u, hw, stencil_step);
    field::for_each_owned(
        u, [&](int i, int j, int k) { u(i, j, k) += dt * du(i, j, k); });
  }

  double l2_manual = 0.0;
  double cnt2 = 0.0;
  field::for_each_inner(u, hw, [&](int i, int j, int k) {
    const auto p = u.global_coords(i, j, k);
    const double u_exact = heat3d::analytic_gaussian(
        p[0] * p[0] + p[1] * p[1] + p[2] * p[2], t_final, heat3d::kD);
    const double diff = u(i, j, k) - u_exact;
    l2_manual += diff * diff;
    cnt2 += 1.0;
  });
  l2_manual = std::sqrt(l2_manual / cnt2);

  // Both paths apply the same physical 7-point central stencil to the same
  // initial condition; they should agree to floating-point round-off.
  // `FdGradient` uses an inner-loop multiply by `1/dx^2 = 1.0` and possibly a
  // different summation order than the explicit lambda below, so we allow a
  // few ulps of slop rather than asking for bit equality.
  REQUIRE_THAT(l2_manual, WithinAbs(l2_compact, 1.0e-7));
}

// -----------------------------------------------------------------------------
// Version-0 (from-scratch) FD driver: bare triple loops + manual padded
// linear indexing + raw pointer arithmetic + plain Lap aux. The first test
// is a smoke + L2-vs-analytic check; the second cross-checks the L2 against
// the compact `FdCpuStack` path the way the manual driver does, to prove
// the from-scratch loop is numerically equivalent to the framework path.
// -----------------------------------------------------------------------------

namespace {

// Helper: run the same hot loop heat3d_fd_scratch.cpp ships, single rank.
// Stays in the test file so the production driver stays free of test hooks.
inline void run_scratch_loop_(pfc::field::PaddedBrick<double> &u,
                              pfc::PaddedHaloExchanger<double> &halo, double dt,
                              int n_steps) {
  const int hw = u.halo_width();
  const int nx = u.nx();
  const int ny = u.ny();
  const int nz = u.nz();
  const int nxp = u.nx_padded();
  const int nyp = u.ny_padded();
  const auto dx = u.spacing();
  const std::size_t sx = 1;
  const std::size_t sy = static_cast<std::size_t>(nxp);
  const std::size_t sz =
      static_cast<std::size_t>(nxp) * static_cast<std::size_t>(nyp);
  const double inv_dx2_x = 1.0 / (dx[0] * dx[0]);
  const double inv_dx2_y = 1.0 / (dx[1] * dx[1]);
  const double inv_dx2_z = 1.0 / (dx[2] * dx[2]);
  double *const u_ptr = u.data();
  std::vector<double> lap(static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz),
                          0.0);
  for (int step = 0; step < n_steps; ++step) {
    halo.exchange_halos(u_ptr, u.size());
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const std::size_t lin = (i + hw) * sx + (j + hw) * sy + (k + hw) * sz;
          const double c = u_ptr[lin];
          lap[static_cast<std::size_t>(i) +
              static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
              static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                  static_cast<std::size_t>(ny)] =
              (u_ptr[lin + sx] - 2.0 * c + u_ptr[lin - sx]) * inv_dx2_x +
              (u_ptr[lin + sy] - 2.0 * c + u_ptr[lin - sy]) * inv_dx2_y +
              (u_ptr[lin + sz] - 2.0 * c + u_ptr[lin - sz]) * inv_dx2_z;
        }
      }
    }
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const std::size_t lin = (i + hw) * sx + (j + hw) * sy + (k + hw) * sz;
          const std::size_t lin_lap =
              static_cast<std::size_t>(i) +
              static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
              static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                  static_cast<std::size_t>(ny);
          u_ptr[lin] += dt * heat3d::kD * lap[lin_lap];
        }
      }
    }
  }
}

} // namespace

TEST_CASE("Scratch FD driver (bare loops, raw pointers): smoke + L2",
          "[heat3d][fd_scratch]") {
  using namespace pfc;
  constexpr int N = 16;
  const int hw = 1;
  const double dt = 1.0e-3;
  const int n_steps = 5;

  auto world = world::create(GridSize({N, N, N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, /*nproc=*/1);

  field::PaddedBrick<double> u(decomp, /*rank=*/0, hw);
  PaddedHaloExchanger<double> halo(decomp, /*rank=*/0, hw, MPI_COMM_WORLD);

  // From-scratch IC: exp(-r^2 / (4 kD)) by hand, exactly as the driver does.
  {
    const int nx = u.nx();
    const int ny = u.ny();
    const int nz = u.nz();
    const auto lower = u.lower_global();
    const auto origin = u.origin();
    const auto dx = u.spacing();
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const double x = origin[0] + (lower[0] + i) * dx[0];
          const double y = origin[1] + (lower[1] + j) * dx[1];
          const double z = origin[2] + (lower[2] + k) * dx[2];
          const double r2 = x * x + y * y + z * z;
          u(i, j, k) = std::exp(-r2 / (4.0 * heat3d::kD));
        }
      }
    }
  }

  double sum0 = 0.0;
  field::for_each_inner(
      u, hw, [&](int i, int j, int k) { sum0 += u(i, j, k) * u(i, j, k); });
  REQUIRE(sum0 > 0.0);

  run_scratch_loop_(u, halo, dt, n_steps);

  double sum1 = 0.0;
  field::for_each_inner(u, hw, [&](int i, int j, int k) {
    REQUIRE(std::isfinite(u(i, j, k)));
    sum1 += u(i, j, k) * u(i, j, k);
  });
  REQUIRE(sum1 < sum0); // diffusion dissipates
  REQUIRE(sum1 > 0.0);

  double l2sq = 0.0;
  double cnt = 0.0;
  const double t_final = static_cast<double>(n_steps) * dt;
  field::for_each_inner(u, hw, [&](int i, int j, int k) {
    const auto p = u.global_coords(i, j, k);
    const double r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    const double u_exact = heat3d::analytic_gaussian(r2, t_final, heat3d::kD);
    const double diff = u(i, j, k) - u_exact;
    l2sq += diff * diff;
    cnt += 1.0;
  });
  const double l2_rms = std::sqrt(l2sq / cnt);
  REQUIRE(std::isfinite(l2_rms));
  REQUIRE(l2_rms < 1.0e-3);
}

TEST_CASE("Scratch FD driver: produces same interior L2 as compact FdCpuStack "
          "path",
          "[heat3d][fd_scratch]") {
  using namespace pfc;
  constexpr int N = 16;
  const int hw = 1;
  const int order = 2;
  const double dt = 1.0e-3;
  const int n_steps = 5;

  HeatModel model;

  sim::stacks::FdCpuStack stack(GridSize({N, N, N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                                GridSpacing({1.0, 1.0, 1.0}), order, 0, 1,
                                MPI_COMM_WORLD);
  stack.u().apply(model.initial_condition);
  auto grad = field::create<heat3d::HeatGrads>(stack.u(), order);
  auto stepper = sim::steppers::create(stack.u(), grad, model, dt);
  for (int step = 0; step < n_steps; ++step) {
    stack.exchange_halos();
    (void)stepper.step(static_cast<double>(step) * dt, stack.u().vec());
  }
  double l2_compact = 0.0;
  double cnt = 0.0;
  const double t_final = static_cast<double>(n_steps) * dt;
  stack.u().for_each_interior([&](double x, double y, double z, double v) {
    const double u_exact =
        heat3d::analytic_gaussian(x * x + y * y + z * z, t_final, heat3d::kD);
    const double diff = v - u_exact;
    l2_compact += diff * diff;
    cnt += 1.0;
  });
  l2_compact = std::sqrt(l2_compact / cnt);

  auto world = world::create(GridSize({N, N, N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, 1);
  field::PaddedBrick<double> u(decomp, 0, hw);
  PaddedHaloExchanger<double> halo(decomp, 0, hw, MPI_COMM_WORLD);
  {
    const int nx = u.nx();
    const int ny = u.ny();
    const int nz = u.nz();
    const auto lower = u.lower_global();
    const auto origin = u.origin();
    const auto dx = u.spacing();
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const double x = origin[0] + (lower[0] + i) * dx[0];
          const double y = origin[1] + (lower[1] + j) * dx[1];
          const double z = origin[2] + (lower[2] + k) * dx[2];
          const double r2 = x * x + y * y + z * z;
          u(i, j, k) = std::exp(-r2 / (4.0 * heat3d::kD));
        }
      }
    }
  }

  run_scratch_loop_(u, halo, dt, n_steps);

  double l2_scratch = 0.0;
  double cnt2 = 0.0;
  field::for_each_inner(u, hw, [&](int i, int j, int k) {
    const auto p = u.global_coords(i, j, k);
    const double u_exact = heat3d::analytic_gaussian(
        p[0] * p[0] + p[1] * p[1] + p[2] * p[2], t_final, heat3d::kD);
    const double diff = u(i, j, k) - u_exact;
    l2_scratch += diff * diff;
    cnt2 += 1.0;
  });
  l2_scratch = std::sqrt(l2_scratch / cnt2);

  // Same physical 2nd-order central 7-point stencil and same IC; agreement
  // to round-off is the contract. Slop matches the manual-driver parity test.
  REQUIRE_THAT(l2_scratch, WithinAbs(l2_compact, 1.0e-7));
}

// -----------------------------------------------------------------------------
// Per-binary CLI parser tests — `parse_fd` / `parse_spectral` (pure, no MPI).
// -----------------------------------------------------------------------------

TEST_CASE("heat3d::parse_fd: no args returns nullopt", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d_fd")};
  REQUIRE_FALSE(heat3d::parse_fd(1, argv).has_value());
}

TEST_CASE("heat3d::parse_fd: happy path", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d_fd"), const_cast<char *>("64"),
                  const_cast<char *>("200"), const_cast<char *>("0.001"),
                  const_cast<char *>("8")};
  const auto cfg = heat3d::parse_fd(5, argv);
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->N == 64);
  REQUIRE(cfg->n_steps == 200);
  REQUIRE_THAT(cfg->dt, WithinAbs(0.001, 1e-15));
  REQUIRE(cfg->fd_order == 8);
}

TEST_CASE("heat3d::parse_fd: missing fd_order returns nullopt", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d_fd"), const_cast<char *>("64"),
                  const_cast<char *>("200"), const_cast<char *>("0.001")};
  REQUIRE_FALSE(heat3d::parse_fd(4, argv).has_value());
}

TEST_CASE("heat3d::parse_fd: rejects out-of-range values", "[heat3d][cli]") {
  SECTION("N too small") {
    char *argv[] = {const_cast<char *>("heat3d_fd"), const_cast<char *>("4"),
                    const_cast<char *>("10"), const_cast<char *>("0.01"),
                    const_cast<char *>("2")};
    REQUIRE_FALSE(heat3d::parse_fd(5, argv).has_value());
  }
  SECTION("fd_order odd") {
    char *argv[] = {const_cast<char *>("heat3d_fd"), const_cast<char *>("32"),
                    const_cast<char *>("100"), const_cast<char *>("0.01"),
                    const_cast<char *>("3")};
    REQUIRE_FALSE(heat3d::parse_fd(5, argv).has_value());
  }
  SECTION("fd_order out of range") {
    char *argv[] = {const_cast<char *>("heat3d_fd"), const_cast<char *>("32"),
                    const_cast<char *>("100"), const_cast<char *>("0.01"),
                    const_cast<char *>("22")};
    REQUIRE_FALSE(heat3d::parse_fd(5, argv).has_value());
  }
}

TEST_CASE("heat3d::parse_spectral: happy path", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d_spectral"), const_cast<char *>("32"),
                  const_cast<char *>("50"), const_cast<char *>("0.005")};
  const auto cfg = heat3d::parse_spectral(4, argv);
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->N == 32);
  REQUIRE(cfg->n_steps == 50);
  REQUIRE_THAT(cfg->dt, WithinAbs(0.005, 1e-15));
  REQUIRE(cfg->fd_order == 2);
}

TEST_CASE("heat3d::parse_spectral: insufficient args returns nullopt",
          "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d_spectral"), const_cast<char *>("32"),
                  const_cast<char *>("50")};
  REQUIRE_FALSE(heat3d::parse_spectral(3, argv).has_value());
}

TEST_CASE("heat3d::parse_spectral: rejects out-of-range values", "[heat3d][cli]") {
  SECTION("dt non-positive") {
    char *argv[] = {const_cast<char *>("heat3d_spectral"), const_cast<char *>("32"),
                    const_cast<char *>("10"), const_cast<char *>("0")};
    REQUIRE_FALSE(heat3d::parse_spectral(4, argv).has_value());
  }
  SECTION("n_steps non-positive") {
    char *argv[] = {const_cast<char *>("heat3d_spectral"), const_cast<char *>("32"),
                    const_cast<char *>("0"), const_cast<char *>("0.005")};
    REQUIRE_FALSE(heat3d::parse_spectral(4, argv).has_value());
  }
}

// -----------------------------------------------------------------------------
// Reporting helpers (analytic reference solution).
// -----------------------------------------------------------------------------

TEST_CASE("heat3d::analytic_gaussian: t=0 collapses to exp(-r^2/(4D))",
          "[heat3d][reporting]") {
  const double D = 1.5;
  const double r2 = 4.0; // |x| = 2
  REQUIRE_THAT(heat3d::analytic_gaussian(r2, /*t=*/0.0, D),
               WithinAbs(std::exp(-r2 / (4.0 * D)), 1e-15));
}

TEST_CASE("heat3d::analytic_gaussian: positive t damps the peak and "
          "spreads the support",
          "[heat3d][reporting]") {
  const double D = 1.0;
  const double t = 0.25;
  const double s = 1.0 + t;
  // Peak at the origin: amplitude is s^{-3/2}.
  REQUIRE_THAT(heat3d::analytic_gaussian(0.0, t, D),
               WithinAbs(std::pow(s, -1.5), 1e-15));
  // Off-origin matches the closed form.
  const double r2 = 1.0 + 4.0; // (1, 2, 0)
  REQUIRE_THAT(heat3d::analytic_gaussian(r2, t, D),
               WithinAbs(std::pow(s, -1.5) * std::exp(-r2 / (4.0 * D * s)), 1e-15));
}

TEST_CASE("heat3d::analytic_gaussian: t > 0 strictly decreases the peak",
          "[heat3d][reporting]") {
  // Solution to a linear diffusion equation: peak amplitude at the origin
  // is monotonically decreasing in t for any D > 0.
  const double D = 1.0;
  const double u0 = heat3d::analytic_gaussian(0.0, 0.0, D);
  const double u1 = heat3d::analytic_gaussian(0.0, 0.5, D);
  const double u2 = heat3d::analytic_gaussian(0.0, 2.0, D);
  REQUIRE(u0 > u1);
  REQUIRE(u1 > u2);
  REQUIRE(u2 > 0.0);
}

// -----------------------------------------------------------------------------
// MPI lifecycle around Catch2 (mirrors apps/tungsten/tests/test_tungsten.cpp).
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "test_heat3d: MPI_Init failed\n";
    return 1;
  }
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
