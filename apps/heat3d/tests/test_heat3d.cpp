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
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/local_field.hpp>
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

TEST_CASE("HeatModel: defaults", "[heat3d][HeatModel]") {
  HeatModel model;

  SECTION("D defaults to 1.0") { REQUIRE_THAT(model.D, WithinAbs(1.0, 1e-12)); }

  SECTION("default IC at the origin equals 1") {
    REQUIRE_THAT(model.initial_condition(0.0, 0.0, 0.0), WithinAbs(1.0, 1e-12));
  }

  SECTION("default IC away from origin matches exp(-r^2/(4D)) with D=1") {
    const double r2 = 1.0 + 4.0 + 9.0; // (1, 2, 3)
    REQUIRE_THAT(model.initial_condition(1.0, 2.0, 3.0),
                 WithinAbs(std::exp(-r2 / 4.0), 1e-12));
  }

  SECTION("default boundary_value is empty") {
    REQUIRE_FALSE(static_cast<bool>(model.boundary_value));
  }
}

TEST_CASE("HeatModel: IC tracks D after construction", "[heat3d][HeatModel]") {
  HeatModel model;
  const double r2 = 4.0; // (2, 0, 0)

  for (double D : {0.25, 0.5, 1.0, 2.0, 7.5}) {
    model.D = D;
    INFO("D = " << D);
    REQUIRE_THAT(model.initial_condition(2.0, 0.0, 0.0),
                 WithinAbs(std::exp(-r2 / (4.0 * D)), 1e-12));
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

TEST_CASE("HeatModel: rhs = D * (xx + yy + zz)", "[heat3d][HeatModel]") {
  HeatModel model;

  SECTION("zero Laplacian gives zero RHS regardless of D") {
    model.D = 17.0;
    heat3d::HeatGrads g{.xx = 0.0, .yy = 0.0, .zz = 0.0};
    REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(0.0, 1e-12));
  }

  SECTION("uniform unit Laplacian (D=1) gives RHS = 3") {
    model.D = 1.0;
    heat3d::HeatGrads g{.xx = 1.0, .yy = 1.0, .zz = 1.0};
    REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(3.0, 1e-12));
  }

  SECTION("rhs is linear in D") {
    heat3d::HeatGrads g{.xx = 1.0, .yy = -2.0, .zz = 0.5};
    const double lap = 1.0 - 2.0 + 0.5; // -0.5
    for (double D : {0.1, 1.0, 3.7}) {
      model.D = D;
      INFO("D = " << D);
      REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(D * lap, 1e-12));
    }
  }

  SECTION("rhs ignores t") {
    model.D = 2.0;
    heat3d::HeatGrads g{.xx = 1.0, .yy = 1.0, .zz = 1.0};
    REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(6.0, 1e-12));
    REQUIRE_THAT(model.rhs(1234.5, g), WithinAbs(6.0, 1e-12));
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
  model.D = 1.5;

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

  SECTION("owned cells match exp(-r^2/(4D)) for the configured D") {
    for (int iz = 0; iz < 4; ++iz) {
      for (int iy = 0; iy < 4; ++iy) {
        for (int ix = 0; ix < 4; ++ix) {
          const double x = static_cast<double>(ix);
          const double y = static_cast<double>(iy);
          const double z = static_cast<double>(iz);
          const double r2 = x * x + y * y + z * z;
          INFO("ix=" << ix << " iy=" << iy << " iz=" << iz);
          REQUIRE_THAT(stack.u()(ix, iy, iz),
                       WithinAbs(std::exp(-r2 / (4.0 * model.D)), 1e-12));
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
  model.D = 1.0;
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
// Per-binary CLI parser tests — `parse_fd` / `parse_spectral` (pure, no MPI).
// -----------------------------------------------------------------------------

TEST_CASE("heat3d::parse_fd: no args returns nullopt", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d_fd")};
  REQUIRE_FALSE(heat3d::parse_fd(1, argv).has_value());
}

TEST_CASE("heat3d::parse_fd: happy path", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d_fd"), const_cast<char *>("64"),
                  const_cast<char *>("200"),       const_cast<char *>("0.001"),
                  const_cast<char *>("2.5"),       const_cast<char *>("8")};
  const auto cfg = heat3d::parse_fd(6, argv);
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->N == 64);
  REQUIRE(cfg->n_steps == 200);
  REQUIRE_THAT(cfg->dt, WithinAbs(0.001, 1e-15));
  REQUIRE_THAT(cfg->D, WithinAbs(2.5, 1e-15));
  REQUIRE(cfg->fd_order == 8);
}

TEST_CASE("heat3d::parse_fd: missing fd_order returns nullopt", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d_fd"), const_cast<char *>("64"),
                  const_cast<char *>("200"), const_cast<char *>("0.001"),
                  const_cast<char *>("2.5")};
  REQUIRE_FALSE(heat3d::parse_fd(5, argv).has_value());
}

TEST_CASE("heat3d::parse_fd: rejects out-of-range values", "[heat3d][cli]") {
  SECTION("N too small") {
    char *argv[] = {const_cast<char *>("heat3d_fd"), const_cast<char *>("4"),
                    const_cast<char *>("10"),        const_cast<char *>("0.01"),
                    const_cast<char *>("1.0"),       const_cast<char *>("2")};
    REQUIRE_FALSE(heat3d::parse_fd(6, argv).has_value());
  }
  SECTION("fd_order odd") {
    char *argv[] = {const_cast<char *>("heat3d_fd"), const_cast<char *>("32"),
                    const_cast<char *>("100"),       const_cast<char *>("0.01"),
                    const_cast<char *>("1.0"),       const_cast<char *>("3")};
    REQUIRE_FALSE(heat3d::parse_fd(6, argv).has_value());
  }
  SECTION("fd_order out of range") {
    char *argv[] = {const_cast<char *>("heat3d_fd"), const_cast<char *>("32"),
                    const_cast<char *>("100"),       const_cast<char *>("0.01"),
                    const_cast<char *>("1.0"),       const_cast<char *>("22")};
    REQUIRE_FALSE(heat3d::parse_fd(6, argv).has_value());
  }
}

TEST_CASE("heat3d::parse_spectral: happy path", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d_spectral"), const_cast<char *>("32"),
                  const_cast<char *>("50"), const_cast<char *>("0.005"),
                  const_cast<char *>("1.0")};
  const auto cfg = heat3d::parse_spectral(5, argv);
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->N == 32);
  REQUIRE(cfg->n_steps == 50);
  REQUIRE_THAT(cfg->dt, WithinAbs(0.005, 1e-15));
  REQUIRE_THAT(cfg->D, WithinAbs(1.0, 1e-15));
  REQUIRE(cfg->fd_order == 2);
}

TEST_CASE("heat3d::parse_spectral: insufficient args returns nullopt",
          "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d_spectral"), const_cast<char *>("32"),
                  const_cast<char *>("50"), const_cast<char *>("0.005")};
  REQUIRE_FALSE(heat3d::parse_spectral(4, argv).has_value());
}

TEST_CASE("heat3d::parse_spectral: rejects out-of-range values", "[heat3d][cli]") {
  SECTION("dt non-positive") {
    char *argv[] = {const_cast<char *>("heat3d_spectral"), const_cast<char *>("32"),
                    const_cast<char *>("10"), const_cast<char *>("0"),
                    const_cast<char *>("1.0")};
    REQUIRE_FALSE(heat3d::parse_spectral(5, argv).has_value());
  }
  SECTION("D non-positive") {
    char *argv[] = {const_cast<char *>("heat3d_spectral"), const_cast<char *>("32"),
                    const_cast<char *>("10"), const_cast<char *>("0.005"),
                    const_cast<char *>("-1.0")};
    REQUIRE_FALSE(heat3d::parse_spectral(5, argv).has_value());
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
