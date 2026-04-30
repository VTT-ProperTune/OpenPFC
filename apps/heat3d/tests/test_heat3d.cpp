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
#include <openpfc/kernel/field/grad_point.hpp>
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

#include <heat3d/cli.hpp>
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

TEST_CASE("HeatModel: rhs = D * (uxx + uyy + uzz)", "[heat3d][HeatModel]") {
  HeatModel model;

  SECTION("zero Laplacian gives zero RHS regardless of D and u") {
    model.D = 17.0;
    pfc::field::GradPoint g{42.0, 0.0, 0.0, 0.0};
    REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(0.0, 1e-12));
  }

  SECTION("uniform unit Laplacian (D=1) gives RHS = 3") {
    model.D = 1.0;
    pfc::field::GradPoint g{0.0, 1.0, 1.0, 1.0};
    REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(3.0, 1e-12));
  }

  SECTION("rhs is linear in D") {
    pfc::field::GradPoint g{0.0, 1.0, -2.0, 0.5};
    const double lap = 1.0 - 2.0 + 0.5; // -0.5
    for (double D : {0.1, 1.0, 3.7}) {
      model.D = D;
      INFO("D = " << D);
      REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(D * lap, 1e-12));
    }
  }

  SECTION("rhs ignores t and the bare value u") {
    model.D = 2.0;
    pfc::field::GradPoint g{99.0, 1.0, 1.0, 1.0}; // u=99 should not matter
    REQUIRE_THAT(model.rhs(0.0, g), WithinAbs(6.0, 1e-12));
    REQUIRE_THAT(model.rhs(1234.5, g), WithinAbs(6.0, 1e-12));
  }
}

// -----------------------------------------------------------------------------
// Integration tests against OpenPFC primitives (single MPI rank).
//
// `pfc::sim::stacks::FdCpuStack` is the recommended bundle: it owns the
// `World`, `Decomposition`, `LocalField`, halo buffers, and exchanger in the
// correct declaration order so internal references (e.g.
// `Decomposition::m_global_world`) stay valid for the lifetime of the stack.
// We use it here as the canonical "one statement to set up an FD-on-CPU
// solver" entry point — exactly what `apps/heat3d/src/cpu/heat3d.cpp` uses.
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

  auto grad = pfc::field::create(stack.u(), order);
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
// CLI parser tests (pure, no MPI).
// -----------------------------------------------------------------------------

TEST_CASE("heat3d::parse: no args returns nullopt", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d")};
  REQUIRE_FALSE(heat3d::parse(1, argv).has_value());
}

TEST_CASE("heat3d::parse: unknown method returns nullopt", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d"), const_cast<char *>("bogus"),
                  const_cast<char *>("32"),     const_cast<char *>("100"),
                  const_cast<char *>("0.01"),   const_cast<char *>("1.0")};
  REQUIRE_FALSE(heat3d::parse(6, argv).has_value());
}

TEST_CASE("heat3d::parse: fd needs 7 positional args (method, 4 numbers, order)",
          "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d"), const_cast<char *>("fd"),
                  const_cast<char *>("32"),     const_cast<char *>("100"),
                  const_cast<char *>("0.01"),   const_cast<char *>("1.0")};
  // 6 args (missing fd_order) -> nullopt
  REQUIRE_FALSE(heat3d::parse(6, argv).has_value());
}

TEST_CASE("heat3d::parse: fd happy path", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d"), const_cast<char *>("fd"),
                  const_cast<char *>("64"),     const_cast<char *>("200"),
                  const_cast<char *>("0.001"),  const_cast<char *>("2.5"),
                  const_cast<char *>("8")};
  const auto cfg = heat3d::parse(7, argv);
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->method == heat3d::Method::Fd);
  REQUIRE(cfg->N == 64);
  REQUIRE(cfg->n_steps == 200);
  REQUIRE_THAT(cfg->dt, WithinAbs(0.001, 1e-15));
  REQUIRE_THAT(cfg->D, WithinAbs(2.5, 1e-15));
  REQUIRE(cfg->fd_order == 8);
}

TEST_CASE("heat3d::parse: spectral happy path (no fd_order)", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d"), const_cast<char *>("spectral"),
                  const_cast<char *>("32"),     const_cast<char *>("50"),
                  const_cast<char *>("0.005"),  const_cast<char *>("1.0")};
  const auto cfg = heat3d::parse(6, argv);
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->method == heat3d::Method::Spectral);
  REQUIRE(cfg->N == 32);
  REQUIRE(cfg->n_steps == 50);
}

TEST_CASE("heat3d::parse: spectral_pw happy path", "[heat3d][cli]") {
  char *argv[] = {const_cast<char *>("heat3d"), const_cast<char *>("spectral_pw"),
                  const_cast<char *>("32"),     const_cast<char *>("10"),
                  const_cast<char *>("0.01"),   const_cast<char *>("1.0")};
  const auto cfg = heat3d::parse(6, argv);
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->method == heat3d::Method::SpectralPointwise);
}

TEST_CASE("heat3d::parse: rejects out-of-range values", "[heat3d][cli]") {
  SECTION("N too small") {
    char *argv[] = {const_cast<char *>("heat3d"), const_cast<char *>("spectral"),
                    const_cast<char *>("4"), // < 8
                    const_cast<char *>("10"),     const_cast<char *>("0.01"),
                    const_cast<char *>("1.0")};
    REQUIRE_FALSE(heat3d::parse(6, argv).has_value());
  }
  SECTION("dt non-positive") {
    char *argv[] = {const_cast<char *>("heat3d"), const_cast<char *>("spectral"),
                    const_cast<char *>("32"),     const_cast<char *>("10"),
                    const_cast<char *>("0"), // <= 0
                    const_cast<char *>("1.0")};
    REQUIRE_FALSE(heat3d::parse(6, argv).has_value());
  }
  SECTION("D non-positive") {
    char *argv[] = {const_cast<char *>("heat3d"), const_cast<char *>("spectral"),
                    const_cast<char *>("32"),     const_cast<char *>("10"),
                    const_cast<char *>("0.01"),   const_cast<char *>("-1.0")};
    REQUIRE_FALSE(heat3d::parse(6, argv).has_value());
  }
  SECTION("fd_order odd") {
    char *argv[] = {const_cast<char *>("heat3d"), const_cast<char *>("fd"),
                    const_cast<char *>("32"),     const_cast<char *>("100"),
                    const_cast<char *>("0.01"),   const_cast<char *>("1.0"),
                    const_cast<char *>("3")}; // odd
    REQUIRE_FALSE(heat3d::parse(7, argv).has_value());
  }
  SECTION("fd_order out of range") {
    char *argv[] = {const_cast<char *>("heat3d"), const_cast<char *>("fd"),
                    const_cast<char *>("32"),     const_cast<char *>("100"),
                    const_cast<char *>("0.01"),   const_cast<char *>("1.0"),
                    const_cast<char *>("22")}; // > 20
    REQUIRE_FALSE(heat3d::parse(7, argv).has_value());
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
