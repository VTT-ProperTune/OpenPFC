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
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/separated_halo_exchange.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/grad_point.hpp>
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

#include <heat3d/heat_model.hpp>

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
// `pfc::decomposition::Decomposition` stores a const reference to its source
// `pfc::World` (`m_global_world`), so the world has to outlive the
// decomposition in the same scope. We therefore construct both objects inline
// in each test rather than returning them from a helper (which would leave
// the decomposition with a dangling reference).
// -----------------------------------------------------------------------------

TEST_CASE("HeatModel + LocalField: u.apply samples the model IC",
          "[heat3d][LocalField]") {
  constexpr int N = 8;
  auto world = pfc::world::create(pfc::GridSize({N, N, N}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, /*nproc=*/1);

  HeatModel model;
  model.D = 1.5;

  pfc::field::LocalField<double> u =
      pfc::field::LocalField<double>::from_subdomain(decomp, /*rank=*/0,
                                                     /*halo_width=*/0);
  u.apply(model.initial_condition);

  SECTION("origin cell evaluates to exp(0) = 1") {
    REQUIRE_THAT(u(0, 0, 0), WithinAbs(1.0, 1e-12));
  }

  SECTION("interior cells match exp(-r^2/(4D)) for the configured D") {
    for (int iz = 0; iz < 4; ++iz) {
      for (int iy = 0; iy < 4; ++iy) {
        for (int ix = 0; ix < 4; ++ix) {
          const double x = static_cast<double>(ix);
          const double y = static_cast<double>(iy);
          const double z = static_cast<double>(iz);
          const double r2 = x * x + y * y + z * z;
          INFO("ix=" << ix << " iy=" << iy << " iz=" << iz);
          REQUIRE_THAT(u(ix, iy, iz),
                       WithinAbs(std::exp(-r2 / (4.0 * model.D)), 1e-12));
        }
      }
    }
  }
}

TEST_CASE("HeatModel + EulerStepper: one explicit-Euler FD step decreases the "
          "L2 norm of a Gaussian (heat dissipation)",
          "[heat3d][Euler]") {
  // Small but well-resolved single-rank grid; halo_width = order/2 = 1.
  constexpr int N = 16;
  const int order = 2;
  const int hw = order / 2;

  auto world = pfc::world::create(pfc::GridSize({N, N, N}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, /*nproc=*/1);

  HeatModel model;
  model.D = 1.0;

  pfc::field::LocalField<double> u =
      pfc::field::LocalField<double>::from_subdomain(decomp, /*rank=*/0, hw);
  u.apply(model.initial_condition);

  // L2 norm of the initial state (interior only).
  double sum0 = 0.0;
  u.for_each_interior([&sum0](double, double, double, double v) { sum0 += v * v; });

  auto face_halos = pfc::halo::allocate_face_halos<double>(decomp, /*rank=*/0, hw);
  pfc::SeparatedFaceHaloExchanger<double> exchanger(decomp, /*rank=*/0, hw,
                                                    MPI_COMM_WORLD);
  auto grad = pfc::field::create(u, order);
  pfc::sim::steppers::EulerStepper stepper(grad, model, /*dt=*/1.0e-3, u.size());

  // A handful of steps; with diffusion the field's energy must drop.
  for (int step = 0; step < 5; ++step) {
    exchanger.exchange_halos(u.data(), u.size(), face_halos);
    (void)stepper.step(static_cast<double>(step) * 1.0e-3, u.vec());
  }

  double sum1 = 0.0;
  u.for_each_interior([&sum1](double, double, double, double v) { sum1 += v * v; });

  REQUIRE(sum1 < sum0);
  // Sanity: the field shouldn't blow up either.
  REQUIRE(sum1 > 0.0);
  REQUIRE(std::isfinite(sum1));
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
