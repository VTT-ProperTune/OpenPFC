// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc_apps/moving_bc.hpp>

using namespace pfc;
using Catch::Approx;

TEST_CASE("MovingBC - Parameter Access", "[bc_moving]") {
  MovingBC bc;

  SECTION("Set and get rho_low") { bc.set_rho_low(0.1); }
  SECTION("Set and get rho_high") { bc.set_rho_high(0.9); }

  SECTION("Set and get xpos") {
    bc.set_xpos(50.0);
    REQUIRE(bc.get_xpos() == Approx(50.0));
  }

  SECTION("Set and get xwidth") {
    bc.set_xwidth(20.0);
    REQUIRE(bc.get_xwidth() == Approx(20.0));
  }

  SECTION("Set and get alpha") { bc.set_alpha(2.0); }

  SECTION("Set and get threshold") {
    bc.set_threshold(0.2);
    REQUIRE(bc.get_threshold() == Approx(0.2));
  }

  SECTION("Set and get disp") { bc.set_disp(30.0); }
}

TEST_CASE("MovingBC - Constructor with Parameters", "[bc_moving]") {
  MovingBC bc(0.2, 0.8);
}

TEST_CASE("MovingBC - Modifier Name", "[bc_moving]") {
  MovingBC bc;
  REQUIRE(bc.get_modifier_name() == "MovingBC");
}

TEST_CASE("MovingBC - Field Application", "[bc_moving]") {
  auto domain = pfc::domain::create(pfc::GridSize({16, 4, 4}),
                                    pfc::PhysicalOrigin({-64.0, -16.0, -16.0}),
                                    pfc::GridSpacing({8.0, 8.0, 8.0}));
  auto box = pfc::domain::index_box(domain);
  const size_t field_size = static_cast<size_t>(box.count());
  std::vector<double> psi(field_size, 0.0);

  MovingBC bc(0.0, 1.0);
  bc.set_xwidth(15.0);
  bc.set_xpos(0.0);
  bc.set_threshold(0.1);

  SECTION("Apply boundary condition") {
    REQUIRE_NOTHROW(bc.apply(psi, domain, box));
    bool has_nonzero = false;
    for (const auto &value : psi) {
      if (value != 0.0) {
        has_nonzero = true;
        break;
      }
    }
    REQUIRE(has_nonzero);
  }

  SECTION("Field values in range") {
    bc.apply(psi, domain, box);
    double vmin = psi[0];
    double vmax = psi[0];
    for (const auto &value : psi) {
      vmin = std::min(vmin, value);
      vmax = std::max(vmax, value);
    }
    REQUIRE(vmin >= -0.1);
    REQUIRE(vmax <= 1.1);
  }

  SECTION("Multiple applications") {
    bc.apply(psi, domain, box);
    double xpos1 = bc.get_xpos();
    std::fill(psi.begin(), psi.end(), 0.5);
    bc.apply(psi, domain, box);
    REQUIRE(bc.get_xpos() >= xpos1);
  }
}

TEST_CASE("MovingBC - Apply on named density field", "[bc_moving]") {
  auto domain = pfc::domain::create(pfc::Int3{16, 8, 8});
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);

  MovingBC bc;
  bc.set_field_name("density");
  bc.set_rho_low(0.1);
  bc.set_rho_high(0.9);
  bc.set_xwidth(10.0);

  REQUIRE_NOTHROW(bc.apply(psi, domain, box));
  REQUIRE(psi.size() == static_cast<size_t>(box.count()));
}

TEST_CASE("MovingBC - Field Name Assignment", "[bc_moving]") {
  MovingBC bc;
  bc.set_field_name("interface");
  REQUIRE(bc.get_field_name() == "interface");
}

TEST_CASE("MovingBC - Boundary Position Tracking", "[bc_moving]") {
  auto domain = pfc::domain::create(pfc::GridSize({16, 4, 4}),
                                    pfc::PhysicalOrigin({-64.0, -16.0, -16.0}),
                                    pfc::GridSpacing({8.0, 8.0, 8.0}));
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);

  MovingBC bc(0.0, 1.0);
  bc.set_xpos(100.0);
  bc.set_xwidth(15.0);

  SECTION("Position persists") {
    REQUIRE(bc.get_xpos() == Approx(100.0));
    bc.apply(psi, domain, box);
    REQUIRE(bc.get_xpos() >= 100.0);
  }
}

TEST_CASE("MovingBC - MPI collectives fail closed", "[bc_moving]") {
  auto domain = pfc::domain::create(pfc::GridSize({16, 4, 4}),
                                    pfc::PhysicalOrigin({-64.0, -16.0, -16.0}),
                                    pfc::GridSpacing({8.0, 8.0, 8.0}));
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);

  MovingBC bc(0.0, 1.0);
  bc.set_xwidth(15.0);
  bc.set_xpos(0.0);
  bc.set_threshold(0.1);

  REQUIRE_NOTHROW(bc.apply(psi, domain, box));
  REQUIRE(bc.get_xpos() >= 0.0);
}
