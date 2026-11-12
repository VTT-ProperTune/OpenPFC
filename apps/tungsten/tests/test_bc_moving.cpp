// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <tungsten/moving_bc.hpp>

using namespace pfc;
using Catch::Approx;
using tungsten::MovingBC;

TEST_CASE("MovingBC rejects incomplete or invalid restart state", "[bc_moving]") {
  MovingBC bc(0.0, 1.0);
  const nlohmann::json valid = {{"xpos", 45.0}, {"idx", 33}, {"first", false}};
  REQUIRE_NOTHROW(bc.restore_checkpoint_state(valid));
  REQUIRE(bc.checkpoint_state() == valid);
  for (const auto &key : {"xpos", "idx", "first"}) {
    auto missing = valid;
    missing.erase(key);
    REQUIRE_THROWS_AS(bc.restore_checkpoint_state(missing), std::invalid_argument);
  }
  for (const auto &idx :
       {nlohmann::json(-1), nlohmann::json(1.5),
        nlohmann::json(std::numeric_limits<unsigned long long>::max())}) {
    auto invalid = valid;
    invalid["idx"] = idx;
    REQUIRE_THROWS_AS(bc.restore_checkpoint_state(invalid), std::invalid_argument);
  }
  auto invalid = valid;
  invalid["xpos"] = std::numeric_limits<double>::infinity();
  REQUIRE_THROWS_AS(bc.restore_checkpoint_state(invalid), std::invalid_argument);
  REQUIRE(bc.checkpoint_state() == valid);
}

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

namespace {

// 32 columns of 4x4 cells: solid (columns 0..11) oscillates +-amp around
// rho_high, liquid (12..31) sits at rho_low. The solid amplitude is so low
// that no sample reaches the historical 0.1 cutoff.
struct LowAmplitudeFront {
  static constexpr double rho_low = -0.46;
  static constexpr double rho_high = -0.10;
  static constexpr double amp = 0.05;
  static constexpr int front_index = 11;
  pfc::Domain domain = pfc::domain::create(pfc::GridSize({32, 4, 4}),
                                           pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                           pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::Box3i box = pfc::domain::index_box(domain);
  std::vector<double> psi;

  LowAmplitudeFront() : psi(static_cast<size_t>(box.count()), 0.0) {
    const int nx = box.size[0];
    long idx = 0;
    for (int k = box.low[2]; k <= box.high[2]; k++) {
      for (int j = box.low[1]; j <= box.high[1]; j++) {
        for (int i = 0; i < nx; i++) {
          const bool solid = i <= front_index;
          const double sign = ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
          psi[idx++] = solid ? rho_high + sign * amp : rho_low;
        }
      }
    }
  }

  MovingBC tracker() const {
    MovingBC bc(rho_low, rho_high);
    bc.set_disp(0.0);
    bc.set_xpos(0.0);
    bc.set_xwidth(1.0);
    return bc;
  }
};

} // namespace

TEST_CASE("MovingBC tracks a low-amplitude front by its mean level", "[bc_moving]") {
  LowAmplitudeFront f;

  SECTION(
      "the default amplitude tracker finds it, relative to the strongest column") {
    MovingBC bc = f.tracker();
    REQUIRE(bc.get_tracker() == tungsten::FrontTracker::amplitude);
    REQUIRE_FALSE(bc.has_explicit_threshold());
    REQUIRE(bc.get_threshold() == Approx(0.5));
    bc.apply(f.psi, f.domain, f.box);
    REQUIRE(bc.get_xpos() == Approx(double(f.front_index)));
  }

  SECTION("amplitude ignores the melt level: a shifted field still tracks") {
    MovingBC bc = f.tracker();
    for (auto &v : f.psi) v += 5.0;
    bc.apply(f.psi, f.domain, f.box);
    REQUIRE(bc.get_xpos() == Approx(double(f.front_index)));
  }

  SECTION("an amplitude fraction above one marks nothing solid") {
    MovingBC bc = f.tracker();
    bc.set_threshold(1.5);
    REQUIRE(bc.has_explicit_threshold());
    bc.apply(f.psi, f.domain, f.box);
    REQUIRE(bc.get_xpos() == Approx(0.0));
  }

  SECTION("a uniform field has no crystal and no front") {
    MovingBC bc = f.tracker();
    std::fill(f.psi.begin(), f.psi.end(), f.rho_high);
    bc.apply(f.psi, f.domain, f.box);
    REQUIRE(bc.get_xpos() == Approx(0.0));
  }

  SECTION("mean_field finds it from the density jump at the midpoint") {
    MovingBC bc = f.tracker();
    bc.set_tracker(tungsten::FrontTracker::mean_field);
    REQUIRE(bc.get_threshold() == Approx(0.5 * (f.rho_low + f.rho_high)));
    bc.apply(f.psi, f.domain, f.box);
    REQUIRE(bc.get_xpos() == Approx(double(f.front_index)));
  }

  SECTION("the historical cutoff cannot see this front") {
    MovingBC bc = f.tracker();
    bc.set_tracker(tungsten::FrontTracker::cutoff);
    REQUIRE(bc.get_threshold() == Approx(0.1));
    bc.apply(f.psi, f.domain, f.box);
    REQUIRE(bc.get_xpos() == Approx(0.0));
  }
}

TEST_CASE("MovingBC front tracker names round-trip", "[bc_moving]") {
  for (const auto mode :
       {tungsten::FrontTracker::cutoff, tungsten::FrontTracker::mean_field,
        tungsten::FrontTracker::amplitude}) {
    REQUIRE(tungsten::front_tracker_from_string(tungsten::to_string(mode)) == mode);
  }
  REQUIRE_THROWS_AS(tungsten::front_tracker_from_string("peak"),
                    std::invalid_argument);
}
