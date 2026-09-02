// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/initial_conditions/single_seed.hpp>

using namespace pfc;
using Catch::Approx;

TEST_CASE("SingleSeed - Parameter Access", "[ic_single_seed]") {
  SingleSeed seed;

  SECTION("Set and get amplitude") {
    seed.set_amplitude(0.3);
    REQUIRE(seed.get_amplitude() == Approx(0.3));
  }

  SECTION("Set and get density") {
    seed.set_density(0.7);
    REQUIRE(seed.get_density() == Approx(0.7));
  }
}

TEST_CASE("SingleSeed - Field Application", "[ic_single_seed]") {
  auto domain = pfc::domain::create(pfc::GridSize({16, 16, 16}),
                                    pfc::PhysicalOrigin({-128.0, -128.0, -128.0}),
                                    pfc::GridSpacing({16.0, 16.0, 16.0}));
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);

  SingleSeed seed;
  seed.set_amplitude(0.2);
  seed.set_density(0.5);

  SECTION("Apply to field") {
    seed.apply(psi, domain, box);

    bool has_nonzero = false;
    for (const auto &value : psi) {
      if (value != 0.0) {
        has_nonzero = true;
        break;
      }
    }
    REQUIRE(has_nonzero);
  }

  SECTION("Field values inside seed") {
    seed.apply(psi, domain, box);

    double max_expected = seed.get_density() + 12.0 * seed.get_amplitude();
    double min_expected = seed.get_density() - 12.0 * seed.get_amplitude();

    bool values_in_range = true;
    for (const auto &value : psi) {
      if (value != 0.0) {
        values_in_range &= value >= min_expected;
        values_in_range &= value <= max_expected;
      }
    }
    REQUIRE(values_in_range);
  }
}

TEST_CASE("SingleSeed - Apply on named density field", "[ic_single_seed]") {
  auto domain = pfc::domain::create(pfc::GridSize({8, 8, 8}),
                                    pfc::PhysicalOrigin({-64.0, -64.0, -64.0}),
                                    pfc::GridSpacing({16.0, 16.0, 16.0}));
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);

  SingleSeed seed;
  seed.set_field_name("density");
  seed.set_amplitude(0.1);
  seed.set_density(0.6);

  REQUIRE_NOTHROW(seed.apply(psi, domain, box));
  REQUIRE(psi.size() == static_cast<size_t>(box.count()));
}

TEST_CASE("SingleSeed - Field Name Assignment", "[ic_single_seed]") {
  SingleSeed seed;
  seed.set_field_name("custom_field");
  REQUIRE(seed.get_field_name() == "custom_field");
}
