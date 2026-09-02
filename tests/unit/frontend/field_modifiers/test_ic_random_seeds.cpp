// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/initial_conditions/random_seeds.hpp>

using namespace pfc;
using Catch::Approx;

TEST_CASE("RandomSeeds - Parameter Access", "[ic_random_seeds]") {
  RandomSeeds seeds;

  SECTION("Set and get amplitude") {
    seeds.set_amplitude(0.25);
    REQUIRE(seeds.get_amplitude() == Approx(0.25));
  }

  SECTION("Set and get density") {
    seeds.set_density(0.55);
    REQUIRE(seeds.get_density() == Approx(0.55));
  }
}

TEST_CASE("RandomSeeds - Field Application", "[ic_random_seeds]") {
  auto domain = pfc::domain::create(pfc::GridSize({32, 32, 32}),
                                    pfc::PhysicalOrigin({-128.0, -128.0, -128.0}),
                                    pfc::GridSpacing({8.0, 8.0, 8.0}));
  auto box = pfc::domain::index_box(domain);
  const size_t field_size = static_cast<size_t>(box.count());
  std::vector<double> psi(field_size, 0.0);

  RandomSeeds seeds;
  seeds.set_amplitude(0.2);
  seeds.set_density(0.5);

  SECTION("Apply to field") {
    seeds.apply(psi, domain, box);
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
    seeds.apply(psi, domain, box);
    double max_expected = seeds.get_density() + 12.0 * seeds.get_amplitude();
    double min_expected = seeds.get_density() - 12.0 * seeds.get_amplitude();
    bool values_in_range = true;
    for (const auto &value : psi) {
      if (value != 0.0) {
        values_in_range &= value >= min_expected - 0.1;
        values_in_range &= value <= max_expected + 0.1;
      }
    }
    REQUIRE(values_in_range);
  }

  SECTION("Deterministic with fixed seed") {
    seeds.apply(psi, domain, box);
    Field field1 = psi;
    std::vector<double> psi2(field_size, 0.0);
    seeds.apply(psi2, domain, box);
    REQUIRE(field1 == psi2);
  }
}

TEST_CASE("RandomSeeds - Apply on named density field", "[ic_random_seeds]") {
  auto domain = pfc::domain::create(pfc::GridSize({16, 16, 16}),
                                    pfc::PhysicalOrigin({-128.0, -128.0, -128.0}),
                                    pfc::GridSpacing({16.0, 16.0, 16.0}));
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);

  RandomSeeds seeds;
  seeds.set_field_name("density");
  seeds.set_amplitude(0.15);
  seeds.set_density(0.65);

  REQUIRE_NOTHROW(seeds.apply(psi, domain, box));
  REQUIRE(psi.size() == static_cast<size_t>(box.count()));
}
