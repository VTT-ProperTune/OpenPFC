// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/field.hpp>
#include <openpfc/kernel/data/world.hpp>

using namespace pfc;

static double gaussian(Real3 r) {
  return std::exp(-(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]));
}

// Regression (audit 4.10 / PL): Field stores the World by value, so a Field
// may safely outlive the World object it was constructed from. A helper that
// returns a Field built from a local World would dangle under the old
// `const World&` member.
static field::Field<double> make_field_from_local_world() {
  auto local_world =
      world::create(GridSize(Int3{6, 5, 4}), PhysicalOrigin(Real3{1.0, 2.0, 3.0}),
                    GridSpacing(Real3{0.5, 0.25, 0.125}));
  return field::create<double>(local_world);
}

TEST_CASE("Field keeps a valid World after its source World is gone", "[field]") {
  auto f = make_field_from_local_world(); // source World destroyed here
  const auto &w = field::get_world(f);
  REQUIRE(get_size(w) == Int3{6, 5, 4});
  REQUIRE(get_spacing(w) == Real3{0.5, 0.25, 0.125});
  REQUIRE(get_origin(w) == Real3{1.0, 2.0, 3.0});
  REQUIRE(field::get_data(f).size() == get_total_size(w));
}

TEST_CASE("Field", "[field]") {
  Int3 size = {8, 8, 8};
  auto world = world::create(size);
  auto f = field::create<double>(world);

  SECTION("Field has correct size") {
    const auto &data = field::get_data(f);
    REQUIRE(data.size() == get_total_size(world));
  }

  SECTION("Field is associated with correct world") {
    const auto &w = field::get_world(f);
    REQUIRE(get_size(w) == get_size(world));
    REQUIRE(get_origin(w) == get_origin(world));
    REQUIRE(get_spacing(w) == get_spacing(world));
  }

  SECTION("Field values can be set and read") {
    auto &data = field::get_data(f);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<double>(i);
    }
    bool values_match = true;
    for (size_t i = 0; i < data.size(); ++i)
      values_match &= data[i] == static_cast<double>(i);
    REQUIRE(values_match);
  }

  SECTION("Field can be move-constructed") {
    auto original_size = field::get_data(f).size();
    auto f2 = std::move(f);
    REQUIRE(field::get_data(f2).size() == original_size);
  }

  SECTION("Field can be constructed from user-provided data (moved)") {
    std::vector<double> data(get_total_size(world), 42.0);
    auto f2 = field::create<double>(world, std::move(data));
    const auto &d2 = field::get_data(f2);
    REQUIRE(d2.size() == get_total_size(world));
    bool values_match = true;
    for (auto val : d2) values_match &= val == 42.0;
    REQUIRE(values_match);
  }

  SECTION("Field can be constructed from user-provided data (copied)") {
    std::vector<double> data(get_total_size(world), 13.0);
    auto f3 = field::create<double>(world, data); // copy
    const auto &d3 = field::get_data(f3);
    REQUIRE(d3.size() == get_total_size(world));
    bool values_match = true;
    for (auto val : d3) values_match &= val == 13.0;
    REQUIRE(values_match);
  }

  SECTION("Field can be constructed from lambda") {
    auto func = [](Real3 r) { return r[0] + r[1] + r[2]; };
    auto f4 = field::create<double>(world, func);
    const auto &data = field::get_data(f4);
    bool values_are_finite = true;
    for (auto v : data) values_are_finite &= std::isfinite(v);
    REQUIRE(values_are_finite);
  }

  SECTION("Field can be filled using apply") {
    auto func = [](Real3 r) { return std::sin(r[0] * r[1] * r[2]); };
    field::apply(f, func);
    const auto &data = field::get_data(f);
    bool values_in_range = true;
    for (auto v : data) values_in_range &= v >= -1.0 && v <= 1.0;
    REQUIRE(values_in_range);
  }

  SECTION("Field can be filled using named function") {
    auto f5 = field::create<double>(world, gaussian);
    const auto &data = field::get_data(f5);
    bool values_in_range = true;
    for (auto v : data) values_in_range &= std::isfinite(v) && v >= 0.0 && v <= 1.0;
    REQUIRE(values_in_range);
  }

  SECTION("Field can be accessed using operator[]") {
    auto g = field::create<double>(world, [](auto r) { return r[0]; });
    auto d = field::create<double>(world, [](auto r) { return r[1]; });
    auto out = field::create<double>(world);

    double c = 2.0;
    for (auto i : field::indices(out)) {
      out[i] = g[i] + c * std::exp(d[i]);
    }

    bool values_match = true;
    for (size_t i = 0; i < get_data(out).size(); ++i)
      values_match &= std::abs(out[i] - (g[i] + c * std::exp(d[i]))) <= 1e-12;
    REQUIRE(values_match);
  }
}
