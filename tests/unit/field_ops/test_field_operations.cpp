// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <numeric>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/field/legacy_adapter.hpp"
#include "openpfc/field/operations.hpp"
#include "openpfc/model.hpp"

using namespace pfc;
using Catch::Approx;

namespace {
class DummyModel : public Model {
public:
  DummyModel(FFT &fft, const World &world) : Model(fft, world) {}
  void step(double) override {}
  void initialize(double) override {}
};
} // namespace

TEST_CASE("field::apply sets constant value over inbox", "[field_ops][unit]") {
  auto world = world::create(GridSize({8, 4, 2}));
  auto decomp = decomposition::create(world, 1);
  auto fft = fft::create(decomp);

  DummyModel model(fft, world);

  std::vector<double> u(fft.size_inbox(), 0.0);
  model.add_real_field("psi", u);

  field::apply(model, "psi", [](const Real3 & /*x*/) { return 0.5; });

  const auto &ref = model.get_real_field("psi");
  for (const auto &val : ref) {
    REQUIRE(val == Approx(0.5));
  }
}

TEST_CASE("field::apply_with_time uses time parameter", "[field_ops][unit]") {
  auto world = world::create(GridSize({4, 4, 1}));
  auto decomp = decomposition::create(world, 1);
  auto fft = fft::create(decomp);

  DummyModel model(fft, world);
  std::vector<double> u(fft.size_inbox(), 0.0);
  model.add_real_field("psi", u);

  field::apply_with_time(model, "psi", /*t=*/2.0,
                         [](const Real3 & /*x*/, double t) { return 1.0 + t; });

  const auto &ref = model.get_real_field("psi");
  for (const auto &val : ref) {
    REQUIRE(val == Approx(3.0));
  }
}

TEST_CASE("field::apply_inplace modifies field based on current value",
          "[field_ops][unit]") {
  auto world = world::create(GridSize({4, 2, 2}));
  auto decomp = decomposition::create(world, 1);
  auto fft = fft::create(decomp);

  DummyModel model(fft, world);
  std::vector<double> u(fft.size_inbox(), 1.0);
  model.add_real_field("psi", u);

  // Double all values
  field::apply_inplace(model, "psi", [](const Real3 & /*x*/, double current) {
    return 2.0 * current;
  });

  const auto &ref = model.get_real_field("psi");
  for (const auto &val : ref) {
    REQUIRE(val == Approx(2.0));
  }
}

TEST_CASE("field::apply_inplace selective update preserves untouched cells",
          "[field_ops][unit]") {
  auto world = world::create(GridSize({8, 1, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, 1);
  auto fft = fft::create(decomp);

  DummyModel model(fft, world);
  std::vector<double> u(fft.size_inbox(), 0.0);
  model.add_real_field("psi", u);

  // Set values only where x > 4.0
  field::apply_inplace(model, "psi", [](const Real3 &x, double current) {
    if (x[0] > 4.0) {
      return 1.0;
    }
    return current;
  });

  const auto &ref = model.get_real_field("psi");
  // Verify some cells are 0.0 (untouched) and some are 1.0 (modified)
  bool has_zero = false;
  bool has_one = false;
  for (const auto &val : ref) {
    if (val == Approx(0.0)) has_zero = true;
    if (val == Approx(1.0)) has_one = true;
  }
  REQUIRE(has_zero);
  REQUIRE(has_one);
}

TEST_CASE("field::apply_inplace_with_time uses time parameter",
          "[field_ops][unit]") {
  auto world = world::create(GridSize({4, 2, 1}));
  auto decomp = decomposition::create(world, 1);
  auto fft = fft::create(decomp);

  DummyModel model(fft, world);
  std::vector<double> u(fft.size_inbox(), 1.0);
  model.add_real_field("psi", u);

  // Blend current value with time-dependent term
  field::apply_inplace_with_time(
      model, "psi", /*t=*/2.0,
      [](const Real3 & /*x*/, double current, double t) { return current + t; });

  const auto &ref = model.get_real_field("psi");
  for (const auto &val : ref) {
    REQUIRE(val == Approx(3.0)); // 1.0 + 2.0
  }
}

TEST_CASE("legacy adapter wraps lambda into FieldModifier", "[field_ops][unit]") {
  auto world = world::create(GridSize({8, 1, 1}));
  auto decomp = decomposition::create(world, 1);
  auto fft = fft::create(decomp);
  DummyModel model(fft, world);

  std::vector<double> u(fft.size_inbox(), 0.0);
  model.add_real_field("default", u);

  auto mod = field::make_legacy_modifier("default",
                                         [](const Real3 & /*x*/) { return 42.0; });

  mod->apply(model, /*t=*/0.0);

  const auto &ref = model.get_real_field("default");
  for (const auto &val : ref) {
    REQUIRE(val == Approx(42.0));
  }
}
