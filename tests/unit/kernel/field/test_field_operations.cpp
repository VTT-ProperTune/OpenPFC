// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <numeric>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/field_operations.hpp>
#include <openpfc/kernel/simulation/model.hpp>

using namespace pfc;
using Catch::Approx;

namespace {
class DummyModel : public Model {
public:
  DummyModel(FFT &fft, const Domain &domain) : Model(fft, domain) {}
  void step(double t) override { (void)t; }
  void initialize(double dt) override { (void)dt; }
};
} // namespace

TEST_CASE("field::apply sets constant value over inbox", "[field_ops][unit]") {
  pfc::Int3 size{8, 4, 2};
  pfc::Domain domain =
      pfc::domain::create(pfc::GridSize(size), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::Int3 lower{0, 0, 0};
  pfc::Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1};
  auto decomp = decomposition::create(domain, 1);
  auto fft = fft::create(decomp);

  DummyModel model(fft, domain);

  std::vector<double> u(fft.size_inbox(), 0.0);
  add_real_field(model, "psi", u);

  field::apply(model, "psi", [](const Real3 & /*x*/) { return 0.5; });

  const auto &ref = model.get_real_field("psi");
  bool values_match = true;
  for (const auto &val : ref) {
    values_match &= val == Approx(0.5);
  }
  REQUIRE(values_match);
}

TEST_CASE("field::apply_with_time uses time parameter", "[field_ops][unit]") {
  pfc::Int3 size{4, 4, 1};
  pfc::Domain domain =
      pfc::domain::create(pfc::GridSize(size), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::Int3 lower{0, 0, 0};
  pfc::Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1};
  auto decomp = decomposition::create(domain, 1);
  auto fft = fft::create(decomp);

  DummyModel model(fft, domain);
  std::vector<double> u(fft.size_inbox(), 0.0);
  add_real_field(model, "psi", u);

  field::apply_with_time(model, "psi", /*t=*/2.0,
                         [](const Real3 & /*x*/, double t) { return 1.0 + t; });

  const auto &ref = model.get_real_field("psi");
  bool values_match = true;
  for (const auto &val : ref) {
    values_match &= val == Approx(3.0);
  }
  REQUIRE(values_match);
}

TEST_CASE("field::apply_inplace modifies field based on current value",
          "[field_ops][unit]") {
  pfc::Int3 size{4, 2, 2};
  pfc::Domain domain =
      pfc::domain::create(pfc::GridSize(size), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::Int3 lower{0, 0, 0};
  pfc::Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1};
  auto decomp = decomposition::create(domain, 1);
  auto fft = fft::create(decomp);

  DummyModel model(fft, domain);
  std::vector<double> u(fft.size_inbox(), 1.0);
  add_real_field(model, "psi", u);

  // Double all values
  field::apply_inplace(model, "psi", [](const Real3 & /*x*/, double current) {
    return 2.0 * current;
  });

  const auto &ref = model.get_real_field("psi");
  bool values_match = true;
  for (const auto &val : ref) {
    values_match &= val == Approx(2.0);
  }
  REQUIRE(values_match);
}

TEST_CASE("field::apply_inplace selective update preserves untouched cells",
          "[field_ops][unit]") {
  pfc::Int3 size{8, 1, 1};
  pfc::Domain domain =
      pfc::domain::create(pfc::GridSize(size), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::Int3 lower{0, 0, 0};
  pfc::Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1};
  auto decomp = decomposition::create(domain, 1);
  auto fft = fft::create(decomp);

  DummyModel model(fft, domain);
  std::vector<double> u(fft.size_inbox(), 0.0);
  add_real_field(model, "psi", u);

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
    if (val == Approx(0.0)) {
      has_zero = true;
    }
    if (val == Approx(1.0)) {
      has_one = true;
    }
  }
  REQUIRE(has_zero);
  REQUIRE(has_one);
}

TEST_CASE("field::apply_inplace_with_time uses time parameter",
          "[field_ops][unit]") {
  pfc::Int3 size{4, 2, 1};
  pfc::Domain domain =
      pfc::domain::create(pfc::GridSize(size), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::Int3 lower{0, 0, 0};
  pfc::Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1};
  auto decomp = decomposition::create(domain, 1);
  auto fft = fft::create(decomp);

  DummyModel model(fft, domain);
  std::vector<double> u(fft.size_inbox(), 1.0);
  add_real_field(model, "psi", u);

  // Blend current value with time-dependent term
  field::apply_inplace_with_time(
      model, "psi", /*t=*/2.0,
      [](const Real3 & /*x*/, double current, double t) { return current + t; });

  const auto &ref = model.get_real_field("psi");
  bool values_match = true;
  for (const auto &val : ref) {
    values_match &= val == Approx(3.0); // 1.0 + 2.0
  }
  REQUIRE(values_match);
}

// Test case for legacy adapter removed - functionality no longer supported after M2
// migration
