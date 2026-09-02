// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/operations.hpp>

using namespace pfc;
using Catch::Approx;

namespace {

struct InboxFixture {
  Domain domain;
  pfc::FFT fft;
  std::vector<double> u;

  InboxFixture(Int3 size, double fill)
      : domain(domain::create(GridSize(size), PhysicalOrigin({0.0, 0.0, 0.0}),
                              GridSpacing({1.0, 1.0, 1.0}))),
        fft(fft::create(decomposition::create(domain, 1))),
        u(fft.size_inbox(), fill) {}
};

} // namespace

TEST_CASE("field::apply sets constant value over inbox", "[field_ops][unit]") {
  InboxFixture fx({8, 4, 2}, 0.0);
  field::apply(fx.u, fx.domain, fx.fft, [](const Real3 & /*x*/) { return 0.5; });

  bool values_match = true;
  for (const auto &val : fx.u) {
    values_match &= val == Approx(0.5);
  }
  REQUIRE(values_match);
}

TEST_CASE("field::apply_with_time uses time parameter", "[field_ops][unit]") {
  InboxFixture fx({4, 4, 1}, 0.0);
  field::apply_with_time(fx.u, fx.domain, fx.fft, /*t=*/2.0,
                         [](const Real3 & /*x*/, double t) { return 1.0 + t; });

  bool values_match = true;
  for (const auto &val : fx.u) {
    values_match &= val == Approx(3.0);
  }
  REQUIRE(values_match);
}

TEST_CASE("field::apply_inplace modifies field based on current value",
          "[field_ops][unit]") {
  InboxFixture fx({4, 2, 2}, 1.0);
  field::apply_inplace(fx.u, fx.domain, fx.fft,
                       [](const Real3 & /*x*/, double current) {
                         return 2.0 * current;
                       });

  bool values_match = true;
  for (const auto &val : fx.u) {
    values_match &= val == Approx(2.0);
  }
  REQUIRE(values_match);
}

TEST_CASE("field::apply_inplace selective update preserves untouched cells",
          "[field_ops][unit]") {
  InboxFixture fx({8, 1, 1}, 0.0);
  field::apply_inplace(fx.u, fx.domain, fx.fft,
                       [](const Real3 &x, double current) {
                         if (x[0] > 4.0) {
                           return 1.0;
                         }
                         return current;
                       });

  bool has_zero = false;
  bool has_one = false;
  for (const auto &val : fx.u) {
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
  InboxFixture fx({4, 2, 1}, 1.0);
  field::apply_inplace_with_time(
      fx.u, fx.domain, fx.fft, /*t=*/2.0,
      [](const Real3 & /*x*/, double current, double t) { return current + t; });

  bool values_match = true;
  for (const auto &val : fx.u) {
    values_match &= val == Approx(3.0); // 1.0 + 2.0
  }
  REQUIRE(values_match);
}
