// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <catch2/catch_approx.hpp>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/field/validation.hpp>
#include <openpfc/kernel/integrator/stage_context.hpp>
#include <openpfc/kernel/integrator/workspace.hpp>

using namespace pfc::field;
using Catch::Approx;

TEST_CASE("Aliasing allows documented ScaledField pattern (Domain version)",
          "[field][state_access][domain][unit]") {
  // LocalField value ctor is private; construct via from_subdomain only.
  // In-place axpy `u += dt * du` uses ScaledField and must not route through
  // FieldOutput::validate_no_alias (documented exception to alias rejection).
  auto world = pfc::domain::create_world(
      pfc::GridSize({4, 4, 4}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, /*nparts=*/1);
  auto u = pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0,
                                                    /*halo=*/0);
  auto du = pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0,
                                                     /*halo=*/0);

  for (std::size_t i = 0; i < u.size(); ++i) {
    u.data()[i] = 1.0;
    du.data()[i] = 2.0;
  }

  const double dt = 0.5;
  u += dt * du; // ScaledField in-place update; no validate_no_alias call

  for (std::size_t i = 0; i < u.size(); ++i) {
    REQUIRE(u.data()[i] == Approx(2.0)); // 1.0 + 0.5 * 2.0
  }
}
