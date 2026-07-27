// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/operations.hpp>

using namespace pfc;

TEST_CASE("Field operations - comprehensive (Domain version)",
          "[field][comprehensive][domain][unit]") {
  auto world = domain::create_world(GridSize({8, 8, 8}),
                                    PhysicalOrigin({0.0, 0.0, 0.0}),
                                    GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = pfc::decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  REQUIRE(fft.size_inbox() > 0);
}
