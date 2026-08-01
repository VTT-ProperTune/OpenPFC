// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/operations.hpp>

using namespace pfc;

TEST_CASE("Field operations - comprehensive (stub)",
          "[field][comprehensive][unit]") {
  pfc::Int3 size{8, 8, 8};
  pfc::Domain domain =
      pfc::domain::create(pfc::GridSize(size), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(domain, 1);
  auto fft = fft::create(decomposition);
  REQUIRE(fft.size_inbox() > 0);
}
