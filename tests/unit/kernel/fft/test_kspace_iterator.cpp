// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/box3i.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>

using namespace pfc;
using namespace pfc::fft::kspace;

TEST_CASE("for_each_kpoint matches a hand-rolled loop on even and odd grids",
          "[fft][kspace][iterator]") {
  for (const auto &size : {std::array<int, 3>{8, 6, 4}, std::array<int, 3>{7, 5, 3}}) {
    auto domain = domain::create(GridSize({size[0], size[1], size[2]}),
                                 PhysicalOrigin({0.0, 0.0, 0.0}),
                                 GridSpacing({0.5, 1.0, 2.0}));
    const auto spacing = domain::get_spacing(domain);
    const auto [fx, fy, fz] = k_frequency_scaling(domain);
    // Full-rank outbox: every global mode (as on a 1-rank r2c plan the
    // complex axis is shorter; here we walk a synthetic inclusive box).
    fft::Box3i outbox{{0, 0, 0},
                      {size[0] - 1, size[1] - 1, size[2] - 1},
                      size};

    std::size_t n = 0;
    bool matches = true;
    for_each_kpoint(outbox, domain, [&](std::size_t idx, double kx, double ky,
                                        double kz, int i, int j, int k) {
      matches &= idx == n;
      matches &= kx == k_component(i, size[0], fx);
      matches &= ky == k_component(j, size[1], fy);
      matches &= kz == k_component(k, size[2], fz);
      (void)spacing;
      ++n;
    });
    REQUIRE(matches);
    REQUIRE(n == static_cast<std::size_t>(size[0]) * static_cast<std::size_t>(size[1]) *
                     static_cast<std::size_t>(size[2]));
  }
}
