// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>

using namespace pfc;
using namespace pfc::fft::kspace;

TEST_CASE("2/3-rule mask zeros the top third of the spectrum",
          "[fft][dealias]") {
  constexpr int N = 16;
  auto domain = domain::create(GridSize({N, N, N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                               GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(domain, 1);
  auto fft = fft::create(decomp);
  const auto outbox = fft.get_outbox_bounds();
  std::vector<double> mask(fft.size_outbox(), -1.0);
  const std::array<int, 3> gsize{N, N, N};
  const std::array<double, 3> spacing{1.0, 1.0, 1.0};
  fill_two_thirds_mask(outbox, gsize, spacing, mask.data(), mask.size());

  bool mask_matches = true;
  std::size_t n_zero = 0;
  for_each_kpoint(outbox, gsize, spacing,
                  [&](std::size_t idx, double kx, double ky, double kz, int, int,
                      int) {
                    const bool keep = two_thirds_keep(kx, ky, kz, spacing);
                    mask_matches &= mask[idx] == (keep ? 1.0 : 0.0);
                    if (!keep) {
                      ++n_zero;
                    }
                  });
  REQUIRE(mask_matches);
  REQUIRE(n_zero > 0);
}
