// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>

using namespace pfc;

namespace {
struct DxOnly {
  double x{};
};
} // namespace

TEST_CASE("SpectralGradient zeros the first derivative of a Nyquist mode",
          "[field][spectral][nyquist]") {
  auto domain = domain::create(GridSize({16, 1, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                               GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(domain, 1);
  auto fft = fft::create(decomp);
  auto u = data::field_from_subdomain<double>(decomp, 0, /*halo=*/0);
  REQUIRE(u.size() == fft.size_inbox());
  for (int i = 0; i < 16; ++i) {
    u(i, 0, 0) = (i % 2 == 0) ? 1.0 : -1.0;
  }

  auto grad = field::create<DxOnly>(u, fft);
  grad.prepare();

  bool near_zero = true;
  for (int i = 0; i < 16; ++i) {
    near_zero &= std::abs(grad(i, 0, 0).x) <= 1e-12;
  }
  REQUIRE(near_zero);
}
