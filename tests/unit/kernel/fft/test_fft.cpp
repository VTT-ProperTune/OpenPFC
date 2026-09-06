// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <complex>
#include <cstddef>
#include <vector>

#include <mpi.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/fft_layout.hpp>

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("r2c z-slab real inbox uses y-slab complex outbox",
          "[fft][layout][unit]") {
  auto domain = domain::create(Int3{768, 768, 768});
  auto decomp = decomposition::create(domain, Int3{1, 1, 16});
  auto layout = fft::layout::create(decomp, 0);
  const auto &real0 = fft::layout::get_real_box(layout, 0);
  const auto &cplx0 = fft::layout::get_complex_box(layout, 0);
  const auto &cplx15 = fft::layout::get_complex_box(layout, 15);
  REQUIRE(real0.size[0] == 768);
  REQUIRE(real0.size[1] == 768);
  REQUIRE(real0.size[2] == 48);
  REQUIRE(cplx0.size[0] == 385);
  REQUIRE(cplx0.size[1] == 48);
  REQUIRE(cplx0.size[2] == 768);
  REQUIRE(cplx15.high[2] == 767);
  REQUIRE(cplx15.low[1] > cplx0.low[1]);
}

TEST_CASE("FFT - basic functionality", "[fft][unit]") {
  auto domain = domain::create(GridSize({8, 1, 1}), PhysicalOrigin({1.0, 1.0, 1.0}),
                               GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(domain, 1);
  auto fft = fft::create(decomposition);
  REQUIRE(fft.size_inbox() > 0);
  REQUIRE(fft.size_outbox() > 0);
  REQUIRE(fft.size_workspace() > 0);
}

TEST_CASE("FFT - forward transformation", "[fft][unit]") {
  // Create an FFT object with a fixed decomposition
  auto domain = domain::create(GridSize({8, 1, 1}), PhysicalOrigin({1.0, 1.0, 1.0}),
                               GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(domain, 1);
  auto fft = fft::create(decomposition);

  // Generate input data: sine wave samples at 8 equally spaced points
  // Input represents: sin(k*x) where x = [0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]
  std::vector<double> input = {0.000, 0.785, 1.571, 2.356,
                               3.142, 3.927, 4.712, 5.498};
  REQUIRE(input.size() == fft.size_inbox());

  // Perform the forward transformation
  std::vector<std::complex<double>> output(fft.size_outbox());
  fft.forward(input, output);

  // Sum of input values should appear in DC component (k=0)
  // Expected: sum(input) ≈ 21.991
  REQUIRE_THAT(std::real(output[0]), WithinAbs(21.991, 0.01));
}

TEST_CASE("FFT - backward transformation", "[fft][unit]") {
  // Create an FFT object with a fixed decomposition
  auto domain = domain::create(GridSize({2, 1, 1}), PhysicalOrigin({1.0, 1.0, 1.0}),
                               GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(domain, 1);
  auto fft = fft::create(decomposition);

  // Generate input data in frequency space
  // Two frequency components: DC=1.0 and first harmonic=2.0
  using complex = std::complex<double>;
  std::vector<complex> input = {complex(1.0, 0.0), complex(2.0, 0.0)};

  // Perform the backward transformation to real space
  std::vector<double> output(fft.size_inbox());
  fft.backward(input, output);

  // Verify output size
  REQUIRE(output.size() == fft.size_inbox());

  // Average value should be (1.0 + 2.0) / 2 = 1.5
  REQUIRE_THAT(output[0], WithinAbs(1.5, 0.01));
}

TEST_CASE("FFT workspace allocation - FFTW reports one complex workspace",
          "[fft][unit][allocation]") {
  auto domain = domain::create(GridSize({8, 1, 1}), PhysicalOrigin({1.0, 1.0, 1.0}),
                               GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(domain, 1);
  auto fft = fft::create(decomposition);

  REQUIRE(fft.size_workspace() > 0);
  REQUIRE(fft.get_allocated_memory_bytes() ==
          fft.size_workspace() * sizeof(std::complex<double>));
}

TEST_CASE("FFT create honors r2c_direction on a non-cubic grid", "[fft][unit]") {
  auto domain =
      domain::create(GridSize({8, 16, 32}), PhysicalOrigin({0.0, 0.0, 0.0}),
                     GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(domain, 1);
  auto fft_x = fft::create(decomposition, 0, MPI_COMM_WORLD, /*r2c_direction=*/0);
  auto fft_z = fft::create(decomposition, 0, MPI_COMM_WORLD, /*r2c_direction=*/2);
  REQUIRE(fft_x.size_inbox() == fft_z.size_inbox());
  REQUIRE(fft_x.size_outbox() == static_cast<std::size_t>(5 * 16 * 32));
  REQUIRE(fft_z.size_outbox() == static_cast<std::size_t>(8 * 16 * 17));
}
