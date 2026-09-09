// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fft_input_preserved.cpp
 * @brief A transform must not modify the buffer it reads.
 *
 * @details
 * `FFT_Impl::forward` and `FFT_Impl::backward` take their input by
 * `const&`, and every caller relies on that. `SpectralETDSystem::attempt()`
 * relies on it hardest:
 *
 *     Ops::forward(m_fft, psi, psi_hat);                     // reads psi
 *     ...
 *     Ops::pointwise(geometry, t, psi, ..., n_real, ...);    // reads psi again
 *
 * The nonlinearity is evaluated from the same field the transform just read.
 * If the transform used that buffer as scratch, every spectral application
 * would compute its nonlinear term from a destroyed field.
 *
 * That is not hypothetical. HeFFTe 2.4.1 declares the input
 * `input_type const input[]` and then, on some FFTW builds, writes to it
 * anyway. Measured on LUMI with two builds differing *only* in the FFTW
 * library — same source, same compiler, same HeFFTe version, same buffer
 * address:
 *
 * | FFTW              | input after `forward` | transform output |
 * |-------------------|-----------------------|------------------|
 * | Cray `3.3.10.10`  | unchanged             | correct to 4e-15 |
 * | vanilla `3.3.10`  | **destroyed**         | correct to 4e-15 |
 *
 * The output is right either way, which is why this hid for so long: nothing
 * looks wrong until a second stage reads the input back. On the platform this
 * project develops on, it never does.
 *
 * These tests pin the contract at the level the wrapper promises it, so a
 * backend that breaks it fails here rather than as an application diverging
 * thousands of steps later on someone else's machine.
 */

#include <complex>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <mpi.h>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>

using namespace pfc;

namespace {

/// Deterministic, broadband, and nothing like zero: a buffer of zeros would
/// survive being used as scratch and prove nothing.
void fill_real(std::vector<double> &v) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    const double x = static_cast<double>(i);
    v[i] = 0.2 * std::cos(0.1 * x) + 0.05 * std::sin(0.37 * x) + 0.5;
  }
}

void fill_complex(std::vector<std::complex<double>> &v) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    const double x = static_cast<double>(i);
    v[i] = {0.3 * std::cos(0.13 * x) + 0.2, 0.1 * std::sin(0.21 * x) - 0.05};
  }
}

} // namespace

TEST_CASE("forward() does not modify its input", "[fft][unit][contract]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  // One rank: the contract is about buffer use inside a single transform, and
  // a decomposition would only obscure which buffer was written.
  if (nproc != 1) {
    SKIP("single-rank buffer-contract check");
  }

  // A 1-D line and a cube: the FFTW backend picks different plan shapes for
  // them, and the 1-D case is the one the apps' science presets use.
  const std::vector<Int3> grids{{512, 1, 1}, {16, 16, 16}};
  for (const auto &n : grids) {
    auto domain = domain::create(GridSize(n), PhysicalOrigin({0.0, 0.0, 0.0}),
                                 GridSpacing({0.25, 1.0, 1.0}));
    auto decomposition = decomposition::create(domain, 1);
    auto fft = fft::create(decomposition);

    std::vector<double> in(fft.size_inbox());
    std::vector<std::complex<double>> out(fft.size_outbox());
    fill_real(in);
    const std::vector<double> before = in;

    fft.forward(in, out);

    INFO("grid " << n[0] << "x" << n[1] << "x" << n[2]);
    for (std::size_t i = 0; i < in.size(); ++i) {
      if (in[i] != before[i]) {
        INFO("first difference at index " << i << ": " << before[i] << " -> "
                                          << in[i]);
        FAIL("forward() modified its input buffer");
      }
    }
  }
}

TEST_CASE("backward() does not modify its input", "[fft][unit][contract]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    SKIP("single-rank buffer-contract check");
  }

  const std::vector<Int3> grids{{512, 1, 1}, {16, 16, 16}};
  for (const auto &n : grids) {
    auto domain = domain::create(GridSize(n), PhysicalOrigin({0.0, 0.0, 0.0}),
                                 GridSpacing({0.25, 1.0, 1.0}));
    auto decomposition = decomposition::create(domain, 1);
    auto fft = fft::create(decomposition);

    std::vector<std::complex<double>> in(fft.size_outbox());
    std::vector<double> out(fft.size_inbox());
    fill_complex(in);
    const std::vector<std::complex<double>> before = in;

    fft.backward(in, out);

    INFO("grid " << n[0] << "x" << n[1] << "x" << n[2]);
    for (std::size_t i = 0; i < in.size(); ++i) {
      if (in[i] != before[i]) {
        INFO("first difference at index " << i);
        FAIL("backward() modified its input buffer");
      }
    }
  }
}

TEST_CASE("a transform still round-trips after the input is protected",
          "[fft][unit][contract]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    SKIP("single-rank buffer-contract check");
  }
  // Guarding the input must not change what the transform computes; a copy
  // that quietly transformed the wrong buffer would pass both tests above.
  auto domain = domain::create(GridSize({512, 1, 1}),
                               PhysicalOrigin({0.0, 0.0, 0.0}),
                               GridSpacing({0.25, 1.0, 1.0}));
  auto decomposition = decomposition::create(domain, 1);
  auto fft = fft::create(decomposition);

  std::vector<double> in(fft.size_inbox());
  std::vector<std::complex<double>> hat(fft.size_outbox());
  std::vector<double> back(fft.size_inbox());
  fill_real(in);
  // Compare against a snapshot, not against `in`: if the transform did eat
  // its input, comparing with the eaten copy would report a round-trip error
  // that is really the first test's failure wearing a disguise.
  const std::vector<double> before = in;

  fft.forward(in, hat);
  fft.backward(hat, back);

  double worst = 0.0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    worst = std::max(worst, std::abs(back[i] - before[i]));
  }
  INFO("worst round-trip deviation " << worst);
  REQUIRE(worst < 1.0e-12);
}
