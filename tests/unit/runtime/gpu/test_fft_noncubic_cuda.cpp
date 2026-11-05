// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * One-rank 12×10×8 round trip on every reduced axis (#762).
 *
 * The shipped call is `pfc::fft::create_cuda`. The same plan is also built
 * directly with HeFFTe, so a wrapper mistake is not hidden behind the
 * provider. Both recover two successive component fields to an absolute
 * error below 2e-12, without rebuilding the FFT plans between components.
 * HIP is not run. The FFT provider is unchanged.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <mpi.h>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/complex_outbox.hpp>
#include <openpfc/runtime/common/heffte_gpu_r2c_layout.hpp>
#include <openpfc/runtime/gpu/fft_gpu.hpp>
#include <openpfc/runtime/gpu/fft_plan_defaults.hpp>

#include <fixtures/session.hpp>

#include "test_helpers.hpp"

namespace {

using Direct = heffte::fft3d_r2c<heffte::backend::cufft>;
using Work = typename Direct::template buffer_container<std::complex<double>>;

double max_abs(const std::vector<double> &got, const std::vector<double> &ref) {
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::abs(got[i] - ref[i]));
  }
  return worst;
}

bool all_finite(const std::vector<double> &values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

std::size_t complex_count(int axis) {
  int n[3] = {12, 10, 8};
  n[axis] = n[axis] / 2 + 1;
  return static_cast<std::size_t>(n[0] * n[1] * n[2]);
}

void report(int axis, int component, double direct_error, double wrapped_error,
            const std::string &device) {
  std::ostringstream out;
  out.setf(std::ios::scientific);
  out.precision(17);
  out << "FFT_NONCUBIC axis " << axis << " component " << component << " "
      << direct_error << " " << wrapped_error << " " << device << '\n';
  std::cout << out.str();
}

} // namespace

TEST_CASE("create_cuda round-trips two 12x10x8 fields on every reduced axis",
          "[gpu][fft][noncubic]") {
  if (!pfc::gpu::test::is_cuda_available()) SKIP("CUDA not available");

  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if (mpi_size != 1) SKIP("one-rank non-cubic FFT");

  cudaDeviceProp prop{};
  REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
  const std::string device = std::string(prop.name) + " " +
                             std::to_string(prop.major) + "." +
                             std::to_string(prop.minor);

  auto domain = pfc::domain::create(pfc::GridSize({12, 10, 8}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  const auto options =
      pfc::fft::default_device_plan_options<heffte::backend::cufft>();
  const auto plan_opts =
      pfc::fft::layout::effective_r2c_plan_options<heffte::backend::cufft>(options);

  for (int axis = 0; axis < 3; ++axis) {
    const auto boxes =
        pfc::runtime::heffte_gpu::make_default_r2c_boxes(decomp, 0, axis, plan_opts);
    Direct direct(boxes.real_inbox, boxes.complex_outbox, boxes.r2c_direction,
                  MPI_COMM_WORLD, options);
    auto wrapped = pfc::fft::create_cuda(decomp, 0, MPI_COMM_WORLD, axis, options);

    REQUIRE(direct.size_inbox() == 12u * 10u * 8u);
    REQUIRE(wrapped.size_inbox() == direct.size_inbox());
    REQUIRE(direct.size_outbox() == complex_count(axis));
    REQUIRE(wrapped.size_outbox() == direct.size_outbox());

    pfc::fft::RealDataBufferCUDA real(direct.size_inbox());
    pfc::fft::ComplexDataBufferCUDA spectrum(direct.size_outbox());
    pfc::fft::RealDataBufferCUDA back(direct.size_inbox());
    Work workspace(direct.size_workspace());

    // Match the installed acceptance consumer's post-step two-field state
    // without relying on its implementation: a 3D wall/periodic mode with
    // eight explicit, coupled linear updates in code units.
    constexpr double pi = 3.14159265358979323846;
    constexpr double hx = 0.7, hy = 0.9, hz = 1.1;
    constexpr double dt = 0.001, coupling = 0.3;
    const double lambda = 2.0 * (std::cos(pi / 12.0) - 1.0) / (hx * hx) +
                          2.0 * (std::cos(2.0 * pi / 10.0) - 1.0) / (hy * hy) +
                          2.0 * (std::cos(2.0 * pi / 8.0) - 1.0) / (hz * hz);
    double a = 0.2, b = -0.1;
    for (int step = 0; step < 8; ++step) {
      const double next_a = a + dt * (lambda * a + coupling * (b - a));
      b += dt * (2.0 * lambda * b + coupling * (a - b));
      a = next_a;
    }
    const std::array<double, 2> amplitudes{a, b};
    for (int component = 0; component < 2; ++component) {
      std::vector<double> field(direct.size_inbox());
      for (std::size_t q = 0; q < field.size(); ++q) {
        const int i = static_cast<int>(q % 12);
        const int j = static_cast<int>((q / 12) % 10);
        const int k = static_cast<int>(q / 120);
        const double mode = std::cos(pi * (i + 0.5) / 12.0) *
                            std::sin(2.0 * pi * j / 10.0) *
                            std::cos(2.0 * pi * k / 8.0);
        field[q] = 1.0 + amplitudes[static_cast<std::size_t>(component)] * mode;
      }

      real.copy_from_host(field);
      direct.forward(real.data(), spectrum.data(), workspace.data());
      direct.backward(spectrum.data(), back.data(), workspace.data(),
                      heffte::scale::full);
      const std::vector<double> direct_host = back.to_host();

      real.copy_from_host(field);
      wrapped.forward(real, spectrum);
      const std::vector<double> preserved_input = real.to_host();
      wrapped.backward(spectrum, back);
      const std::vector<double> wrapped_host = back.to_host();

      REQUIRE(all_finite(direct_host));
      REQUIRE(all_finite(wrapped_host));
      const double direct_error = max_abs(direct_host, field);
      const double wrapped_error = max_abs(wrapped_host, field);
      report(axis, component, direct_error, wrapped_error, device);
      REQUIRE(preserved_input == field);
      REQUIRE(direct_error < 2e-12);
      REQUIRE(wrapped_error < 2e-12);
    }
  }
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) MPI_Init(&argc, &argv);
  const int result = pfc::test::run(argc, argv);
  int mpi_finalized = 0;
  MPI_Finalized(&mpi_finalized);
  if (mpi_finalized == 0) MPI_Finalize();
  return result;
}
