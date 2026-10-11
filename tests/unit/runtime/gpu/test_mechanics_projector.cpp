// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "test_helpers.hpp"
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <openpfc/runtime/gpu/fft_single_device.hpp>
#include <openpfc/solvers/finite_strain_fft/gpu/equilibrium.hpp>
using namespace pfc::finite_strain;
TEST_CASE("device tensor projection matches an independent full discrete transform",
          "[mechanics][projector][device]") {
  if (!pfc::gpu::test::is_cuda_available()) SKIP("GPU unavailable");
  std::array<int, 3> shape{5, 3, 7};
  std::array<double, 3> spacing{.2, .4, .1};
  constexpr int n = 105;
  pfc::fft::SingleDeviceFFT<pfc::CUDASpace> fft(shape);
  gpu::CompatibilityProjector<pfc::CUDASpace> project(fft, shape, spacing);
  pfc::solvers::gpu::VectorOperations<pfc::backend::CUDATag> ops;
  auto x = ops.make(9 * n), y = ops.make(9 * n), z = ops.make(9 * n);
  std::vector<double> input(9 * n), reference(9 * n);
  for (int i = 0; i < 9 * n; ++i)
    input[i] = .2 + std::sin(.731 * i) + .4 * std::cos(.177 * i);
  x.copy_from_host(input);
  project.apply(x, y);
  project.apply(y, z);
  project.apply(y, y);
  std::vector<std::complex<double>> hat(9 * n);
  const double pi = std::acos(-1.);
  auto pos = [&](int p) { return std::array<int, 3>{p % 5, (p / 5) % 3, p / 15}; };
  for (int k = 0; k < n; ++k) {
    auto frequency = pos(k);
    double q[3], q2 = 0;
    for (int d = 0; d < 3; ++d) {
      int v = frequency[d];
      if (v > shape[d] / 2) v -= shape[d];
      q[d] = v / (shape[d] * spacing[d]);
      q2 += q[d] * q[d];
    }
    std::complex<double> raw[9] = {};
    for (int p = 0; p < n; ++p) {
      auto coordinate = pos(p);
      double theta = 0;
      for (int d = 0; d < 3; ++d)
        theta += 2 * pi * frequency[d] * coordinate[d] / shape[d];
      auto phase = std::polar(1., -theta);
      for (int c = 0; c < 9; ++c) raw[c] += input[c * n + p] * phase;
    }
    for (int row = 0; row < 3; ++row) {
      std::complex<double> dot = 0;
      for (int c = 0; c < 3; ++c) dot += raw[3 * row + c] * q[c];
      for (int c = 0; c < 3; ++c)
        hat[(3 * row + c) * n + k] = q2 ? dot * q[c] / q2 : 0;
    }
  }
  for (int p = 0; p < n; ++p)
    for (int k = 0; k < n; ++k) {
      auto coordinate = pos(p), frequency = pos(k);
      double theta = 0;
      for (int d = 0; d < 3; ++d)
        theta += 2 * pi * frequency[d] * coordinate[d] / shape[d];
      auto phase = std::polar(1., theta) / double(n);
      for (int c = 0; c < 9; ++c)
        reference[c * n + p] += (hat[c * n + k] * phase).real();
    }
  auto got = y.to_host(), twice = z.to_host();
  for (int i = 0; i < 9 * n; ++i) {
    REQUIRE(std::abs(got[i] - reference[i]) < 1e-11);
    REQUIRE(std::abs(got[i] - twice[i]) < 1e-12);
  }
  REQUIRE_THROWS_AS(
      (gpu::CompatibilityProjector<pfc::CUDASpace>(fft, {4, 3, 7}, spacing)),
      std::invalid_argument);
}
TEST_CASE("collective device material failure invalidates every trial",
          "[mechanics][device][MPI][material_failure]") {
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 2) SKIP("requires two communicating ranks");
  if (!pfc::gpu::test::is_cuda_available()) SKIP("GPU unavailable");
  constexpr int n = 27;
  pfc::fft::SingleDeviceFFT<pfc::CUDASpace> fft({3, 3, 3});
  gpu::CompatibilityProjector<pfc::CUDASpace> project(fft, {3, 3, 3}, {1., 1., 1.});
  pfc::solvers::gpu::VectorOperations<pfc::backend::CUDATag> ops(MPI_COMM_WORLD);
  FccSlipParameters parameters{2, 1, .5, .001, 5, 2.25, .1, .02, .2, .05, 1, 1.4};
  gpu::FccEquilibrium<pfc::CUDASpace> model(project, ops, parameters);
  model.material.set_orientations(std::vector<Tensor2>(n, identity2()));
  auto f = ops.make(9 * n);
  std::vector<double> values(9 * n);
  for (int c : {0, 4, 8})
    for (int i = 0; i < n; ++i) values[c * n + i] = 1;
  for (int i = 0; i < n; ++i) values[n + i] = .1;
  f.copy_from_host(values);
  const auto before = model.material.committed.to_host();
  if (rank == 0) model.material.parameters.time_step = 0;
  const auto result = pfc::solvers::newton(ops, model, f);
  REQUIRE(result.status == pfc::solvers::NonlinearStatus::NonFinite);
  REQUIRE_THROWS_AS(model.material.accept_increment(), std::logic_error);
  REQUIRE(model.material.committed.to_host() == before);
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
