// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <openpfc/kernel/fft/batched_transform.hpp>
#include <openpfc/kernel/fft/mixed_transform.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <random>

using namespace pfc::fft;
using Catch::Matchers::WithinAbs;
#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
using Space = pfc::CUDASpace;
static bool select_device() {
  int count = 0, rank;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  GPU_CHECK(cudaSetDevice(rank % count));
  return true;
}
#else
using Space = pfc::HIPSpace;
static bool select_device() {
  int count = 0, rank;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return false;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  GPU_CHECK(hipSetDevice(rank % count));
  return true;
}
#endif
template <class T> struct Device {
  T *p = nullptr;
  std::size_t count;
  explicit Device(std::size_t n) : count(n) {
    GPU_CHECK(pfc::gpuMalloc(reinterpret_cast<void **>(&p), n * sizeof(T)));
  }
  ~Device() { pfc::gpuFree(p); }
  std::span<T> span() { return {p, count}; }
  void upload(const std::vector<T> &v) {
    GPU_CHECK(
        pfc::gpuMemcpy(p, v.data(), count * sizeof(T), pfc::gpuMemcpyHostToDevice));
  }
  std::vector<T> read() {
    std::vector<T> v(count);
    GPU_CHECK(
        pfc::gpuMemcpy(v.data(), p, count * sizeof(T), pfc::gpuMemcpyDeviceToHost));
    return v;
  }
};
TEST_CASE("batched device lines preserve raw normalization and padded strides",
          "[batched_boundary]") {
  if (!select_device()) SKIP("no allocated compatible device");
  for (auto type : {RealTransformType::DCT2, RealTransformType::DST2})
    for (int n : {1, 2, 3, 4, 5, 6, 15, 16, 31, 32, 65})
      for (auto layout : {std::array<int, 2>{1, 2 * n + 3}, {37, 2}, {5, 1}}) {
        BoundedAxis axis{type, UniformGrid::CellCentered, n, 2.3};
        BatchedRealTransform<Space> plan(axis, 5, layout[0], layout[1]);
        std::vector<double> input(plan.size_buffer(), -999),
            expect(input.size(), -999);
        for (int b = 0; b < 5; ++b)
          for (int j = 0; j < n; ++j)
            input[b * layout[1] + j * layout[0]] =
                std::sin(.39 * (b + 1) * (j + 1)) + .17 * j;
        Device<double> in(input.size()), out(input.size());
        in.upload(input);
        out.upload(expect);
        plan.forward(in.span(), out.span());
        auto c = out.read();
        for (int b = 0; b < 5; ++b)
          for (int k = 0; k < n; ++k) {
            double sum = 0;
            for (int j = 0; j < n; ++j) {
              double theta = std::numbers::pi * (j + .5) *
                             (k + int(type == RealTransformType::DST2)) / n;
              sum += 2 * input[b * layout[1] + j * layout[0]] *
                     (type == RealTransformType::DST2 ? std::sin(theta)
                                                      : std::cos(theta));
            }
            expect[b * layout[1] + k * layout[0]] = sum;
          }
        for (std::size_t i = 0; i < c.size(); ++i)
          REQUIRE_THAT(
              c[i], WithinAbs(expect[i], 2e-11 * std::max(1., std::abs(expect[i]))));
        REQUIRE(in.read() == input);
        RealTransform evaluator(axis);
        for (int b = 0; b < 5; ++b) {
          std::vector<double> line(n);
          for (int k = 0; k < n; ++k) line[k] = c[b * layout[1] + k * layout[0]];
          int order = type == RealTransformType::DCT2 ? 1 : 0;
          REQUIRE_THAT(evaluator.evaluate(line, axis.origin, order),
                       WithinAbs(0, 2e-11));
          REQUIRE_THAT(evaluator.evaluate(line, axis.origin + axis.length, order),
                       WithinAbs(0, 2e-11));
        }
        plan.backward(out.span(), in.span());
        auto back = in.read();
        for (std::size_t i = 0; i < back.size(); ++i)
          REQUIRE_THAT(back[i], WithinAbs(input[i], 2e-11));
        REQUIRE(out.read() == c);
        plan.forward(in.span(), in.span());
        plan.backward(in.span(), in.span());
        back = in.read();
        for (std::size_t i = 0; i < back.size(); ++i)
          REQUIRE_THAT(back[i], WithinAbs(input[i], 2e-11));
      }
  REQUIRE_THROWS_AS(
      (BatchedRealTransform<Space>(
          {RealTransformType::DCT1, UniformGrid::Endpoints, 3, 1}, 1, 1, 3)),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      (BatchedRealTransform<Space>(
          {RealTransformType::DCT2, UniformGrid::CellCentered, 3, 1}, 2, 1, 1)),
      std::invalid_argument);
}
TEST_CASE("batched device mixed coefficients and operators match host slabs",
          "[batched_boundary][MPI]") {
  if (!select_device()) SKIP("no allocated compatible device");
  int ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  for (auto type : {RealTransformType::DCT2, RealTransformType::DST2})
    for (int nx : {6, 7})
      for (int n : {1, 2, 3, 4, 5, 8}) {
        int ny = std::max(7, ranks);
        BoundedAxis z{type, UniformGrid::CellCentered, n, 2.3, .2};
        BatchedMixedTransform<Space> plan(nx, ny, 3.1, 4.2, z);
        MixedTransform host(nx, ny, 3.1, 4.2, z);
        std::vector<double> input(plan.size_inbox()), expect(input.size());
        auto box = plan.inbox();
        pfc::for_each_index(box, [&](auto p) {
          input[box.to_linear(p)] =
              std::sin(.7 * (p[0] + 1) * (p[1] + 1) * (p[2] + 1)) + .2 * p[2];
        });
        Device<double> in(input.size()), out(input.size());
        Device<std::complex<double>> coeff(plan.size_outbox());
        in.upload(input);
        plan.forward(in.span(), coeff.span());
        std::vector<std::complex<double>> expected_c(coeff.count);
        host.forward(input, expected_c);
        auto c = coeff.read();
        for (std::size_t i = 0; i < c.size(); ++i)
          REQUIRE(std::abs(c[i] - expected_c[i]) <
                  2e-11 * std::max(1., std::abs(expected_c[i])));
        plan.backward(coeff.span(), out.span());
        auto back = out.read();
        REQUIRE(coeff.read() == c);
        REQUIRE(in.read() == input);
        for (std::size_t i = 0; i < input.size(); ++i)
          REQUIRE_THAT(back[i], WithinAbs(input[i], 2e-11));
        for (auto symbol :
             {LaplacianSymbol::Spectral, LaplacianSymbol::CenteredDifference}) {
          in.upload(input);
          host.laplacian(input, expect, symbol);
          plan.laplacian(in.span(), out.span(), symbol);
          back = out.read();
          for (std::size_t i = 0; i < input.size(); ++i)
            REQUIRE_THAT(back[i], WithinAbs(expect[i], 2e-10));
          for (double alpha : {0., 2., 100.}) {
            // Produce matched arbitrary forcing with the independent host operator.
            for (std::size_t i = 0; i < input.size(); ++i)
              expect[i] = -expect[i] + alpha * input[i];
            in.upload(expect);
            plan.helmholtz(in.span(), out.span(), alpha, symbol);
            back = out.read();
            double sum = std::accumulate(input.begin(), input.end(), 0.), global_sum;
            MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double mean = alpha == 0 && type == RealTransformType::DCT2
                              ? global_sum / (double(nx) * ny * n)
                              : 0;
            for (std::size_t i = 0; i < input.size(); ++i)
              REQUIRE_THAT(back[i], WithinAbs(input[i] - mean, 2e-11));
            host.laplacian(input, expect, symbol);
          }
        }
        if (type == RealTransformType::DCT2) {
          std::fill(expect.begin(), expect.end(), 1);
          in.upload(expect);
          REQUIRE_THROWS_AS(plan.poisson(in.span(), out.span()),
                            std::invalid_argument);
          plan.helmholtz(in.span(), out.span(), 2);
          back = out.read();
          for (double v : back) REQUIRE_THAT(v, WithinAbs(.5, 2e-12));
          std::fill(expect.begin(), expect.end(), 0);
          in.upload(expect);
          plan.poisson(in.span(), out.span());
          for (double v : out.read()) REQUIRE(v == 0);
        }
      }
}

TEST_CASE("batched device mixed validation is collective",
          "[batched_boundary][MPI]") {
  if (!select_device()) SKIP("no allocated compatible device");
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  BoundedAxis z{RealTransformType::DCT2, UniformGrid::CellCentered, 3, 2};
  REQUIRE_THROWS_AS((BatchedMixedTransform<Space>(7, ranks - 1, 3, 4, z)),
                    std::invalid_argument);
  if (ranks > 1)
    REQUIRE_THROWS_AS((BatchedMixedTransform<Space>(7, ranks, 3 + rank, 4, z)),
                      std::invalid_argument);
  BatchedMixedTransform<Space> plan(7, std::max(7, ranks), 3, 4, z);
  Device<double> in(plan.size_inbox()), out(plan.size_inbox());
  std::vector<double> values(plan.size_inbox(), 0);
  in.upload(values);
  REQUIRE_THROWS_AS(plan.helmholtz(in.span(), out.span(), -1),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      plan.poisson(in.span(), out.span(), LaplacianSymbol::Spectral, -1),
      std::invalid_argument);
  if (ranks > 1)
    REQUIRE_THROWS_AS(plan.helmholtz(in.span(), out.span(), 1 + rank),
                      std::invalid_argument);
  values[0] = std::numeric_limits<double>::quiet_NaN();
  in.upload(values);
  REQUIRE_THROWS_AS(plan.poisson(in.span(), out.span()), std::invalid_argument);
}
