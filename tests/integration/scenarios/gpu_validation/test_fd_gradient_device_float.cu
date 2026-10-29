// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fd_gradient_device_float.cu
 * @brief `FDGradientDevice<G, float>`, mixed precision through
 *        `G::value_type`, and `with_core` over same-layout fields.
 *
 * @details
 * Two padded fields hold different quadratics, whose order-2 Laplacians are
 * exact constants. One evaluator is built for the first field; a kernel
 * evaluates both through `with_core`, as a model with a runtime number of
 * same-shaped fields does. Checked:
 *
 *   - double values: the analytic Laplacian to round-off;
 *   - float values, float sums: the analytic Laplacian to float tolerance;
 *   - float values, double sums (`value_type = double`): the host stencil in
 *     double on the float-rounded values, to double round-off.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/runtime/cuda/fd_gradient_device.hpp>

namespace {

struct LapDouble {
  double xx{}, yy{}, zz{};
};
struct LapFloat {
  float xx{}, yy{}, zz{};
};
struct LapMixed {
  using value_type = double;
  double xx{}, yy{}, zz{};
};

constexpr int kN = 6, kHw = 1, kP = kN + 2 * kHw;
constexpr double kH = 0.25;

// u = 1 + x^2 - 0.5 y^2 + 2 z^2 (Laplacian 5); w = 3 x^2 + y^2 + z (8).
double field(int which, double x, double y, double z) {
  return which == 0 ? 1.0 + x * x - 0.5 * y * y + 2.0 * z * z
                    : 3.0 * x * x + y * y + z;
}

std::size_t lin(int i, int j, int k) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(kP) * j +
         static_cast<std::size_t>(kP) * kP * k;
}

template <class T> std::vector<T> padded(int which) {
  std::vector<T> v(static_cast<std::size_t>(kP) * kP * kP);
  for (int k = 0; k < kP; ++k)
    for (int j = 0; j < kP; ++j)
      for (int i = 0; i < kP; ++i)
        v[lin(i, j, k)] = static_cast<T>(
            field(which, (i - kHw) * kH, (j - kHw) * kH, (k - kHw) * kH));
  return v;
}

template <class G, class T>
__global__ void laplacians(pfc::gpu::FDGradientDevicePODT<T> e, const T *second,
                           double *out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= kN * kN * kN) return;
  const int i = t % kN, j = (t / kN) % kN, k = t / (kN * kN);
  const G a = pfc::gpu::evaluate_fd_grad<G>(e, i, j, k);
  const G b = pfc::gpu::evaluate_fd_grad<G>(pfc::gpu::with_core(e, second), i, j, k);
  out[t] = static_cast<double>(a.xx + a.yy + a.zz);
  out[kN * kN * kN + t] = static_cast<double>(b.xx + b.yy + b.zz);
}

template <class G, class T> std::vector<double> run() {
  const auto u = padded<T>(0), w = padded<T>(1);
  const std::size_t bytes = u.size() * sizeof(T), cells = kN * kN * kN;
  T *du = nullptr, *dw = nullptr;
  double *dout = nullptr;
  REQUIRE(cudaMalloc(&du, bytes) == cudaSuccess);
  REQUIRE(cudaMalloc(&dw, bytes) == cudaSuccess);
  REQUIRE(cudaMalloc(&dout, 2 * cells * sizeof(double)) == cudaSuccess);
  cudaMemcpy(du, u.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dw, w.data(), bytes, cudaMemcpyHostToDevice);
  const pfc::gpu::FDGradientDevice<G, T> eval(du, kN, kN, kN, kH, kH, kH, kHw, 2);
  laplacians<G, T><<<(cells + 127) / 128, 128>>>(eval.pod(), dw, dout);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<double> out(2 * cells);
  cudaMemcpy(out.data(), dout, out.size() * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(du);
  cudaFree(dw);
  cudaFree(dout);
  return out;
}

// Host order-2 stencil in double on the float-rounded values, in the same
// per-axis order as the device evaluator.
double host_laplacian(const std::vector<float> &v, int i, int j, int k) {
  const double c2 = 1.0 / (kH * kH);
  const int pi = i + kHw, pj = j + kHw, pk = k + kHw;
  const auto at = [&](int a, int b, int c) {
    return static_cast<double>(v[lin(a, b, c)]);
  };
  const double xx =
      -2.0 * c2 * at(pi, pj, pk) + c2 * (at(pi + 1, pj, pk) + at(pi - 1, pj, pk));
  const double yy =
      -2.0 * c2 * at(pi, pj, pk) + c2 * (at(pi, pj + 1, pk) + at(pi, pj - 1, pk));
  const double zz =
      -2.0 * c2 * at(pi, pj, pk) + c2 * (at(pi, pj, pk + 1) + at(pi, pj, pk - 1));
  return xx + yy + zz;
}

bool cuda_runtime_available() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return count > 0;
}

} // namespace

TEST_CASE("FDGradientDevice in float, mixed and double over two fields via "
          "with_core",
          "[cuda][fd_gradient_device][float][integration]") {
  if (!cuda_runtime_available()) {
    SUCCEED("Skipping: no CUDA device on this host.");
    return;
  }
  const int cells = kN * kN * kN;
  const double expected[2] = {5.0, 8.0};

  const auto d = run<LapDouble, double>();
  for (int f = 0; f < 2; ++f)
    for (int t = 0; t < cells; ++t)
      REQUIRE(d[f * cells + t] == Catch::Approx(expected[f]).margin(1e-11));

  const auto s = run<LapFloat, float>();
  for (int f = 0; f < 2; ++f)
    for (int t = 0; t < cells; ++t)
      REQUIRE(s[f * cells + t] == Catch::Approx(expected[f]).margin(5e-4));

  const auto m = run<LapMixed, float>();
  const std::vector<float> fields[2] = {padded<float>(0), padded<float>(1)};
  for (int f = 0; f < 2; ++f)
    for (int t = 0; t < cells; ++t) {
      const int i = t % kN, j = (t / kN) % kN, k = t / (kN * kN);
      REQUIRE(m[f * cells + t] ==
              Catch::Approx(host_laplacian(fields[f], i, j, k)).margin(1e-11));
    }
}

#endif // OpenPFC_ENABLE_CUDA
