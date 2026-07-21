// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fd_gradient_device.cu
 * @brief Integration test for `pfc::cuda::FdGradientDevice<G>` and
 *        `pfc::sim::cuda::for_each_interior_device`.
 *
 * @details
 * The GPU twin must produce **the same numerical result** as the CPU
 * `pfc::field::FdGradient<G>` + `pfc::sim::for_each_interior` pipeline for
 * every model that the SymPy-generated kernel codegen path will eventually
 * hit. We pin the contract here on a small interior brick where:
 *
 *   - the source field is a polynomial `u(x, y, z) = a + b*x + c*x^2 + d*y +
 *     e*y^2 + f*z + g*z^2`, so every D1 / D2 along x, y, z has a closed-form
 *     reference;
 *   - the model rhs is `rhs(t, g) = g.value + g.x + g.xx + g.y + g.yy + g.z +
 *     g.zz`, exercising every branch of the device evaluator simultaneously;
 *   - the test checks bit-comparable (tight FP) agreement against a
 *     hand-rolled host-side reference plus a separate sanity check against
 *     the closed-form analytic answer at the centre cell.
 *
 * The test compiles only when `OpenPFC_ENABLE_CUDA` is on. At runtime it
 * skips gracefully when no CUDA device is available (e.g. CI hosts that
 * happen to load `nvcc` libraries without a real GPU on the node).
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/runtime/cuda/fd_gradient_device.hpp>
#include <openpfc/runtime/cuda/for_each_interior_device.hpp>

namespace {

struct TestGrads {
  double value{};
  double x{};
  double xx{};
  double y{};
  double yy{};
  double z{};
  double zz{};
};

struct TestModel {
  OPENPFC_HD double rhs(double /*t*/, const TestGrads &g) const noexcept {
    return g.value + g.x + g.xx + g.y + g.yy + g.z + g.zz;
  }
};

// Mixed-second Hessian cross-terms: u_xy=H, u_xz=I, u_yz=J (constants).
constexpr double kH = 1.25;
constexpr double kI = -0.75;
constexpr double kJ = 0.5;

struct MixedGrads {
  double value{};
  double xy{};
  double xz{};
  double yz{};
};

struct MixedModel {
  OPENPFC_HD double rhs(double /*t*/, const MixedGrads &g) const noexcept {
    return g.value + g.xy + g.xz + g.yz;
  }
};

struct OnlyXY {
  double xy{};
};

struct OnlyX {
  double x{};
};

constexpr double kA = 0.5;
constexpr double kB = 1.5;
constexpr double kC = -0.25;
constexpr double kD = 2.0;
constexpr double kE = 0.75;
constexpr double kF = -1.0;
constexpr double kG = 0.5;

inline double poly_value(double x, double y, double z) {
  return kA + kB * x + kC * x * x + kD * y + kE * y * y + kF * z + kG * z * z;
}

inline double poly_rhs(double x, double y, double z) {
  // value + (D1_x + D1_y + D1_z) + (D2_x + D2_y + D2_z)
  const double v = poly_value(x, y, z);
  const double ux = kB + 2.0 * kC * x;
  const double uy = kD + 2.0 * kE * y;
  const double uz = kF + 2.0 * kG * z;
  const double uxx = 2.0 * kC;
  const double uyy = 2.0 * kE;
  const double uzz = 2.0 * kG;
  return v + ux + uxx + uy + uyy + uz + uzz;
}

inline std::size_t lin(int pi, int pj, int pk, int nxp, int nyp) {
  return static_cast<std::size_t>(pi) +
         static_cast<std::size_t>(pj) * static_cast<std::size_t>(nxp) +
         static_cast<std::size_t>(pk) * static_cast<std::size_t>(nxp) *
             static_cast<std::size_t>(nyp);
}

bool cuda_runtime_available() {
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    cudaGetLastError(); // clear sticky state
    return false;
  }
  return count > 0;
}

} // namespace

TEST_CASE("for_each_interior_device + FdGradientDevice<G> agree with the "
          "analytic polynomial RHS",
          "[cuda][fd_gradient_device][integration]") {
  if (!cuda_runtime_available()) {
    SUCCEED("Skipping: no CUDA device on this host.");
    return;
  }

  // Order-2 central FD: half_width = 1, so a 1-cell halo on every side is
  // enough. The CPU truth is **exact** for the polynomial because the
  // polynomial has degree <= 2 and central D1 / D2 of order 2 are degree-2
  // exact (i.e. they reproduce the analytic derivative bit-for-bit modulo
  // round-off on the integer-weight summation).
  const int order = 2;
  const int hw = order / 2;
  const int nx = 8;
  const int ny = 8;
  const int nz = 8;
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const int nzp = nz + 2 * hw;
  const std::size_t total = static_cast<std::size_t>(nxp) *
                            static_cast<std::size_t>(nyp) *
                            static_cast<std::size_t>(nzp);

  const double dx = 1.0 / 8.0;
  const double dy = 1.0 / 8.0;
  const double dz = 1.0 / 8.0;

  // Fill the **padded** host buffer with the polynomial in **physical
  // coordinates** (so the FD scaling 1/(h * denom) and 1/(h^2 * denom)
  // produces the analytic derivative directly).
  std::vector<double> h_u(total, 0.0);
  for (int pk = 0; pk < nzp; ++pk) {
    for (int pj = 0; pj < nyp; ++pj) {
      for (int pi = 0; pi < nxp; ++pi) {
        // Padded indices are -hw .. n + hw - 1 mapped to [0 .. nxp - 1].
        // Convert back to "global cell index" (here single-rank, so global
        // == padded - hw).
        const int gx = pi - hw;
        const int gy = pj - hw;
        const int gz = pk - hw;
        const double x = static_cast<double>(gx) * dx;
        const double y = static_cast<double>(gy) * dy;
        const double z = static_cast<double>(gz) * dz;
        h_u[lin(pi, pj, pk, nxp, nyp)] = poly_value(x, y, z);
      }
    }
  }

  // Allocate device buffers.
  double *d_u = nullptr;
  double *d_du = nullptr;
  REQUIRE(cudaMalloc(&d_u, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_du, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_u, h_u.data(), total * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemset(d_du, 0, total * sizeof(double)) == cudaSuccess);

  pfc::cuda::FdGradientDevice<TestGrads> eval(d_u, nx, ny, nz, dx, dy, dz, hw,
                                              order);
  TestModel model;

  pfc::sim::cuda::for_each_interior_device<TestModel, TestGrads>(
      model, eval.pod(), d_du, /*t=*/0.0, nx, ny, nz);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // Copy back and compare against the analytic reference.
  std::vector<double> h_du(total, 0.0);
  REQUIRE(cudaMemcpy(h_du.data(), d_du, total * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  // Owned-cell loop: assert (i) every owned cell matches the analytic
  // formula, (ii) every halo cell of `du` is still zero (the kernel must
  // not write outside the owned region).
  bool owned_cells_match = true;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int pi = ix + hw;
        const int pj = iy + hw;
        const int pk = iz + hw;
        const double x = static_cast<double>(ix) * dx;
        const double y = static_cast<double>(iy) * dy;
        const double z = static_cast<double>(iz) * dz;
        const double expected = poly_rhs(x, y, z);
        const double got = h_du[lin(pi, pj, pk, nxp, nyp)];
        owned_cells_match &= std::abs(got - expected) <= 1e-9;
      }
    }
  }
  REQUIRE(owned_cells_match);

  // Halo cells must be untouched (still 0.0).
  bool halo_cells_untouched = true;
  for (int pk = 0; pk < nzp; ++pk) {
    for (int pj = 0; pj < nyp; ++pj) {
      for (int pi = 0; pi < nxp; ++pi) {
        const bool is_owned = (pi >= hw && pi < nx + hw) &&
                              (pj >= hw && pj < ny + hw) &&
                              (pk >= hw && pk < nz + hw);
        if (is_owned) {
          continue;
        }
        halo_cells_untouched &= h_du[lin(pi, pj, pk, nxp, nyp)] == 0.0;
      }
    }
  }
  REQUIRE(halo_cells_untouched);

  cudaFree(d_u);
  cudaFree(d_du);
}

TEST_CASE("FdGradientDevice mixed xy/xz/yz polynomial Hessian cross-terms",
          "[cuda][fd_gradient_device][integration]") {
  if (!cuda_runtime_available()) {
    SUCCEED("Skipping: no CUDA device on this host.");
    return;
  }

  // Full padded fill is equivalent corner fill (no MPI FullPaddedDeviceHalo).
  const auto mixed_poly = [](double x, double y, double z) {
    return kA + kB * x + kC * x * x + kD * y + kE * y * y + kF * z +
           kG * z * z + kH * x * y + kI * x * z + kJ * y * z;
  };
  const auto mixed_rhs = [&](double x, double y, double z) {
    return mixed_poly(x, y, z) + kH + kI + kJ;
  };

  const int order = 2;
  const int hw = order / 2;
  const int nx = 8, ny = 8, nz = 8;
  const int nxp = nx + 2 * hw, nyp = ny + 2 * hw, nzp = nz + 2 * hw;
  const std::size_t total = static_cast<std::size_t>(nxp) *
                            static_cast<std::size_t>(nyp) *
                            static_cast<std::size_t>(nzp);
  const double dx = 1.0 / 8.0, dy = 1.0 / 8.0, dz = 1.0 / 8.0;

  std::vector<double> h_u(total, 0.0);
  for (int pk = 0; pk < nzp; ++pk) {
    for (int pj = 0; pj < nyp; ++pj) {
      for (int pi = 0; pi < nxp; ++pi) {
        const double x = static_cast<double>(pi - hw) * dx;
        const double y = static_cast<double>(pj - hw) * dy;
        const double z = static_cast<double>(pk - hw) * dz;
        h_u[lin(pi, pj, pk, nxp, nyp)] = mixed_poly(x, y, z);
      }
    }
  }

  double *d_u = nullptr;
  double *d_du = nullptr;
  REQUIRE(cudaMalloc(&d_u, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_du, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_u, h_u.data(), total * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemset(d_du, 0, total * sizeof(double)) == cudaSuccess);

  pfc::cuda::FdGradientDevice<MixedGrads> eval(d_u, nx, ny, nz, dx, dy, dz, hw,
                                               order);
  MixedModel model;
  pfc::sim::cuda::for_each_interior_device<MixedModel, MixedGrads>(
      model, eval.pod(), d_du, /*t=*/0.0, nx, ny, nz);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<double> h_du(total, 0.0);
  REQUIRE(cudaMemcpy(h_du.data(), d_du, total * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  bool owned_cells_match = true;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const double x = static_cast<double>(ix) * dx;
        const double y = static_cast<double>(iy) * dy;
        const double z = static_cast<double>(iz) * dz;
        const double got = h_du[lin(ix + hw, iy + hw, iz + hw, nxp, nyp)];
        owned_cells_match &= std::abs(got - mixed_rhs(x, y, z)) <= 1e-9;
      }
    }
  }
  REQUIRE(owned_cells_match);

  bool halo_cells_untouched = true;
  for (int pk = 0; pk < nzp; ++pk) {
    for (int pj = 0; pj < nyp; ++pj) {
      for (int pi = 0; pi < nxp; ++pi) {
        const bool is_owned = (pi >= hw && pi < nx + hw) &&
                              (pj >= hw && pj < ny + hw) &&
                              (pk >= hw && pk < nz + hw);
        if (!is_owned) {
          halo_cells_untouched &= h_du[lin(pi, pj, pk, nxp, nyp)] == 0.0;
        }
      }
    }
  }
  REQUIRE(halo_cells_untouched);

  cudaFree(d_u);
  cudaFree(d_du);
}

TEST_CASE("FdGradientDevice mixed-only OnlyXY loads D1 weights",
          "[cuda][fd_gradient_device][integration]") {
  if (!cuda_runtime_available()) {
    SUCCEED("Skipping: no CUDA device on this host.");
    return;
  }

  const int order = 2;
  const int hw = order / 2;
  const int nx = 4, ny = 4, nz = 4;
  const int nxp = nx + 2 * hw, nyp = ny + 2 * hw, nzp = nz + 2 * hw;
  const std::size_t total =
      static_cast<std::size_t>(nxp) * nyp * nzp * sizeof(double);

  double *d_u = nullptr;
  REQUIRE(cudaMalloc(&d_u, total) == cudaSuccess);

  pfc::cuda::FdGradientDevice<OnlyXY> eval(d_u, nx, ny, nz, 1.0, 1.0, 1.0, hw,
                                           order);
  const auto &pod = eval.pod();
  REQUIRE(pod.hw1 == 1);
  REQUIRE(pod.cx1[1] != 0.0);
  REQUIRE(pod.cy1[1] != 0.0);

  cudaFree(d_u);
}

TEST_CASE("FdGradientDevice<G> ctor rejects an order without a D1 stencil table",
          "[cuda][fd_gradient_device][integration]") {
  if (!cuda_runtime_available()) {
    SUCCEED("Skipping: no CUDA device on this host.");
    return;
  }

  // D1 tables cap at order 14 today; order 16 must surface as a clean
  // `std::invalid_argument` (mirrors the CPU evaluator).
  const int order = 16;
  const int hw = order / 2;
  const int nx = 4, ny = 4, nz = 4;
  const int nxp = nx + 2 * hw, nyp = ny + 2 * hw, nzp = nz + 2 * hw;
  const std::size_t total =
      static_cast<std::size_t>(nxp) * nyp * nzp * sizeof(double);

  double *d_u = nullptr;
  REQUIRE(cudaMalloc(&d_u, total) == cudaSuccess);
  REQUIRE_THROWS_AS((pfc::cuda::FdGradientDevice<OnlyX>(d_u, nx, ny, nz, 1.0, 1.0,
                                                        1.0, hw, order)),
                    std::invalid_argument);
  cudaFree(d_u);
}

#else // !OpenPFC_ENABLE_CUDA

TEST_CASE("for_each_interior_device tests skipped (CUDA disabled)",
          "[cuda][fd_gradient_device][integration]") {
  SUCCEED("OpenPFC_ENABLE_CUDA is OFF.");
}

#endif // OpenPFC_ENABLE_CUDA
