// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_multi_field_device.cu
 * @brief Integration tests for multi-field `for_each_interior_device`.
 *
 * Covers DevicePtrPackN / scatter_device, CompositeGradientDevice,
 * evaluate_fd_grad_composite with catalog grads names, and GPU-vs-CPU
 * agreement for 2-field and 3-field models.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>
#include <openpfc/runtime/cuda/fd_gradient_device.hpp>
#include <openpfc/runtime/cuda/for_each_interior_device.hpp>

namespace {

constexpr double kWaveC = 1.0;

bool cuda_runtime_available() {
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return count > 0;
}

inline std::size_t lin(int pi, int pj, int pk, int nxp, int nyp) {
  return static_cast<std::size_t>(pi) +
         static_cast<std::size_t>(pj) * static_cast<std::size_t>(nxp) +
         static_cast<std::size_t>(pk) * static_cast<std::size_t>(nxp) *
             static_cast<std::size_t>(nyp);
}

inline std::size_t owned_origin(int hw, int nxp, int nyp) {
  return lin(hw, hw, hw, nxp, nyp);
}

// Catalog per-field grads (evaluate_fd_grad fills xx/yy/value).
struct UGrads {
  double xx{};
  double yy{};
};
struct VGrads {
  double value{};
};
struct WaveLocal {
  UGrads u;
  VGrads v;
  auto as_tuple() { return std::tie(u, v); }
  auto as_tuple() const { return std::tie(u, v); }
};

struct WaveDeviceModel {
  double inv_dx2{1.0};
  double inv_dy2{1.0};
  OPENPFC_HD pfc::sim::cuda::DeviceInc2 rhs(double /*t*/,
                                            const WaveLocal &g) const noexcept {
    const double lap_u = inv_dx2 * g.u.xx + inv_dy2 * g.u.yy;
    return pfc::sim::cuda::DeviceInc2{g.v.value, kWaveC * kWaveC * lap_u};
  }
};

struct PhiGrads {
  double value{};
  double xx{};
  double yy{};
};
struct TemprGrads {
  double value{};
};
struct KobayashiLocal {
  PhiGrads phi;
  TemprGrads tempr;
};

struct KobayashiDeviceModel {
  double alpha{0.5};
  OPENPFC_HD pfc::sim::cuda::DeviceInc2
  rhs(double /*t*/, const KobayashiLocal &g) const noexcept {
    const double lap = g.phi.xx + g.phi.yy;
    return pfc::sim::cuda::DeviceInc2{g.phi.value * (1.0 - g.phi.value) + alpha * lap,
                                      -g.tempr.value + g.phi.value};
  }
};

struct AGrads {
  double value{};
};
struct BGrads {
  double xx{};
};
struct CGrads {
  double yy{};
};
struct TripleLocal {
  AGrads a;
  BGrads b;
  CGrads c;
};

struct TripleDeviceModel {
  OPENPFC_HD pfc::sim::cuda::DeviceInc3 rhs(double /*t*/,
                                            const TripleLocal &g) const noexcept {
    return pfc::sim::cuda::DeviceInc3{g.a.value, g.b.xx, g.c.yy};
  }
};

struct CompEvalLocal {
  UGrads u;
  VGrads v;
};

struct CompEvalOut {
  double u_xx{};
  double u_yy{};
  double v_value{};
};

__global__ void evaluate_composite_kernel(
    pfc::cuda::CompositeGradientDevicePOD eval, CompEvalOut *out, int nx, int ny,
    int nz) {
  const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int iy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int iz = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (ix >= nx || iy >= ny || iz >= nz) {
    return;
  }
  const CompEvalLocal g =
      pfc::cuda::evaluate_fd_grad_composite<CompEvalLocal, UGrads, VGrads>(
          eval, ix, iy, iz);
  const auto &f0 = eval.fields[0];
  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(ix + f0.hw) +
                           static_cast<std::ptrdiff_t>(iy + f0.hw) * f0.sy +
                           static_cast<std::ptrdiff_t>(iz + f0.hw) * f0.sz;
  out[c].u_xx = g.u.xx;
  out[c].u_yy = g.u.yy;
  out[c].v_value = g.v.value;
}

__global__ void scatter_device_kernel(pfc::sim::cuda::DevicePtrPack3 du,
                                     pfc::sim::cuda::DeviceInc3 inc,
                                     std::ptrdiff_t c) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    pfc::sim::cuda::detail::scatter_device(du, inc, c);
  }
}

} // namespace

TEST_CASE("test_scatter_device_tuple",
          "[cuda][for_each_interior_device][multi-field][integration]") {
  if (!cuda_runtime_available()) {
    SUCCEED("Skipping: no CUDA device");
    return;
  }

  constexpr std::size_t n = 8;
  double *d0 = nullptr;
  double *d1 = nullptr;
  double *d2 = nullptr;
  REQUIRE(cudaMalloc(&d0, n * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d1, n * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d2, n * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemset(d0, 0, n * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemset(d1, 0, n * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemset(d2, 0, n * sizeof(double)) == cudaSuccess);

  const auto pack = pfc::sim::cuda::make_device_ptr_pack(d0, d1, d2);
  const pfc::sim::cuda::DeviceInc3 inc{1.25, -2.5, 3.75};
  const std::ptrdiff_t c = 3;
  scatter_device_kernel<<<1, 1>>>(pack, inc, c);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<double> h0(n, 0.0), h1(n, 0.0), h2(n, 0.0);
  REQUIRE(cudaMemcpy(h0.data(), d0, n * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(h1.data(), d1, n * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(h2.data(), d2, n * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  CHECK(h0[static_cast<std::size_t>(c)] == Catch::Approx(1.25));
  CHECK(h1[static_cast<std::size_t>(c)] == Catch::Approx(-2.5));
  CHECK(h2[static_cast<std::size_t>(c)] == Catch::Approx(3.75));
  for (std::size_t i = 0; i < n; ++i) {
    if (static_cast<std::ptrdiff_t>(i) == c) {
      continue;
    }
    CHECK(h0[i] == 0.0);
    CHECK(h1[i] == 0.0);
    CHECK(h2[i] == 0.0);
  }

  cudaFree(d0);
  cudaFree(d1);
  cudaFree(d2);
}

TEST_CASE("test_evaluate_fd_grad_composite",
          "[cuda][fd_gradient_device][multi-field][integration]") {
  if (!cuda_runtime_available()) {
    SUCCEED("Skipping: no CUDA device");
    return;
  }

  const int order = 2;
  const int hw = order / 2;
  const int nx = 8, ny = 8, nz = 1;
  const int nxp = nx + 2 * hw, nyp = ny + 2 * hw, nzp = nz + 2 * hw;
  const std::size_t total =
      static_cast<std::size_t>(nxp) * nyp * static_cast<std::size_t>(nzp);
  const double dx = 1.0, dy = 1.0, dz = 1.0;

  // Quadratic so order-2 central D2 is exact: u = x^2 + 2 y^2, v = 3 + x
  std::vector<double> h_u(total, 0.0), h_v(total, 0.0);
  for (int pk = 0; pk < nzp; ++pk) {
    for (int pj = 0; pj < nyp; ++pj) {
      for (int pi = 0; pi < nxp; ++pi) {
        const double x = static_cast<double>(pi - hw) * dx;
        const double y = static_cast<double>(pj - hw) * dy;
        h_u[lin(pi, pj, pk, nxp, nyp)] = x * x + 2.0 * y * y;
        h_v[lin(pi, pj, pk, nxp, nyp)] = 3.0 + x;
      }
    }
  }

  double *d_u = nullptr;
  double *d_v = nullptr;
  CompEvalOut *d_out = nullptr;
  REQUIRE(cudaMalloc(&d_u, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_v, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_out, total * sizeof(CompEvalOut)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_u, h_u.data(), total * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v, h_v.data(), total * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemset(d_out, 0, total * sizeof(CompEvalOut)) == cudaSuccess);

  pfc::cuda::FdGradientDevice<UGrads> eval_u(d_u, nx, ny, nz, dx, dy, dz, hw,
                                             order);
  pfc::cuda::FdGradientDevice<VGrads> eval_v(d_v, nx, ny, nz, dx, dy, dz, hw,
                                             order);
  auto composite =
      pfc::cuda::create_composite_device<CompEvalLocal>(eval_u, eval_v);

  dim3 block(8, 8, 1);
  dim3 grid(1, 1, 1);
  evaluate_composite_kernel<<<grid, block>>>(composite.pod(), d_out, nx, ny, nz);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<CompEvalOut> h_out(total);
  REQUIRE(cudaMemcpy(h_out.data(), d_out, total * sizeof(CompEvalOut),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  auto world = pfc::domain::create_world(pfc::GridSize({nx, ny, nz}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({dx, dy, dz}));
  auto decomp = pfc::decomposition::create(world, /*nparts=*/1);
  // M2 migration: replace PaddedBrick with pfc::data::Field
  pfc::data::Field<double, pfc::HostSpace> brick_u =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> brick_v =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  REQUIRE(brick_u.local_size()[0] == nx);
  // M2 migration: Use owned region indexing (0..n-1) instead of padded (-hw..n+hw)
  for (int pk = 0; pk < nz; ++pk) {
    for (int pj = 0; pj < ny; ++pj) {
      for (int pi = 0; pi < nx; ++pi) {
        const double x = static_cast<double>(pi) * dx;
        const double y = static_cast<double>(pj) * dy;
        brick_u.set_xy(pi, pj, pk) = x * x + 2.0 * y * y;
        brick_v.set_xy(pi, pj, pk) = 3.0 + x;
      }
    }
  }
  // M2 migration: Use FDGradient constructor directly instead of pfc::field::create
  pfc::gradient::FDGradient<UGrads> cpu_grad_u(brick_u, order);
  pfc::gradient::FDGradient<VGrads> cpu_grad_v(brick_v, order);

  bool match = true;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const auto gu = cpu_grad_u(ix, iy, iz);
        const auto gv = cpu_grad_v(ix, iy, iz);
        const auto &got = h_out[lin(ix + hw, iy + hw, iz + hw, nxp, nyp)];
        match &= std::abs(got.u_xx - gu.xx) <= 1e-12;
        match &= std::abs(got.u_yy - gu.yy) <= 1e-12;
        match &= std::abs(got.v_value - gv.value) <= 1e-12;
        // Analytic: u_xx=2, u_yy=4, v=3+x
        match &= std::abs(got.u_xx - 2.0) <= 1e-12;
        match &= std::abs(got.u_yy - 4.0) <= 1e-12;
        match &= std::abs(got.v_value - (3.0 + static_cast<double>(ix) * dx)) <=
                 1e-12;
      }
    }
  }
  REQUIRE(match);

  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_out);
}

TEST_CASE("test_wave2d_double_field_kernel",
          "[cuda][for_each_interior_device][multi-field][integration]") {
  if (!cuda_runtime_available()) {
    SUCCEED("Skipping: no CUDA device");
    return;
  }

  const int order = 2;
  const int hw = order / 2;
  const int nx = 8, ny = 8, nz = 1;
  const double dx = 1.0, dy = 1.0, dz = 1.0;
  // Grads from FdGradient* are already scaled; keep inv_* = 1 for agreement.
  const double inv_dx2 = 1.0;
  const double inv_dy2 = 1.0;

  auto world = pfc::domain::create_world(pfc::GridSize({nx, ny, nz}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({dx, dy, dz}));
  auto decomp = pfc::decomposition::create(world, 1);
  // M2 migration: replace PaddedBrick with pfc::data::Field
  pfc::data::Field<double, pfc::HostSpace> u =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> v =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> du_cpu =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> dv_cpu =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);

  // M2 migration: Change from padded indexing (-hw..n+hw) to owned region (0..n-1)
  for (int pk = 0; pk < nz; ++pk) {
    for (int pj = 0; pj < ny; ++pj) {
      for (int pi = 0; pi < nx; ++pi) {
        const double x = static_cast<double>(pi) * dx;
        const double y = static_cast<double>(pj) * dy;
        u.set_xy(pi, pj, pk) = std::sin(0.3 * x) * std::cos(0.2 * y);
        v.set_xy(pi, pj, pk) = 0.5 * std::cos(0.3 * x) * std::sin(0.2 * y);
        du_cpu.set_xy(pi, pj, pk) = 0.0;
        dv_cpu.set_xy(pi, pj, pk) = 0.0;
      }
    }
  }

  // M2 migration: Use FDGradient constructors directly
  pfc::gradient::FDGradient<UGrads> grad_u(u, order);
  pfc::gradient::FDGradient<VGrads> grad_v(v, order);
  // M2 migration: Update composite creation for FDGradient usage
  auto composite = pfc::field::create_composite<WaveLocal>(grad_u, grad_v);
  WaveDeviceModel cpu_model{.inv_dx2 = inv_dx2, .inv_dy2 = inv_dy2};
  // M2 migration: Update memory layout calculations for Field instead of PaddedBrick
  auto du_tuple = std::make_tuple(du_cpu.data() + du_cpu.storage_halo() *
                                       (du_cpu.padded_extent(0) * du_cpu.padded_extent(1) +
                                        du_cpu.padded_extent(1) * du_cpu.storage_halo() +
                                        du_cpu.storage_halo()),
                                  dv_cpu.data() + dv_cpu.storage_halo() *
                                       (dv_cpu.padded_extent(0) * dv_cpu.padded_extent(1) +
                                        dv_cpu.padded_extent(1) * dv_cpu.storage_halo() +
                                        dv_cpu.storage_halo()));
  // Field storage_halo and padded_extent compute the same layout as PaddedBrick; du
  // pointers must share that base.
  pfc::sim::for_each_interior(cpu_model, composite, du_tuple, /*t=*/0.0);

  const int nxp = u.padded_extent(0);
  const int nyp = u.padded_extent(1);
  const int nzp = u.padded_extent(2);
  const std::size_t total = u.size();

  double *d_u = nullptr;
  double *d_v = nullptr;
  double *d_du = nullptr;
  double *d_dv = nullptr;
  REQUIRE(cudaMalloc(&d_u, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_v, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_du, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_dv, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_u, u.data(), total * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_v, v.data(), total * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemset(d_du, 0, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemset(d_dv, 0, total * sizeof(double)) == cudaSuccess);

  pfc::cuda::FdGradientDevice<UGrads> dev_u(d_u, nx, ny, nz, dx, dy, dz, hw,
                                            order);
  pfc::cuda::FdGradientDevice<VGrads> dev_v(d_v, nx, ny, nz, dx, dy, dz, hw,
                                            order);
  auto dev_composite =
      pfc::cuda::create_composite_device<WaveLocal>(dev_u, dev_v);
  WaveDeviceModel gpu_model{.inv_dx2 = inv_dx2, .inv_dy2 = inv_dy2};
  auto pack = pfc::sim::cuda::make_device_ptr_pack(d_du, d_dv);

  pfc::sim::cuda::for_each_interior_device<WaveDeviceModel, WaveLocal, UGrads,
                                           VGrads>(gpu_model, dev_composite.pod(),
                                                   pack, /*t=*/0.0, nx, ny, nz);

  std::vector<double> h_du(total, 0.0), h_dv(total, 0.0);
  REQUIRE(cudaMemcpy(h_du.data(), d_du, total * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(h_dv.data(), d_dv, total * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  bool match = true;
  bool halo_ok = true;
  for (int pk = 0; pk < nzp; ++pk) {
    for (int pj = 0; pj < nyp; ++pj) {
      for (int pi = 0; pi < nxp; ++pi) {
        const bool owned = (pi >= hw && pi < nx + hw) && (pj >= hw && pj < ny + hw) &&
                           (pk >= hw && pk < nz + hw);
        const std::size_t i = lin(pi, pj, pk, nxp, nyp);
        if (owned) {
          const int ix = pi - hw, iy = pj - hw, iz = pk - hw;
          match &= std::abs(h_du[i] - du_cpu(ix, iy, iz)) <= 1e-12;
          match &= std::abs(h_dv[i] - dv_cpu(ix, iy, iz)) <= 1e-12;
        } else {
          halo_ok &= h_du[i] == 0.0;
          halo_ok &= h_dv[i] == 0.0;
        }
      }
    }
  }
  REQUIRE(match);
  REQUIRE(halo_ok);

  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_du);
  cudaFree(d_dv);
}

TEST_CASE("test_kobayashi_double_field_kernel",
          "[cuda][for_each_interior_device][multi-field][integration]") {
  if (!cuda_runtime_available()) {
    SUCCEED("Skipping: no CUDA device");
    return;
  }

  const int order = 2;
  const int hw = order / 2;
  const int nx = 8, ny = 8, nz = 1;
  const double dx = 1.0, dy = 1.0, dz = 1.0;
  constexpr double kTeq = 1.0;

  auto world = pfc::domain::create_world(pfc::GridSize({nx, ny, nz}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({dx, dy, dz}));
  auto decomp = pfc::decomposition::create(world, 1);
  // M2 migration: replace PaddedBrick with pfc::data::Field
  pfc::data::Field<double, pfc::HostSpace> phi =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> tempr =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> dphi_cpu =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> dtempr_cpu =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);

  // M2 migration: adjust coordinate calculation for owned region (0..n-1)
  for (int pk = 0; pk < nz; ++pk) {
    for (int pj = 0; pj < ny; ++pj) {
      for (int pi = 0; pi < nx; ++pi) {
        const double r2 = static_cast<double>((pi - nx / 2) * (pi - nx / 2) +
                                              (pj - ny / 2) * (pj - ny / 2));
        phi.set_xy(pi, pj, pk) = r2 < 4.0 ? 1.0 : 0.0;
        tempr.set_xy(pi, pj, pk) = kTeq - 0.05;
        dphi_cpu.set_xy(pi, pj, pk) = 0.0;
        dtempr_cpu.set_xy(pi, pj, pk) = 0.0;
      }
    }
  }

  // M2 migration: Use FDGradient constructors instead of pfc::field::create
  pfc::gradient::FDGradient<PhiGrads> gphi(phi, order);
  pfc::gradient::FDGradient<TemprGrads> gtempr(tempr, order);
  auto composite =
      pfc::field::create_composite<KobayashiLocal>(gphi, gtempr);
  KobayashiDeviceModel cpu_model{};
  // M2 migration: Update memory layout calculations for Field
  const int nxp = phi.padded_extent(0);
  const int nyp = phi.padded_extent(1);
  auto du_tuple = std::make_tuple(dphi_cpu.data() + dphi_cpu.storage_halo() *
                                       (dphi_cpu.padded_extent(0) * dphi_cpu.padded_extent(1) +
                                        dphi_cpu.padded_extent(1) * dphi_cpu.storage_halo() +
                                        dphi_cpu.storage_halo()),
                                  dtempr_cpu.data() + dtempr_cpu.storage_halo() *
                                       (dtempr_cpu.padded_extent(0) * dtempr_cpu.padded_extent(1) +
                                        dtempr_cpu.padded_extent(1) * dtempr_cpu.storage_halo() +
                                        dtempr_cpu.storage_halo()));
  pfc::sim::for_each_interior(cpu_model, composite, du_tuple, 0.0);

  const std::size_t total = phi.size();
  double *d_phi = nullptr, *d_tempr = nullptr, *d_dphi = nullptr, *d_dtempr = nullptr;
  REQUIRE(cudaMalloc(&d_phi, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_tempr, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_dphi, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_dtempr, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_phi, phi.data(), total * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_tempr, tempr.data(), total * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemset(d_dphi, 0, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemset(d_dtempr, 0, total * sizeof(double)) == cudaSuccess);

  pfc::cuda::FdGradientDevice<PhiGrads> dev_phi(d_phi, nx, ny, nz, dx, dy, dz, hw,
                                                order);
  pfc::cuda::FdGradientDevice<TemprGrads> dev_tempr(d_tempr, nx, ny, nz, dx, dy,
                                                    dz, hw, order);
  auto dev_composite =
      pfc::cuda::create_composite_device<KobayashiLocal>(dev_phi, dev_tempr);
  KobayashiDeviceModel gpu_model{};
  auto pack = pfc::sim::cuda::make_device_ptr_pack(d_dphi, d_dtempr);
  pfc::sim::cuda::for_each_interior_device(gpu_model, dev_composite, pack, 0.0, nx,
                                           ny, nz);

  std::vector<double> h_dphi(total), h_dtempr(total);
  REQUIRE(cudaMemcpy(h_dphi.data(), d_dphi, total * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(h_dtempr.data(), d_dtempr, total * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  bool match = true;
  const int nzp = phi.padded_extent(2);
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const std::size_t i = lin(ix + hw, iy + hw, iz + hw, nxp, nyp);
        match &= std::abs(h_dphi[i] - dphi_cpu(ix, iy, iz)) <= 1e-12;
        match &= std::abs(h_dtempr[i] - dtempr_cpu(ix, iy, iz)) <= 1e-12;
      }
    }
  }
  REQUIRE(match);
  (void)nzp;

  cudaFree(d_phi);
  cudaFree(d_tempr);
  cudaFree(d_dphi);
  cudaFree(d_dtempr);
}

TEST_CASE("test_synthetic_triple_field_kernel",
          "[cuda][for_each_interior_device][multi-field][integration]") {
  if (!cuda_runtime_available()) {
    SUCCEED("Skipping: no CUDA device");
    return;
  }

  const int order = 2;
  const int hw = order / 2;
  const int nx = 8, ny = 8, nz = 1;
  const double dx = 1.0, dy = 1.0, dz = 1.0;

  auto world = pfc::domain::create_world(pfc::GridSize({nx, ny, nz}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({dx, dy, dz}));
  auto decomp = pfc::decomposition::create(world, 1);
  // M2 migration: replace PaddedBrick with pfc::data::Field
  pfc::data::Field<double, pfc::HostSpace> a =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> b =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> c =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> da_cpu =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> db_cpu =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);
  pfc::data::Field<double, pfc::HostSpace> dc_cpu =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, /*halo=*/hw);

  // M2 migration: use owned region (0..n-1) instead of padded indexing (-hw..n+hw)
  for (int pk = 0; pk < nz; ++pk) {
    for (int pj = 0; pj < ny; ++pj) {
      for (int pi = 0; pi < nx; ++pi) {
        const double x = static_cast<double>(pi);
        const double y = static_cast<double>(pj);
        a.set_xy(pi, pj, pk) = 0.1 * x + 0.2 * y;
        b.set_xy(pi, pj, pk) = x * x;
        c.set_xy(pi, pj, pk) = y * y;
        da_cpu.set_xy(pi, pj, pk) = db_cpu.set_xy(pi, pj, pk) = dc_cpu.set_xy(pi, pj, pk) = 0.0;
      }
    }
  }

  // M2 migration: Use FDGradient constructors instead of pfc::field::create
  pfc::gradient::FDGradient<AGrads> ga(a, order);
  pfc::gradient::FDGradient<BGrads> gb(b, order);
  pfc::gradient::FDGradient<CGrads> gc(c, order);
  auto composite = pfc::field::create_composite<TripleLocal>(ga, gb, gc);
  TripleDeviceModel cpu_model{};
  // M2 migration: Update memory layout calculations for Field
  const int nxp = a.padded_extent(0);
  const int nyp = a.padded_extent(1);
  const auto off = da_cpu.storage_halo() *
                   (da_cpu.padded_extent(0) * da_cpu.padded_extent(1) +
                    da_cpu.padded_extent(1) * da_cpu.storage_halo() +
                    da_cpu.storage_halo());
  auto du_tuple = std::make_tuple(da_cpu.data() + off, db_cpu.data() + off,
                                  dc_cpu.data() + off);
  pfc::sim::for_each_interior(cpu_model, composite, du_tuple, 0.0);

  const std::size_t total = a.size();
  double *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  double *d_da = nullptr, *d_db = nullptr, *d_dc = nullptr;
  REQUIRE(cudaMalloc(&d_a, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_b, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_c, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_da, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_db, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_dc, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_a, a.data(), total * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_b, b.data(), total * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_c, c.data(), total * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemset(d_da, 0, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemset(d_db, 0, total * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMemset(d_dc, 0, total * sizeof(double)) == cudaSuccess);

  pfc::cuda::FdGradientDevice<AGrads> da_ev(d_a, nx, ny, nz, dx, dy, dz, hw, order);
  pfc::cuda::FdGradientDevice<BGrads> db_ev(d_b, nx, ny, nz, dx, dy, dz, hw, order);
  pfc::cuda::FdGradientDevice<CGrads> dc_ev(d_c, nx, ny, nz, dx, dy, dz, hw, order);
  auto dev_composite =
      pfc::cuda::create_composite_device<TripleLocal>(da_ev, db_ev, dc_ev);
  TripleDeviceModel gpu_model{};
  auto pack = pfc::sim::cuda::make_device_ptr_pack(d_da, d_db, d_dc);
  pfc::sim::cuda::for_each_interior_device(gpu_model, dev_composite, pack, 0.0, nx,
                                           ny, nz);

  std::vector<double> h_da(total), h_db(total), h_dc(total);
  REQUIRE(cudaMemcpy(h_da.data(), d_da, total * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(h_db.data(), d_db, total * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(h_dc.data(), d_dc, total * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  bool match = true;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const std::size_t i = lin(ix + hw, iy + hw, iz + hw, nxp, nyp);
        match &= std::abs(h_da[i] - da_cpu(ix, iy, iz)) <= 1e-12;
        match &= std::abs(h_db[i] - db_cpu(ix, iy, iz)) <= 1e-12;
        match &= std::abs(h_dc[i] - dc_cpu(ix, iy, iz)) <= 1e-12;
      }
    }
  }
  REQUIRE(match);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_da);
  cudaFree(d_db);
  cudaFree(d_dc);
}

#else // !OpenPFC_ENABLE_CUDA

TEST_CASE("multi-field device tests skipped (CUDA disabled)",
          "[cuda][for_each_interior_device][multi-field][integration]") {
  SUCCEED("Skipping: OpenPFC_ENABLE_CUDA is off.");
}

#endif
