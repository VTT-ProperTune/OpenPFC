// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_multi_field_device.hip
 * @brief HIP twin of `test_multi_field_device.cu` for multi-field
 *        `for_each_interior_device`.
 *
 * Covers DevicePtrPackN / scatter_device, CompositeGradientDevice,
 * evaluate_fd_grad_composite with catalog grads names, and GPU-vs-CPU
 * agreement for 2-field and 3-field models. Compiling under HIP is an
 * M3 requirement; executing on a GPU is M-LUMI.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>
#include <openpfc/runtime/hip/fd_gradient_device.hpp>
#include <openpfc/runtime/hip/for_each_interior_device.hpp>

namespace {

constexpr double kWaveC = 1.0;

bool hip_runtime_available() {
  int count = 0;
  const hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess) {
    static_cast<void>(hipGetLastError());
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

pfc::data::Field<double> make_padded_host_field(int nx, int ny, int nz, double dx,
                                                double dy, double dz, int hw) {
  const auto domain = pfc::domain::create(pfc::GridSize({nx, ny, nz}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({dx, dy, dz}));
  const auto decomp = pfc::decomposition::create(domain, /*nparts=*/1);
  return pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, hw);
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
  OPENPFC_HD pfc::sim::hip::DeviceInc2 rhs(double /*t*/,
                                            const WaveLocal &g) const noexcept {
    const double lap_u = inv_dx2 * g.u.xx + inv_dy2 * g.u.yy;
    return pfc::sim::hip::DeviceInc2{g.v.value, kWaveC * kWaveC * lap_u};
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
  OPENPFC_HD pfc::sim::hip::DeviceInc2
  rhs(double /*t*/, const KobayashiLocal &g) const noexcept {
    const double lap = g.phi.xx + g.phi.yy;
    return pfc::sim::hip::DeviceInc2{g.phi.value * (1.0 - g.phi.value) + alpha * lap,
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
  OPENPFC_HD pfc::sim::hip::DeviceInc3 rhs(double /*t*/,
                                            const TripleLocal &g) const noexcept {
    return pfc::sim::hip::DeviceInc3{g.a.value, g.b.xx, g.c.yy};
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
    pfc::hip::CompositeGradientDevicePOD eval, CompEvalOut *out, int nx, int ny,
    int nz) {
  const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int iy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int iz = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (ix >= nx || iy >= ny || iz >= nz) {
    return;
  }
  const CompEvalLocal g =
      pfc::hip::evaluate_fd_grad_composite<CompEvalLocal, UGrads, VGrads>(
          eval, ix, iy, iz);
  const auto &f0 = eval.fields[0];
  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(ix + f0.hw) +
                           static_cast<std::ptrdiff_t>(iy + f0.hw) * f0.sy +
                           static_cast<std::ptrdiff_t>(iz + f0.hw) * f0.sz;
  out[c].u_xx = g.u.xx;
  out[c].u_yy = g.u.yy;
  out[c].v_value = g.v.value;
}

__global__ void scatter_device_kernel(pfc::sim::hip::DevicePtrPack3 du,
                                     pfc::sim::hip::DeviceInc3 inc,
                                     std::ptrdiff_t c) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    pfc::sim::hip::detail::scatter_device(du, inc, c);
  }
}

} // namespace

TEST_CASE("test_scatter_device_tuple",
          "[hip][for_each_interior_device][multi-field][integration]") {
  if (!hip_runtime_available()) {
    SUCCEED("Skipping: no HIP device");
    return;
  }

  constexpr std::size_t n = 8;
  double *d0 = nullptr;
  double *d1 = nullptr;
  double *d2 = nullptr;
  REQUIRE(hipMalloc(&d0, n * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d1, n * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d2, n * sizeof(double)) == hipSuccess);
  REQUIRE(hipMemset(d0, 0, n * sizeof(double)) == hipSuccess);
  REQUIRE(hipMemset(d1, 0, n * sizeof(double)) == hipSuccess);
  REQUIRE(hipMemset(d2, 0, n * sizeof(double)) == hipSuccess);

  const auto pack = pfc::sim::hip::make_device_ptr_pack(d0, d1, d2);
  const pfc::sim::hip::DeviceInc3 inc{1.25, -2.5, 3.75};
  const std::ptrdiff_t c = 3;
  scatter_device_kernel<<<1, 1>>>(pack, inc, c);
  REQUIRE(hipDeviceSynchronize() == hipSuccess);

  std::vector<double> h0(n, 0.0), h1(n, 0.0), h2(n, 0.0);
  REQUIRE(hipMemcpy(h0.data(), d0, n * sizeof(double),
                     hipMemcpyDeviceToHost) == hipSuccess);
  REQUIRE(hipMemcpy(h1.data(), d1, n * sizeof(double),
                     hipMemcpyDeviceToHost) == hipSuccess);
  REQUIRE(hipMemcpy(h2.data(), d2, n * sizeof(double),
                     hipMemcpyDeviceToHost) == hipSuccess);

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

  REQUIRE(hipFree(d0) == hipSuccess);
  REQUIRE(hipFree(d1) == hipSuccess);
  REQUIRE(hipFree(d2) == hipSuccess);
}

TEST_CASE("test_evaluate_fd_grad_composite",
          "[hip][fd_gradient_device][multi-field][integration]") {
  if (!hip_runtime_available()) {
    SUCCEED("Skipping: no HIP device");
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
  REQUIRE(hipMalloc(&d_u, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_v, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_out, total * sizeof(CompEvalOut)) == hipSuccess);
  REQUIRE(hipMemcpy(d_u, h_u.data(), total * sizeof(double),
                     hipMemcpyHostToDevice) == hipSuccess);
  REQUIRE(hipMemcpy(d_v, h_v.data(), total * sizeof(double),
                     hipMemcpyHostToDevice) == hipSuccess);
  REQUIRE(hipMemset(d_out, 0, total * sizeof(CompEvalOut)) == hipSuccess);

  pfc::hip::FdGradientDevice<UGrads> eval_u(d_u, nx, ny, nz, dx, dy, dz, hw,
                                             order);
  pfc::hip::FdGradientDevice<VGrads> eval_v(d_v, nx, ny, nz, dx, dy, dz, hw,
                                             order);
  auto composite =
      pfc::hip::create_composite_device<CompEvalLocal>(eval_u, eval_v);

  dim3 block(8, 8, 1);
  dim3 grid(1, 1, 1);
  evaluate_composite_kernel<<<grid, block>>>(composite.pod(), d_out, nx, ny, nz);
  REQUIRE(hipDeviceSynchronize() == hipSuccess);

  std::vector<CompEvalOut> h_out(total);
  REQUIRE(hipMemcpy(h_out.data(), d_out, total * sizeof(CompEvalOut),
                     hipMemcpyDeviceToHost) == hipSuccess);

  auto brick_u = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto brick_v = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  REQUIRE(brick_u.local_size()[0] == nx);
  for (int pk = -hw; pk < nz + hw; ++pk) {
    for (int pj = -hw; pj < ny + hw; ++pj) {
      for (int pi = -hw; pi < nx + hw; ++pi) {
        const double x = static_cast<double>(pi) * dx;
        const double y = static_cast<double>(pj) * dy;
        brick_u(pi, pj, pk) = x * x + 2.0 * y * y;
        brick_v(pi, pj, pk) = 3.0 + x;
      }
    }
  }
  auto cpu_grad_u = pfc::field::create<UGrads>(brick_u, order);
  auto cpu_grad_v = pfc::field::create<VGrads>(brick_v, order);

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

  REQUIRE(hipFree(d_u) == hipSuccess);
  REQUIRE(hipFree(d_v) == hipSuccess);
  REQUIRE(hipFree(d_out) == hipSuccess);
}

TEST_CASE("test_wave2d_double_field_kernel",
          "[hip][for_each_interior_device][multi-field][integration]") {
  if (!hip_runtime_available()) {
    SUCCEED("Skipping: no HIP device");
    return;
  }

  const int order = 2;
  const int hw = order / 2;
  const int nx = 8, ny = 8, nz = 1;
  const double dx = 1.0, dy = 1.0, dz = 1.0;
  // Grads from FdGradient* are already scaled; keep inv_* = 1 for agreement.
  const double inv_dx2 = 1.0;
  const double inv_dy2 = 1.0;

  auto u = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto v = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto du_cpu = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto dv_cpu = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);

  for (int pk = -hw; pk < nz + hw; ++pk) {
    for (int pj = -hw; pj < ny + hw; ++pj) {
      for (int pi = -hw; pi < nx + hw; ++pi) {
        const double x = static_cast<double>(pi) * dx;
        const double y = static_cast<double>(pj) * dy;
        u(pi, pj, pk) = std::sin(0.3 * x) * std::cos(0.2 * y);
        v(pi, pj, pk) = 0.5 * std::cos(0.3 * x) * std::sin(0.2 * y);
        du_cpu(pi, pj, pk) = 0.0;
        dv_cpu(pi, pj, pk) = 0.0;
      }
    }
  }

  auto grad_u = pfc::field::create<UGrads>(u, order);
  auto grad_v = pfc::field::create<VGrads>(v, order);
  auto composite = pfc::field::create_composite<WaveLocal>(grad_u, grad_v);
  WaveDeviceModel cpu_model{.inv_dx2 = inv_dx2, .inv_dy2 = inv_dy2};
  auto du_tuple = std::make_tuple(du_cpu.data() + owned_origin(hw, u.padded_extent(0),
                                                               u.padded_extent(1)),
                                  dv_cpu.data() + owned_origin(hw, v.padded_extent(0),
                                                               v.padded_extent(1)));
  // Padded Field FdGradient indexes from owned origin with padded strides; du
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
  REQUIRE(hipMalloc(&d_u, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_v, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_du, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_dv, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMemcpy(d_u, u.data(), total * sizeof(double),
                     hipMemcpyHostToDevice) == hipSuccess);
  REQUIRE(hipMemcpy(d_v, v.data(), total * sizeof(double),
                     hipMemcpyHostToDevice) == hipSuccess);
  REQUIRE(hipMemset(d_du, 0, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMemset(d_dv, 0, total * sizeof(double)) == hipSuccess);

  pfc::hip::FdGradientDevice<UGrads> dev_u(d_u, nx, ny, nz, dx, dy, dz, hw,
                                            order);
  pfc::hip::FdGradientDevice<VGrads> dev_v(d_v, nx, ny, nz, dx, dy, dz, hw,
                                            order);
  auto dev_composite =
      pfc::hip::create_composite_device<WaveLocal>(dev_u, dev_v);
  WaveDeviceModel gpu_model{.inv_dx2 = inv_dx2, .inv_dy2 = inv_dy2};
  auto pack = pfc::sim::hip::make_device_ptr_pack(d_du, d_dv);

  pfc::sim::hip::for_each_interior_device<WaveDeviceModel, WaveLocal, UGrads,
                                           VGrads>(gpu_model, dev_composite.pod(),
                                                   pack, /*t=*/0.0, nx, ny, nz);

  std::vector<double> h_du(total, 0.0), h_dv(total, 0.0);
  REQUIRE(hipMemcpy(h_du.data(), d_du, total * sizeof(double),
                     hipMemcpyDeviceToHost) == hipSuccess);
  REQUIRE(hipMemcpy(h_dv.data(), d_dv, total * sizeof(double),
                     hipMemcpyDeviceToHost) == hipSuccess);

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

  REQUIRE(hipFree(d_u) == hipSuccess);
  REQUIRE(hipFree(d_v) == hipSuccess);
  REQUIRE(hipFree(d_du) == hipSuccess);
  REQUIRE(hipFree(d_dv) == hipSuccess);
}

TEST_CASE("test_kobayashi_double_field_kernel",
          "[hip][for_each_interior_device][multi-field][integration]") {
  if (!hip_runtime_available()) {
    SUCCEED("Skipping: no HIP device");
    return;
  }

  const int order = 2;
  const int hw = order / 2;
  const int nx = 8, ny = 8, nz = 1;
  const double dx = 1.0, dy = 1.0, dz = 1.0;
  constexpr double kTeq = 1.0;

  auto phi = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto tempr = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto dphi_cpu = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto dtempr_cpu = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);

  for (int pk = -hw; pk < nz + hw; ++pk) {
    for (int pj = -hw; pj < ny + hw; ++pj) {
      for (int pi = -hw; pi < nx + hw; ++pi) {
        const double r2 = static_cast<double>((pi - nx / 2) * (pi - nx / 2) +
                                              (pj - ny / 2) * (pj - ny / 2));
        phi(pi, pj, pk) = r2 < 4.0 ? 1.0 : 0.0;
        tempr(pi, pj, pk) = kTeq - 0.05;
        dphi_cpu(pi, pj, pk) = 0.0;
        dtempr_cpu(pi, pj, pk) = 0.0;
      }
    }
  }

  auto gphi = pfc::field::create<PhiGrads>(phi, order);
  auto gtempr = pfc::field::create<TemprGrads>(tempr, order);
  auto composite =
      pfc::field::create_composite<KobayashiLocal>(gphi, gtempr);
  KobayashiDeviceModel cpu_model{};
  const int nxp = phi.padded_extent(0);
  const int nyp = phi.padded_extent(1);
  auto du_tuple = std::make_tuple(dphi_cpu.data() + owned_origin(hw, nxp, nyp),
                                  dtempr_cpu.data() + owned_origin(hw, nxp, nyp));
  pfc::sim::for_each_interior(cpu_model, composite, du_tuple, 0.0);

  const std::size_t total = phi.size();
  double *d_phi = nullptr, *d_tempr = nullptr, *d_dphi = nullptr, *d_dtempr = nullptr;
  REQUIRE(hipMalloc(&d_phi, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_tempr, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_dphi, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_dtempr, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMemcpy(d_phi, phi.data(), total * sizeof(double),
                     hipMemcpyHostToDevice) == hipSuccess);
  REQUIRE(hipMemcpy(d_tempr, tempr.data(), total * sizeof(double),
                     hipMemcpyHostToDevice) == hipSuccess);
  REQUIRE(hipMemset(d_dphi, 0, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMemset(d_dtempr, 0, total * sizeof(double)) == hipSuccess);

  pfc::hip::FdGradientDevice<PhiGrads> dev_phi(d_phi, nx, ny, nz, dx, dy, dz, hw,
                                                order);
  pfc::hip::FdGradientDevice<TemprGrads> dev_tempr(d_tempr, nx, ny, nz, dx, dy,
                                                    dz, hw, order);
  auto dev_composite =
      pfc::hip::create_composite_device<KobayashiLocal>(dev_phi, dev_tempr);
  KobayashiDeviceModel gpu_model{};
  auto pack = pfc::sim::hip::make_device_ptr_pack(d_dphi, d_dtempr);
  pfc::sim::hip::for_each_interior_device(gpu_model, dev_composite, pack, 0.0, nx,
                                           ny, nz);

  std::vector<double> h_dphi(total), h_dtempr(total);
  REQUIRE(hipMemcpy(h_dphi.data(), d_dphi, total * sizeof(double),
                     hipMemcpyDeviceToHost) == hipSuccess);
  REQUIRE(hipMemcpy(h_dtempr.data(), d_dtempr, total * sizeof(double),
                     hipMemcpyDeviceToHost) == hipSuccess);

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

  REQUIRE(hipFree(d_phi) == hipSuccess);
  REQUIRE(hipFree(d_tempr) == hipSuccess);
  REQUIRE(hipFree(d_dphi) == hipSuccess);
  REQUIRE(hipFree(d_dtempr) == hipSuccess);
}

TEST_CASE("test_synthetic_triple_field_kernel",
          "[hip][for_each_interior_device][multi-field][integration]") {
  if (!hip_runtime_available()) {
    SUCCEED("Skipping: no HIP device");
    return;
  }

  const int order = 2;
  const int hw = order / 2;
  const int nx = 8, ny = 8, nz = 1;
  const double dx = 1.0, dy = 1.0, dz = 1.0;

  auto a = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto b = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto c = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto da_cpu = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto db_cpu = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);
  auto dc_cpu = make_padded_host_field(nx, ny, nz, dx, dy, dz, hw);

  for (int pk = -hw; pk < nz + hw; ++pk) {
    for (int pj = -hw; pj < ny + hw; ++pj) {
      for (int pi = -hw; pi < nx + hw; ++pi) {
        const double x = static_cast<double>(pi);
        const double y = static_cast<double>(pj);
        a(pi, pj, pk) = 0.1 * x + 0.2 * y;
        b(pi, pj, pk) = x * x;
        c(pi, pj, pk) = y * y;
        da_cpu(pi, pj, pk) = db_cpu(pi, pj, pk) = dc_cpu(pi, pj, pk) = 0.0;
      }
    }
  }

  auto ga = pfc::field::create<AGrads>(a, order);
  auto gb = pfc::field::create<BGrads>(b, order);
  auto gc = pfc::field::create<CGrads>(c, order);
  auto composite = pfc::field::create_composite<TripleLocal>(ga, gb, gc);
  TripleDeviceModel cpu_model{};
  const int nxp = a.padded_extent(0);
  const int nyp = a.padded_extent(1);
  const auto off = owned_origin(hw, nxp, nyp);
  auto du_tuple = std::make_tuple(da_cpu.data() + off, db_cpu.data() + off,
                                  dc_cpu.data() + off);
  pfc::sim::for_each_interior(cpu_model, composite, du_tuple, 0.0);

  const std::size_t total = a.size();
  double *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  double *d_da = nullptr, *d_db = nullptr, *d_dc = nullptr;
  REQUIRE(hipMalloc(&d_a, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_b, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_c, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_da, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_db, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMalloc(&d_dc, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMemcpy(d_a, a.data(), total * sizeof(double),
                     hipMemcpyHostToDevice) == hipSuccess);
  REQUIRE(hipMemcpy(d_b, b.data(), total * sizeof(double),
                     hipMemcpyHostToDevice) == hipSuccess);
  REQUIRE(hipMemcpy(d_c, c.data(), total * sizeof(double),
                     hipMemcpyHostToDevice) == hipSuccess);
  REQUIRE(hipMemset(d_da, 0, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMemset(d_db, 0, total * sizeof(double)) == hipSuccess);
  REQUIRE(hipMemset(d_dc, 0, total * sizeof(double)) == hipSuccess);

  pfc::hip::FdGradientDevice<AGrads> da_ev(d_a, nx, ny, nz, dx, dy, dz, hw, order);
  pfc::hip::FdGradientDevice<BGrads> db_ev(d_b, nx, ny, nz, dx, dy, dz, hw, order);
  pfc::hip::FdGradientDevice<CGrads> dc_ev(d_c, nx, ny, nz, dx, dy, dz, hw, order);
  auto dev_composite =
      pfc::hip::create_composite_device<TripleLocal>(da_ev, db_ev, dc_ev);
  TripleDeviceModel gpu_model{};
  auto pack = pfc::sim::hip::make_device_ptr_pack(d_da, d_db, d_dc);
  pfc::sim::hip::for_each_interior_device(gpu_model, dev_composite, pack, 0.0, nx,
                                           ny, nz);

  std::vector<double> h_da(total), h_db(total), h_dc(total);
  REQUIRE(hipMemcpy(h_da.data(), d_da, total * sizeof(double),
                     hipMemcpyDeviceToHost) == hipSuccess);
  REQUIRE(hipMemcpy(h_db.data(), d_db, total * sizeof(double),
                     hipMemcpyDeviceToHost) == hipSuccess);
  REQUIRE(hipMemcpy(h_dc.data(), d_dc, total * sizeof(double),
                     hipMemcpyDeviceToHost) == hipSuccess);

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

  REQUIRE(hipFree(d_a) == hipSuccess);
  REQUIRE(hipFree(d_b) == hipSuccess);
  REQUIRE(hipFree(d_c) == hipSuccess);
  REQUIRE(hipFree(d_da) == hipSuccess);
  REQUIRE(hipFree(d_db) == hipSuccess);
  REQUIRE(hipFree(d_dc) == hipSuccess);
}

#else // !OpenPFC_ENABLE_HIP

TEST_CASE("multi-field device tests skipped (HIP disabled)",
          "[hip][for_each_interior_device][multi-field][integration]") {
  SUCCEED("Skipping: OpenPFC_ENABLE_HIP is off.");
}

#endif
