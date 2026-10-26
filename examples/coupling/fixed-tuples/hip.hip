// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "verify.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <openpfc/runtime/hip/databuffer_hip.hpp>
#include <openpfc/runtime/hip/fd_gradient_device.hpp>
#include <openpfc/runtime/hip/for_each_interior_device.hpp>
namespace coupled {
// Existing device driver uses named fixed-arity increment packs. Adapt only
// scattering; Model remains the single owner of the two equations.
struct ScatterAdapter {
  Model model;
  OPENPFC_HD pfc::sim::gpu::DeviceInc2 rhs(double t, const Local &g) const noexcept {
    auto inc = model.rhs(t, g);
    return {inc.du, inc.dv};
  }
};
inline void verify_device() {
  int count = 0;
  require(hipGetDeviceCount(&count) == hipSuccess && count > 0,
          "actual HIP device required");
  constexpr int n = 32, hw = 3;
  auto geometry = domain(n);
  auto decomp = pfc::decomposition::create(geometry, 1);
  auto u = pfc::data::field_from_subdomain<double>(decomp, 0, hw);
  auto v = pfc::data::field_from_subdomain<double>(decomp, 0, hw);
  pfc::comm::HaloExchangeOptions opt;
  opt.connectivity = pfc::comm::HaloConnectivity::Full;
  pfc::comm::HaloExchange<pfc::HostSpace, double> hu(u, decomp, 0, MPI_COMM_SELF,
                                                     opt);
  opt.exchange_base = 1;
  pfc::comm::HaloExchange<pfc::HostSpace, double> hv(v, decomp, 0, MPI_COMM_SELF,
                                                     opt);
  auto gu = pfc::gradient::FDGradient<UGrads>(
      u, 6, [&] { hu.exchange(); }, opt.connectivity);
  auto gv = pfc::gradient::FDGradient<VGrads>(
      v, 6, [&] { hv.exchange(); }, opt.connectivity);
  auto host = pfc::field::create_composite<Local>(gu, gv);
  std::vector<double> du(u.size(), -999), dv(v.size(), -999);
  using Buffer = pfc::core::DataBuffer<pfc::backend::HIPTag, double>;
  Buffer qu(u.size()), qv(v.size()), out_u(u.size()), out_v(v.size());
  out_u.copy_from_host(du);
  out_v.copy_from_host(dv);
  const double h = geometry.spacing[0];
  auto dev_u =
      pfc::hip::FDGradientDevice<UGrads>(qu.data(), n, n, n, h, h, h, hw, 6);
  auto dev_v =
      pfc::hip::FDGradientDevice<VGrads>(qv.data(), n, n, n, h, h, h, hw, 6);
  auto device = pfc::hip::create_composite_device<Local>(dev_u, dev_v);
  const pfc::sim::gpu::DevicePtrPack2 outputs{out_u.data(), out_v.data()};
  double maximum = 0, analytic = 0;
  for (double t : {0., .125, .75}) {
    fill(u, v, t);
    poison_ghosts(u);
    poison_ghosts(v);
    pfc::sim::for_each_interior(
        Model{}, host,
        std::make_tuple(du.data() + u.idx(0, 0, 0), dv.data() + v.idx(0, 0, 0)), t);
    qu.copy_from_host(u.vec());
    qv.copy_from_host(v.vec());
    pfc::sim::gpu::for_each_interior_device<ScatterAdapter, Local, UGrads, VGrads>(
        ScatterAdapter{}, device.pod(), outputs, t, n, n, n);
    auto got_u = out_u.to_host(), got_v = out_v.to_host();
    for (std::size_t i = 0; i < du.size(); ++i) {
      require(std::isfinite(got_u[i]) && std::isfinite(got_v[i]),
              "nonfinite device output");
      maximum = std::max(
          {maximum, std::abs(got_u[i] - du[i]), std::abs(got_v[i] - dv[i])});
    }
    analytic = std::max(analytic, residual(u, v, got_u, got_v, t));
  }
  require(maximum < 1e-11, "CPU/HIP FD6 residual/scatter parity");
  require(analytic < .004, "HIP analytic manufactured RHS");
  std::cout << "HIP actual-device parity=" << maximum << " residual=" << analytic
            << '\n';
}
} // namespace coupled
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int result = 0;
  try {
    coupled::verify_device();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    result = 1;
  }
  MPI_Finalize();
  return result;
}
