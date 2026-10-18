// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <stdexcept>
#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#include <openpfc/runtime/hip/explicit_rk.hpp>
#include <openpfc/runtime/hip/fd_gradient_device.hpp>
#else
#include <cuda_runtime.h>
#include <openpfc/runtime/cuda/explicit_rk.hpp>
#include <openpfc/runtime/cuda/fd_gradient_device.hpp>
#endif
namespace nfield_rk {
#if defined(__HIPCC__) || defined(__HIP__)
using MemorySpace = pfc::HIPSpace;
#else
using MemorySpace = pfc::CUDASpace;
#endif
inline int rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}
inline int ranks() {
  int r = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &r);
  return r;
}
inline void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
inline auto geometry() {
  return pfc::domain::create(pfc::GridSize({8, 8, 8}),
                             pfc::PhysicalOrigin({0., 0., 0.}),
                             pfc::GridSpacing({1., 1., 1.}));
}
inline auto decomposition(const pfc::Domain &g) {
  return pfc::decomposition::create(g, pfc::Int3{ranks(), 1, 1});
}
template <class Field> void fill_constant(Field &f, double value) {
  f.with_host_view(
      [&](double *ptr, std::size_t count) { std::fill(ptr, ptr + count, value); });
  f.sync_to_device();
}
template <class Field> double max_abs_diff(Field &f, double expected) {
  double e = 0;
  f.with_host_read([&](const double *ptr, std::size_t) {
    auto extent = f.size3();
    for (int k = 0; k < extent[2]; ++k)
      for (int j = 0; j < extent[1]; ++j)
        for (int i = 0; i < extent[0]; ++i) {
          const double got = ptr[f.idx(i, j, k)];
          require(std::isfinite(got), "nonfinite n-field accepted output");
          e = std::max(e, std::abs(got - expected));
        }
  });
  return e;
}
inline double verify_three() {
  constexpr int hw = 2;
  constexpr double dt = 0.01, t = 0.3;
  auto g = geometry();
  auto dec = decomposition(g);
  const auto box = pfc::decomposition::local_box(dec, rank());
  pfc::data::Field<double, MemorySpace> u(g, box, hw), v(g, box, hw), w(g, box, hw);
  pfc::comm::HaloExchangeOptions opts;
  opts.connectivity = pfc::comm::HaloConnectivity::Full;
  pfc::comm::HaloExchange<MemorySpace, double> halos({&u, &v, &w}, dec, rank(),
                                                     MPI_COMM_WORLD, opts);
  auto gu = pfc::gpu::create<Value>(u, 2);
  auto gv = pfc::gpu::create<Value>(v, 2);
  auto gw = pfc::gpu::create<Value>(w, 2);
  auto eval = pfc::gpu::create_composite_device<Local3>(gu, gv, gw);
  eval.set_prepare([&] { halos.exchange(); });
  Model3 model;
  const auto tableau = pfc::sim::steppers::make_rk4_classical<double>();
  auto rk = pfc::sim::steppers::create(std::tie(u, v, w), eval, model, dt, tableau);
  fill_constant(u, 0.25);
  fill_constant(v, 0.50);
  fill_constant(w, 0.75);
  int final_checks = 0;
  rk.step_checked(
      t,
      [&](double proposed) {
        ++final_checks;
        require(std::abs(proposed - (t + dt)) < 1e-15, "3-field final time");
      },
      u, v, w);
  require(final_checks == 1, "3-field final check count");
  double eu = 0.25, ev = 0.50, ew = 0.75;
  rk4_triple(t, dt, eu, ev, ew);
  const double err =
      std::max({max_abs_diff(u, eu), max_abs_diff(v, ev), max_abs_diff(w, ew)});
  require(err < 2e-12, "3-field resident RK disagrees with independent RK4");
  return err;
}
inline double verify_four() {
  constexpr int hw = 2;
  constexpr double dt = 0.01, t = 0.3;
  auto g = geometry();
  auto dec = decomposition(g);
  const auto box = pfc::decomposition::local_box(dec, rank());
  pfc::data::Field<double, MemorySpace> u(g, box, hw), v(g, box, hw), w(g, box, hw),
      q(g, box, hw);
  pfc::comm::HaloExchangeOptions opts;
  opts.connectivity = pfc::comm::HaloConnectivity::Full;
  pfc::comm::HaloExchange<MemorySpace, double> halos({&u, &v, &w, &q}, dec, rank(),
                                                     MPI_COMM_WORLD, opts);
  auto gu = pfc::gpu::create<Value>(u, 2);
  auto gv = pfc::gpu::create<Value>(v, 2);
  auto gw = pfc::gpu::create<Value>(w, 2);
  auto gq = pfc::gpu::create<Value>(q, 2);
  auto eval = pfc::gpu::create_composite_device<Local4>(gu, gv, gw, gq);
  eval.set_prepare([&] { halos.exchange(); });
  Model4 model;
  const auto tableau = pfc::sim::steppers::make_rk4_classical<double>();
  auto rk =
      pfc::sim::steppers::create(std::tie(u, v, w, q), eval, model, dt, tableau);
  fill_constant(u, 0.20);
  fill_constant(v, 0.40);
  fill_constant(w, 0.60);
  fill_constant(q, 0.80);
  int final_checks = 0;
  rk.step_checked(
      t,
      [&](double proposed) {
        ++final_checks;
        require(std::abs(proposed - (t + dt)) < 1e-15, "4-field final time");
      },
      u, v, w, q);
  require(final_checks == 1, "4-field final check count");
  double eu = 0.20, ev = 0.40, ew = 0.60, eq = 0.80;
  rk4_quad(t, dt, eu, ev, ew, eq);
  const double err = std::max({max_abs_diff(u, eu), max_abs_diff(v, ev),
                               max_abs_diff(w, ew), max_abs_diff(q, eq)});
  require(err < 2e-12, "4-field resident RK disagrees with independent RK4");
  return err;
}
inline void report_device_memory() {
  std::size_t free_b = 0, total_b = 0;
#if defined(__HIPCC__) || defined(__HIP__)
  if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return;
#else
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return;
#endif
  if (rank() == 0) {
    const double used_gib = (total_b - free_b) / (1024.0 * 1024.0 * 1024.0);
    const double free_gib = free_b / (1024.0 * 1024.0 * 1024.0);
    std::cout << "RK_NFIELD_MEM used_gib=" << used_gib << " free_gib=" << free_gib
              << " ranks=" << ranks() << '\n';
  }
}

inline int run_resident() {
  const auto t0 = std::chrono::steady_clock::now();
  const double e3 = verify_three();
  const double e4 = verify_four();
  const auto t1 = std::chrono::steady_clock::now();
  const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  if (rank() == 0)
    std::cout << "n-field resident RK4 3-field err=" << e3 << " 4-field err=" << e4
              << " ranks=" << ranks() << " wall_ms=" << wall_ms << '\n';
  report_device_memory();
  return 0;
}
} // namespace nfield_rk
