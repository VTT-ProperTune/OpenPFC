// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// NodalMultigrid on the device reproduces the host operator, V-cycle and PCG
// (#544). Runs on any number of ranks, one device per rank, e.g.
//   mpiexec -n 2 test_nodal_multigrid_cuda
// (each rank selects device rank % count).

#include "mechanics_backend.hpp"

#include <cmath>
#include <random>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/solvers/field_operations.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/nodal_multigrid.hpp>

namespace {

using HostMG = pfc::solvers::NodalMultigrid<pfc::HostSpace>;
using DevMG = pfc::solvers::NodalMultigrid<mechanics_test::Space>;
using HostF = HostMG::FieldT;
using DevF = DevMG::FieldT;

double value(int kind, int gx, int gy, int gz, unsigned seed) {
  std::mt19937 gen(seed +
                   7919u * static_cast<unsigned>(gx + 1000 * gy + 1000000 * gz) +
                   31u * static_cast<unsigned>(kind));
  return std::uniform_real_distribution<double>(-1.0, 1.0)(gen);
}

template <class Fn> void fill_both(HostF &h, DevF &d, Fn &&fn) {
  const auto n = h.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        const auto g = h.global(i, j, k);
        h(i, j, k) = fn(g[0], g[1], g[2]);
      }
  d.with_host_view([&](double *p, std::size_t) {
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const auto g = h.global(i, j, k);
          p[d.idx(i, j, k)] = fn(g[0], g[1], g[2]);
        }
  });
  d.sync_to_device();
}

double rel_diff(const HostF &h, DevF &d) {
  const auto n = h.local_size();
  double e = 0.0, s = 0.0;
  d.with_host_read([&](const double *p, std::size_t) {
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const double a = h(i, j, k), b = p[d.idx(i, j, k)];
          e += (a - b) * (a - b);
          s += a * a;
        }
  });
  double v[2] = {e, s}, g[2];
  MPI_Allreduce(v, g, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return std::sqrt(g[0] / g[1]);
}

} // namespace

TEST_CASE("NodalMultigrid device operator, V-cycle and PCG match the host",
          "[nodal_multigrid][device]") {
  if (!mechanics_test::available()) SKIP("GPU unavailable");
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const int n = 32;
  auto dom = pfc::domain::with_spacing({n, n, n}, {1.0 / n, 1.2 / n, 0.9 / n},
                                       {true, true, true});
  auto dec = pfc::decomposition::create(dom, nproc);
  const auto box = pfc::decomposition::local_box(dec, rank);
  for (auto sm :
       {HostMG::Smoother::Chebyshev, HostMG::Smoother::ColouredGaussSeidel})
    for (int gal : {1, 99})
      for (double sigma : {0.0, 1.0}) {
        HostMG::Controls hc;
        hc.sigma = sigma;
        hc.smoother = sm;
        hc.galerkin_from_level = gal;
        hc.agglomerate_max = 512;
        DevMG::Controls dc;
        dc.sigma = sigma;
        dc.smoother = sm == HostMG::Smoother::Chebyshev
                          ? DevMG::Smoother::Chebyshev
                          : DevMG::Smoother::ColouredGaussSeidel;
        dc.galerkin_from_level = gal;
        dc.agglomerate_max = 512;
        HostMG hm(dec, rank, MPI_COMM_WORLD, hc);
        DevMG dm(dec, rank, MPI_COMM_WORLD, dc);
        REQUIRE(hm.levels() == dm.levels());
        HostF ha(dom, box, 0);
        DevF da(dom, box, 0);
        fill_both(ha, da, [](int x, int y, int z) {
          return 1.0 + 0.95 * value(1, x, y, z, 3);
        });
        hm.update(ha);
        dm.update(da);
        for (int l = 0; l < hm.levels(); ++l)
          REQUIRE(
              std::abs(hm.level_info(l).lambda_max - dm.level_info(l).lambda_max) <=
              1e-10 * hm.level_info(l).lambda_max);
        HostF hr(dom, box, 1), hz(dom, box, 1);
        DevF dr(dom, box, 1), dz(dom, box, 1);
        fill_both(hr, dr, [](int x, int y, int z) { return value(9, x, y, z, 7); });
        hm.apply(hr, hz);
        dm.apply(dr, dz);
        const double vcycle = rel_diff(hz, dz);
        hm.apply_operator(hr, hz);
        dm.apply_operator(dr, dz);
        const double op = rel_diff(hz, dz);
        INFO("smoother " << static_cast<int>(sm) << " galerkin " << gal << " sigma "
                         << sigma << " V-cycle " << vcycle << " operator " << op);
        REQUIRE(vcycle < 1e-11);
        REQUIRE(op < 1e-13);
        if (sigma > 0) {
          pfc::solvers::FieldOperations<pfc::HostSpace> hops(dom, box, 1, 1,
                                                             MPI_COMM_WORLD);
          pfc::solvers::FieldOperations<mechanics_test::Space> dops(dom, box, 1, 1,
                                                                    MPI_COMM_WORLD);
          auto hb = hops.make(hops.size()), hx = hops.make(hops.size());
          auto db = dops.make(dops.size()), dx = dops.make(dops.size());
          fill_both(hb[0], db[0],
                    [](int x, int y, int z) { return value(4, x, y, z, 1); });
          pfc::solvers::KrylovControls c;
          c.relative_tolerance = 1e-10;
          c.maximum_iterations = 200;
          const auto hres = pfc::solvers::cg(
              hops, [&](const auto &u, auto &y) { hm.apply_operator(u[0], y[0]); },
              hb, hx, c, [&](const auto &r, auto &z, int) { hm.apply(r[0], z[0]); });
          const auto dres = pfc::solvers::cg(
              dops, [&](const auto &u, auto &y) { dm.apply_operator(u[0], y[0]); },
              db, dx, c, [&](const auto &r, auto &z, int) { dm.apply(r[0], z[0]); });
          REQUIRE(hres.converged());
          REQUIRE(dres.converged());
          REQUIRE(hres.iterations == dres.iterations);
          REQUIRE(rel_diff(hx[0], dx[0]) < 1e-9);
        }
      }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (mechanics_test::available()) {
    int count = 0;
#if defined(OPENPFC_TEST_MECHANICS_HIP)
    hipGetDeviceCount(&count);
    if (count > 0) hipSetDevice(rank % count);
#else
    cudaGetDeviceCount(&count);
    if (count > 0) cudaSetDevice(rank % count);
#endif
  }
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
