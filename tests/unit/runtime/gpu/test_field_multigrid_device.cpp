// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// FieldMultigrid on the device reproduces the host V-cycle and operator (#470).

#include "mechanics_backend.hpp"

#include <cmath>
#include <random>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/solvers/field_multigrid.hpp>

namespace {

using HostMG = pfc::solvers::FieldMultigrid<pfc::HostSpace>;
using DevMG = pfc::solvers::FieldMultigrid<mechanics_test::Space>;
using HostF = HostMG::FieldT;
using DevF = DevMG::FieldT;

double value(int kind, int gx, int gy, int gz, unsigned seed) {
  std::mt19937 gen(seed +
                   7919u * static_cast<unsigned>(gx + 1000 * gy + 1000000 * gz) +
                   31u * static_cast<unsigned>(kind));
  return std::uniform_real_distribution<double>(-1.0, 1.0)(gen);
}

/// Write the same owned values into a host field and a device field.
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

TEST_CASE("FieldMultigrid device V-cycle and operator match the host",
          "[field_multigrid][device]") {
  if (!mechanics_test::available()) SKIP("GPU unavailable");
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  for (int nz : {1, 16}) {
    const int n = nz > 1 ? 16 : 64;
    const double h = 1.0 / n;
    auto dom = pfc::domain::with_spacing({n, n, nz}, {h, h, nz > 1 ? h : 1.0},
                                         {true, true, true});
    auto dec = pfc::decomposition::create(dom, nproc);
    HostMG hm(dec, rank, MPI_COMM_WORLD);
    DevMG dm(dec, rank, MPI_COMM_WORLD);
    REQUIRE(hm.levels() == dm.levels());
    REQUIRE(hm.levels() >= 2);
    auto kappa = [](int kind) {
      return [kind](int x, int y, int z) {
        return 1.0 + 0.9 * value(kind, x, y, z, 3);
      };
    };
    fill_both(hm.c(), dm.c(),
              [](int x, int y, int z) { return 0.5 + 0.4 * value(0, x, y, z, 5); });
    fill_both(hm.dx(), dm.dx(), kappa(1));
    fill_both(hm.dy(), dm.dy(), kappa(2));
    fill_both(hm.dz(), dm.dz(), kappa(3));
    fill_both(hm.mask(), dm.mask(), [n](int x, int y, int) {
      return (x - n / 4) * (x - n / 4) + (y - n / 4) * (y - n / 4) < 9 ? 0.0 : 1.0;
    });
    hm.update();
    dm.update();
    const auto box = pfc::decomposition::local_box(dec, rank);
    HostF hr(dom, box, 1), hz(dom, box, 1);
    DevF dr(dom, box, 1), dz(dom, box, 1);
    fill_both(hr, dr, [](int x, int y, int z) { return value(9, x, y, z, 7); });
    dr.sync_to_device();
    hm.apply(hr, hz);
    dm.apply(dr, dz);
    const double vcycle = rel_diff(hz, dz);
    hm.apply_operator(hr, hz);
    dm.apply_operator(dr, dz);
    const double op = rel_diff(hz, dz);
    INFO("nz=" << nz << " levels=" << hm.levels() << " V-cycle " << vcycle
               << " operator " << op);
    REQUIRE(vcycle < 1e-12);
    REQUIRE(op < 1e-13);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
