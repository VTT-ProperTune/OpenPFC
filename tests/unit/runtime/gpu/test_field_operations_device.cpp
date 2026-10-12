// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_field_operations_device.cpp
 * @brief Device FieldOperations match the host on owned dots, Full2D halos and CG.
 *
 * The stencil itself runs on a host view of the exchanged device fields. What
 * is on the device is the vector algebra, the owned reduction and the halo
 * exchange. HIP uses the same source when OPENPFC_TEST_MECHANICS_HIP is set.
 */

#include "mechanics_backend.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/solvers/field_operations.hpp>
#include <openpfc/solvers/krylov.hpp>

namespace {

using Host = pfc::HostSpace;
using Dev = mechanics_test::Space;
using HField = pfc::data::Field<double, Host>;
using DField = pfc::data::Field<double, Dev>;
using HVec = pfc::solvers::FieldVector<Host>;
using DVec = pfc::solvers::FieldVector<Dev>;
using HOps = pfc::solvers::FieldOperations<Host>;
using DOps = pfc::solvers::FieldOperations<Dev>;

constexpr int kN = 16;

pfc::comm::HaloExchangeOptions full2d() {
  pfc::comm::HaloExchangeOptions opt;
  opt.connectivity = pfc::comm::HaloConnectivity::Full;
  opt.directions = pfc::halo::presets::Full2D();
  return opt;
}

double kappa(double x, double y) {
  constexpr double pi = 3.14159265358979323846;
  return 1.0 + 0.5 * std::sin(2 * pi * x) * std::cos(2 * pi * y);
}

template <class Field> void each(const Field &f, auto &&fn) {
  const auto n = f.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) fn(i, j, k);
}

template <class Field> void upload(const HField &host, Field &dev) {
  dev.with_host_view(
      [&](double *p, std::size_t n) { std::copy(host.data(), host.data() + n, p); });
  dev.sync_to_device();
}

double rel(const HField &host, DField &dev) {
  double num = 0, den = 0;
  dev.with_host_read([&](const double *p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      const double d = p[i] - host.data()[i];
      num += d * d;
      den += host.data()[i] * host.data()[i];
    }
  });
  double ng = 0, dg = 0;
  MPI_Allreduce(&num, &ng, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&den, &dg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return std::sqrt(ng / std::max(1e-30, dg));
}

void apply_host(const HField &u0, const HField &u1, const HField &dx,
                const HField &dy, HField &o0, HField &o1) {
  const double ih2 = double(kN) * kN;
  each(o0, [&](int i, int j, int k) {
    for (int which = 0; which < 2; ++which) {
      const HField &u = which == 0 ? u0 : u1;
      const HField &other = which == 0 ? u1 : u0;
      HField &o = which == 0 ? o0 : o1;
      const double uc = u(i, j, k);
      const double diff = dx(i, j, k) * (uc - u(i + 1, j, k)) +
                          dx(i - 1, j, k) * (uc - u(i - 1, j, k)) +
                          dy(i, j, k) * (uc - u(i, j + 1, k)) +
                          dy(i, j - 1, k) * (uc - u(i, j - 1, k));
      o(i, j, k) = ih2 * diff + uc + 0.2 * other(i, j, k);
    }
  });
}

} // namespace

TEST_CASE("device FieldOperations match the host dot, halo and CG",
          "[field_ops][gpu]") {
  if (!mechanics_test::available()) SKIP("no device");
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const double h = 1.0 / kN;
  auto domain =
      pfc::domain::with_spacing({kN, kN, 1}, {h, h, 1.0}, {true, true, true});
  auto decomp = pfc::decomposition::create(domain, nproc);
  const auto box = pfc::decomposition::local_box(decomp, rank);

  HOps hops(domain, box, 2, 1, MPI_COMM_WORLD);
  DOps dops(domain, box, 2, 1, MPI_COMM_WORLD);
  HVec hx = hops.make(hops.size()), hy = hops.make(hops.size());
  DVec dx = dops.make(dops.size()), dy = dops.make(dops.size());
  for (int c = 0; c < 2; ++c) {
    each(hx[c], [&](int i, int j, int k) {
      const auto g = hx[c].global(i, j, k);
      hx[c](i, j, k) = 0.2 * (c + 1) * (g[0] + 1) - 0.05 * g[1];
      hy[c](i, j, k) = 0.1 * (g[1] + 2) + 0.03 * c;
    });
    const auto n = hx[c].local_size();
    for (int j = -1; j < n[1] + 1; ++j)
      for (int i = -1; i < n[0] + 1; ++i)
        if (i < 0 || j < 0 || i >= n[0] || j >= n[1]) {
          hx[c](i, j, 0) = 1e6;
          hy[c](i, j, 0) = -1e6;
        }
    upload(hx[c], dx[c]);
    upload(hy[c], dy[c]);
  }
  const double hd = hops.dot(hx, hy);
  const double dd = dops.dot(dx, dy);
  REQUIRE(std::abs(hd - dd) <= 1e-12 * std::abs(hd));

  pfc::solvers::FieldHalo<Host> hhalo(hops, decomp, rank, MPI_COMM_WORLD, full2d());
  pfc::solvers::FieldHalo<Dev> dhalo(dops, decomp, rank, MPI_COMM_WORLD, full2d());
  const HVec &hh = hhalo(hx);
  const DVec &dh = dhalo(dx);
  REQUIRE(rel(hh[0], const_cast<DField &>(dh[0])) < 1e-15);
  REQUIRE(rel(hh[1], const_cast<DField &>(dh[1])) < 1e-15);

  HField hdx(domain, box, 1), hdy(domain, box, 1);
  DField ddx(domain, box, 1), ddy(domain, box, 1);
  each(hdx, [&](int i, int j, int k) {
    const auto g = hdx.global(i, j, k);
    const double x = g[0] * h, y = g[1] * h;
    hdx(i, j, k) = kappa(x + h, y + 0.5 * h);
    hdy(i, j, k) = kappa(x + 0.5 * h, y + h);
  });
  {
    pfc::comm::HaloExchange<Host, double> ex({&hdx, &hdy}, decomp, rank,
                                             MPI_COMM_WORLD, full2d());
    ex.exchange();
  }
  upload(hdx, ddx);
  upload(hdy, ddy);
  {
    pfc::comm::HaloExchange<Dev, double> ex({&ddx, &ddy}, decomp, rank,
                                            MPI_COMM_WORLD, full2d());
    ex.exchange();
  }

  auto host_action = [&](const HVec &v, HVec &w) {
    const HVec &u = hhalo(v);
    apply_host(u[0], u[1], hdx, hdy, w[0], w[1]);
  };
  auto dev_action = [&](const DVec &v, DVec &w) {
    const DVec &u = dhalo(v);
    HField a = HField(domain, box, 1), b = HField(domain, box, 1);
    HField o0 = HField(domain, box, 1), o1 = HField(domain, box, 1);
    auto pull = [](DField &src, HField &dst) {
      src.with_host_read(
          [&](const double *p, std::size_t n) { std::copy(p, p + n, dst.data()); });
    };
    // Halo scratch is const at the boundary; the read only refreshes the host
    // mirror.
    pull(const_cast<DField &>(u[0]), a);
    pull(const_cast<DField &>(u[1]), b);
    pull(ddx, hdx); // refresh host mirror of coefficients after the device exchange
    pull(ddy, hdy);
    apply_host(a, b, hdx, hdy, o0, o1);
    auto push = [](const HField &src, DField &dst) {
      dst.with_host_view([&](double *p, std::size_t n) {
        std::copy(src.data(), src.data() + n, p);
      });
      dst.sync_to_device();
    };
    push(o0, w[0]);
    push(o1, w[1]);
  };

  HVec hb = hops.make(hops.size()), hxsol = hops.make(hops.size());
  DVec db = dops.make(dops.size()), dxsol = dops.make(dops.size());
  for (int c = 0; c < 2; ++c)
    each(hb[c], [&](int i, int j, int k) {
      const auto g = hb[c].global(i, j, k);
      hb[c](i, j, k) = std::sin(0.2 * (g[0] + 1 + c)) * std::cos(0.15 * g[1]);
    });
  for (int c = 0; c < 2; ++c) upload(hb[c], db[c]);

  pfc::solvers::KrylovControls controls;
  controls.preconditioning = pfc::solvers::Preconditioning::None;
  controls.relative_tolerance = 1e-8;
  controls.maximum_iterations = 400;
  const auto hc = pfc::solvers::cg(hops, host_action, hb, hxsol, controls);
  const auto dc = pfc::solvers::cg(dops, dev_action, db, dxsol, controls);
  REQUIRE(hc.converged());
  REQUIRE(dc.converged());
  REQUIRE(hc.iterations == dc.iterations);
  REQUIRE(rel(hxsol[0], dxsol[0]) < 1e-8);
  REQUIRE(rel(hxsol[1], dxsol[1]) < 1e-8);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
