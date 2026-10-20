// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// NodalMultigrid on the device reproduces the host operator, V-cycle and PCG
// (#544). Runs on any number of ranks, one device per rank, e.g.
//   mpiexec -n 2 test_nodal_multigrid_cuda
// (each rank selects device rank % count).

#include "mechanics_backend.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string_view>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/solvers/field_operations.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/nodal_multigrid.hpp>
#include <openpfc/solvers/nodal_dirichlet_multigrid.hpp>
#include <openpfc/solvers/nodal_dirichlet_device_operator.hpp>
#include <openpfc/solvers/nodal_dirichlet_device_transfers.hpp>
#include <openpfc/solvers/nodal_dirichlet_device_vcycle.hpp>
#include <openpfc/solvers/nodal_dirichlet_kernels.hpp>
#include <openpfc/runtime/gpu/nodal_multigrid_gpu.hpp>

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


TEST_CASE("CUDA/HIP Dirichlet Q1 row launcher matches shared host kernel",
          "[nodal_dirichlet][device]") {
  if (!mechanics_test::available()) SKIP("GPU unavailable");
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) SKIP("single-rank device operator parity only");
  const pfc::Int3 n{15, 15, 31};
  const pfc::Real3 h{1.0 / 16, 1.3 / 16, 0.7 / 32};
  const auto dom = pfc::domain::with_spacing(n, h, {false, false, false});
  const auto dec = pfc::decomposition::create(dom, pfc::Int3{1, 1, 1});
  const auto box = pfc::decomposition::local_box(dec, 0);
  HostF host(dom, box, 2);
  DevF device(dom, box, 2), device_out(dom, box, 2);
  std::fill(host.vec().begin(), host.vec().end(), 8888.0);
  const auto local = host.local_size();
  for (int k = 0; k < local[2]; ++k)
    for (int j = 0; j < local[1]; ++j)
      for (int i = 0; i < local[0]; ++i) {
        const auto g = host.global(i, j, k);
        host(i, j, k) = value(8, g[0], g[1], g[2], 21u);
      }
  device.with_host_view([&](double *ptr, std::size_t extent) {
    REQUIRE(extent == host.size());
    std::copy(host.data(), host.data() + host.size(), ptr);
  });
  device.sync_to_device();

  pfc::solvers::nmg::Level shape;
  for (int a = 0; a < 3; ++a) {
    shape.n[a] = n[a];
    shape.low[a] = 0;
    shape.gn[a] = n[a];
  }
  const double spacing[3] = {h[0], h[1], h[2]};
  const auto weights =
      pfc::solvers::nmg::dirichlet::make_weights(spacing, 1.0, 0.41);
#if defined(OPENPFC_TEST_MECHANICS_HIP)
  pfc::solvers::gpu::nmg::dirichlet_apply_hip(
      shape, weights, device.data(), device_out.data());
#else
  pfc::solvers::gpu::nmg::dirichlet_apply_cuda(
      shape, weights, device.data(), device_out.data());
#endif
  device_out.note_device_write();

  double delta_sq = 0.0, norm_sq = 0.0;
  device_out.with_host_read([&](const double *ptr, std::size_t) {
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const double expected = pfc::solvers::nmg::dirichlet::action(
              shape, weights, host.data(), i, j, k);
          const double actual = ptr[device_out.idx(i, j, k)];
          delta_sq += (actual - expected) * (actual - expected);
          norm_sq += expected * expected;
        }
  });
  REQUIRE(std::sqrt(delta_sq / norm_sq) <= 1e-13);

  // Fused damped Jacobi: read the old owned+halo field but write to a
  // distinct new buffer, exactly as the host symmetric smoother does.
  DevF rhs(dom, box, 2), next(dom, box, 2);
  rhs.with_host_view([&](double *p, std::size_t size) {
    std::fill(p, p + size, 0.0);
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
          p[rhs.idx(i, j, k)] = 0.17 + 0.01 * (i + 2 * j + 3 * k);
  });
  rhs.sync_to_device();
#if defined(OPENPFC_TEST_MECHANICS_HIP)
  pfc::solvers::gpu::nmg::dirichlet_jacobi_hip(
      shape, weights, device.data(), rhs.data(), next.data(), 0.19);
#else
  pfc::solvers::gpu::nmg::dirichlet_jacobi_cuda(
      shape, weights, device.data(), rhs.data(), next.data(), 0.19);
#endif
  next.note_device_write();
  double jacobi_error = 0.0;
  next.with_host_read([&](const double *p, std::size_t) {
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const double b = 0.17 + 0.01 * (i + 2 * j + 3 * k);
          const double expected = pfc::solvers::nmg::dirichlet::jacobi(
              shape, weights, host.data(), host(i, j, k), b, 0.19, i, j, k);
          jacobi_error = std::max(jacobi_error,
                                   std::abs(expected - p[next.idx(i, j, k)]));
        }
  });
  REQUIRE(jacobi_error < 1e-12);

  // Adjacent Q1 free-node spaces. The launchers own only numerical point
  // arithmetic; the caller's Field handles device residency and padding.
  const pfc::Int3 nc{7, 7, 15};
  const auto domc = pfc::domain::with_spacing(
      nc, {2 * h[0], 2 * h[1], 2 * h[2]},
      {false, false, false});
  const auto decc = pfc::decomposition::create(domc, pfc::Int3{1, 1, 1});
  const auto boxc = pfc::decomposition::local_box(decc, 0);
  DevF coarse(domc, boxc, 2), reduced(domc, boxc, 2), prolongated(dom, box, 2);
  pfc::solvers::nmg::Level coarseshape;
  for (int a = 0; a < 3; ++a) {
    coarseshape.n[a] = nc[a];
    coarseshape.gn[a] = nc[a];
    coarseshape.low[a] = 0;
  }
  std::vector<double> coarse_host(
      pfc::solvers::nmg::padded_size(
          coarseshape, pfc::solvers::nmg::dirichlet::kHalo), 1111.0);
  for (int k = 0; k < nc[2]; ++k)
    for (int j = 0; j < nc[1]; ++j)
      for (int i = 0; i < nc[0]; ++i)
        coarse_host[pfc::solvers::nmg::index(
            coarseshape, pfc::solvers::nmg::dirichlet::kHalo, i, j, k)] =
            value(6, i, j, k, 35u);
  coarse.with_host_view([&](double *p, std::size_t size) {
    REQUIRE(size == coarse_host.size());
    std::copy(coarse_host.begin(), coarse_host.end(), p);
  });
  coarse.sync_to_device();
  prolongated.with_host_view([&](double *p, std::size_t size) {
    std::fill(p, p + size, 0.0);
  });
  prolongated.sync_to_device();
#if defined(OPENPFC_TEST_MECHANICS_HIP)
  pfc::solvers::gpu::nmg::dirichlet_restrict_hip(
      shape, device.data(), coarseshape, reduced.data());
  pfc::solvers::gpu::nmg::dirichlet_prolong_hip(
      shape, prolongated.data(), coarseshape, coarse.data());
#else
  pfc::solvers::gpu::nmg::dirichlet_restrict_cuda(
      shape, device.data(), coarseshape, reduced.data());
  pfc::solvers::gpu::nmg::dirichlet_prolong_cuda(
      shape, prolongated.data(), coarseshape, coarse.data());
#endif
  reduced.note_device_write();
  prolongated.note_device_write();
  double restriction_error = 0.0, prolongation_error = 0.0;
  reduced.with_host_read([&](const double *p, std::size_t) {
    for (int k = 0; k < nc[2]; ++k)
      for (int j = 0; j < nc[1]; ++j)
        for (int i = 0; i < nc[0]; ++i) {
          const double expected = pfc::solvers::nmg::dirichlet::restrict_pt(
              shape, host.data(), i, j, k);
          restriction_error = std::max(
              restriction_error, std::abs(expected - p[reduced.idx(i, j, k)]));
        }
  });
  prolongated.with_host_read([&](const double *p, std::size_t) {
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const double expected = pfc::solvers::nmg::dirichlet::prolong_p(
              coarseshape, coarse_host.data(), i, j, k);
          prolongation_error = std::max(
              prolongation_error,
              std::abs(expected - p[prolongated.idx(i, j, k)]));
        }
  });
  REQUIRE(restriction_error < 1e-12);
  REQUIRE(prolongation_error < 1e-12);
}


TEST_CASE("Dirichlet Q1 device FieldHalo operator matches host across rank bricks",
          "[nodal_dirichlet][device]") {
  if (!mechanics_test::available()) SKIP("GPU unavailable");

  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 1 && ranks != 2 && ranks != 4)
    SKIP("explicit device decomposition gate covers 1, 2 or 4 ranks");

  // A noncubic physical box and a two-level nested Q1 free-node hierarchy.
  // NodalDirichletMultigrid stays the independently qualified host oracle;
  // the device adapter owns no solver hierarchy and uses OpenPFC FieldHalo.
  const pfc::Int3 n{31, 31, 31};
  const pfc::Real3 h{1.0 / 32, 1.3 / 32, 0.7 / 32};
  const auto domain = pfc::domain::with_spacing(
      n, h, {false, false, false});
  const auto decomp = ranks == 4
      ? pfc::decomposition::create(domain, pfc::Int3{2, 2, 1})
      : pfc::decomposition::create(domain, ranks);
  const auto box = pfc::decomposition::local_box(decomp, rank);

  pfc::solvers::NodalDirichletMultigrid host_solver(
      decomp, rank, MPI_COMM_WORLD);
  pfc::solvers::NodalDirichletDeviceOperator<mechanics_test::Space> device_action(
      decomp, rank, MPI_COMM_WORLD);
  REQUIRE(device_action.local_unknowns() ==
          static_cast<std::size_t>(box.size[0]) * box.size[1] * box.size[2]);

  // Different external halo widths and repeated field updates probe both
  // the owned-copy staging and the residency/halo invalidation contract.
  for (const auto widths : {std::array<int, 2>{1, 1},
                            std::array<int, 2>{0, 2},
                            std::array<int, 2>{2, 0}}) {
    HostF hi(domain, box, widths[0]), ho(domain, box, widths[1]);
    DevF di(domain, box, widths[0]), doutput(domain, box, widths[1]);
    for (int sample_kind : {0, 1}) {
      fill_both(hi, di, [=](int gx, int gy, int gz) {
        return value(sample_kind, gx, gy, gz, 601u);
      });
      host_solver.apply_operator(hi, ho);
      device_action.apply_operator(di, doutput);
      INFO("MPI ranks " << ranks << ", input halo " << widths[0]
                         << ", output halo " << widths[1]
                         << ", sample " << sample_kind);
      REQUIRE(rel_diff(ho, doutput) <= 1e-12);
    }
  }

  // The operand can alias the destination because the shared halo-2
  // scratch captures its owned values before device output is written.
  HostF alias_host(domain, box, 1);
  DevF alias_device(domain, box, 1);
  fill_both(alias_host, alias_device,
            [](int gx, int gy, int gz) {
              return value(12, gx, gy, gz, 706u);
            });
  HostF expected(domain, box, 1);
  host_solver.apply_operator(alias_host, expected);
  device_action.apply_operator(alias_device, alias_device);
  REQUIRE(rel_diff(expected, alias_device) <= 1e-12);

  // A device-resident unpreconditioned CG reference exercises the existing
  // FieldOperations and the rank-owned action, not a premature device V-cycle.
  // The exact discrete solution is known independently from how b is formed.
  pfc::solvers::FieldOperations<mechanics_test::Space> ops(
      domain, box, 1, 1, MPI_COMM_WORLD);
  auto exact = ops.make(ops.size());
  auto rhs = ops.make(ops.size());
  auto solution = ops.make(ops.size());
  HostF host_exact(domain, box, 1);
  fill_both(host_exact, exact[0],
            [](int gx, int gy, int gz) {
              return value(19, gx, gy, gz, 713u);
            });
  device_action.apply_operator(exact[0], rhs[0]);
  pfc::solvers::KrylovControls controls;
  controls.preconditioning = pfc::solvers::Preconditioning::None;
  controls.relative_tolerance = 1e-8;
  controls.maximum_iterations = 400;
  const auto result = pfc::solvers::cg(
      ops, [&](const auto &u, auto &out) {
        device_action.apply_operator(u[0], out[0]);
      }, rhs, solution, controls);
  REQUIRE(result.converged());
  REQUIRE(rel_diff(host_exact, solution[0]) <= 5e-6);
}


TEST_CASE("Dirichlet Q1 device nested FieldHalo P/Pt match host bricks",
          "[nodal_dirichlet][device]") {
  if (!mechanics_test::available()) SKIP("GPU unavailable");
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 1 && ranks != 2 && ranks != 4)
    SKIP("device nested-Q1 MPI parity covers 1, 2 or 4 ranks");

  // Check the 2:1 element-count elongated family as well as cubic Q1.
  for (const pfc::Int3 nf :
       {pfc::Int3{31, 31, 31}, pfc::Int3{31, 31, 63}}) {
  const pfc::Int3 nc{(nf[0] - 1) / 2, (nf[1] - 1) / 2, (nf[2] - 1) / 2};
  const pfc::Real3 hf{1.0 / 32, 1.3 / 32,
                      0.7 / static_cast<double>(nf[2] + 1)};
  const pfc::Real3 hc{2 * hf[0], 2 * hf[1], 2 * hf[2]};
  const auto df = pfc::domain::with_spacing(nf, hf, {false, false, false});
  const auto dc = pfc::domain::with_spacing(nc, hc, {false, false, false});
  const pfc::Int3 process_grid = ranks == 4 ? pfc::Int3{2, 2, 1}
                                            : pfc::Int3{1, 1, ranks};
  const auto decomp_f = pfc::decomposition::create(df, process_grid);
  const auto decomp_c = pfc::decomposition::create(dc, process_grid);
  const auto bf = pfc::decomposition::local_box(decomp_f, rank);
  const auto bc = pfc::decomposition::local_box(decomp_c, rank);

  pfc::solvers::NodalDirichletDeviceTransfers<mechanics_test::Space> transfer(
      decomp_f, decomp_c, rank, MPI_COMM_WORLD);
  REQUIRE(transfer.fine_owned_box().size == bf.size);
  REQUIRE(transfer.coarse_owned_box().size == bc.size);

  pfc::solvers::FieldOperations<pfc::HostSpace> fops(df, bf, 1, 2,
                                                     MPI_COMM_WORLD);
  pfc::solvers::FieldOperations<pfc::HostSpace> cops(dc, bc, 1, 2,
                                                     MPI_COMM_WORLD);
  pfc::comm::HaloExchangeOptions options;
  options.connectivity = pfc::comm::HaloConnectivity::Full;
  pfc::solvers::FieldHalo<pfc::HostSpace> fh(
      fops, decomp_f, rank, MPI_COMM_WORLD, options);
  pfc::solvers::FieldHalo<pfc::HostSpace> ch(
      cops, decomp_c, rank, MPI_COMM_WORLD, options);
  pfc::solvers::nmg::Level F, C;
  for (int d = 0; d < 3; ++d) {
    F.n[d] = bf.size[d]; F.low[d] = bf.low[d]; F.gn[d] = nf[d];
    C.n[d] = bc.size[d]; C.low[d] = bc.low[d]; C.gn[d] = nc[d];
  }

  HostF source_h(df, bf, 0), correction_h(dc, bc, 1);
  DevF source_d(df, bf, 0), correction_d(dc, bc, 1);
  HostF host_coarse(dc, bc, 0), host_fine(df, bf, 1);
  DevF device_coarse(dc, bc, 0), device_fine(df, bf, 1);

  for (int trial = 0; trial < 2; ++trial) {
    fill_both(source_h, source_d, [trial](int x, int y, int z) {
      return value(10 + trial, x, y, z, 1201u);
    });
    fill_both(correction_h, correction_d, [trial](int x, int y, int z) {
      return value(20 + trial, x, y, z, 1601u);
    });
    fill_both(host_fine, device_fine, [trial](int x, int y, int z) {
      return value(30 + trial, x, y, z, 1801u);
    });

    auto &rf = fh.scratch()[0];
    std::fill(rf.vec().begin(), rf.vec().end(), -1024.0);
    for (int z = 0; z < bf.size[2]; ++z)
      for (int y = 0; y < bf.size[1]; ++y)
        for (int x = 0; x < bf.size[0]; ++x)
          rf(x, y, z) = source_h(x, y, z);
    fh.exchange();

    auto &uc = ch.scratch()[0];
    std::fill(uc.vec().begin(), uc.vec().end(), 2048.0);
    for (int z = 0; z < bc.size[2]; ++z)
      for (int y = 0; y < bc.size[1]; ++y)
        for (int x = 0; x < bc.size[0]; ++x)
          uc(x, y, z) = correction_h(x, y, z);
    ch.exchange();

    for (int z = 0; z < bc.size[2]; ++z)
      for (int y = 0; y < bc.size[1]; ++y)
        for (int x = 0; x < bc.size[0]; ++x)
          host_coarse(x, y, z) =
              pfc::solvers::nmg::dirichlet::restrict_pt(
                  F, rf.data(), bc.low[0] + x, bc.low[1] + y,
                  bc.low[2] + z);

    for (int z = 0; z < bf.size[2]; ++z)
      for (int y = 0; y < bf.size[1]; ++y)
        for (int x = 0; x < bf.size[0]; ++x)
          host_fine(x, y, z) +=
              pfc::solvers::nmg::dirichlet::prolong_p(
                  C, uc.data(), bf.low[0] + x, bf.low[1] + y,
                  bf.low[2] + z);

    transfer.restrict_pt(source_d, device_coarse);
    transfer.prolong_add(correction_d, device_fine);
    INFO("MPI ranks " << ranks << " process grid " << process_grid[0]
                      << "x" << process_grid[1] << "x" << process_grid[2]
                      << ", trial " << trial);
    REQUIRE(rel_diff(host_coarse, device_coarse) <= 1e-12);
    REQUIRE(rel_diff(host_fine, device_fine) <= 1e-12);
  }
  }
}



TEST_CASE("CUDA/HIP two-level Dirichlet Q1 V-cycle matches host and preconditions CG",
          "[nodal_dirichlet][device]") {
  if (!mechanics_test::available()) SKIP("GPU unavailable");
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 1 && ranks != 2 && ranks != 4)
    SKIP("two-level device V-cycle gate covers 1/2/4 ranks");

  const pfc::Int3 nf{15, 15, 15}, nc{7, 7, 7};
  const pfc::Real3 hf{1.0 / 16, 1.3 / 16, 0.7 / 16};
  const pfc::Real3 hc{2 * hf[0], 2 * hf[1], 2 * hf[2]};
  const auto fd = pfc::domain::with_spacing(nf, hf, {false, false, false});
  const auto cd = pfc::domain::with_spacing(nc, hc, {false, false, false});
  const pfc::Int3 proc = ranks == 4 ? pfc::Int3{2, 2, 1}
                                    : pfc::Int3{1, 1, ranks};
  const auto fdec = pfc::decomposition::create(fd, proc);
  const auto cdec = pfc::decomposition::create(cd, proc);
  const auto box = pfc::decomposition::local_box(fdec, rank);
  using Vcycle = pfc::solvers::NodalDirichletDeviceTwoLevelVcycle<
      mechanics_test::Space>;
  Vcycle device_vcycle(fdec, cdec, rank, MPI_COMM_WORLD);
  pfc::solvers::NodalDirichletMultigrid host_vcycle(
      fdec, rank, MPI_COMM_WORLD);
  pfc::solvers::NodalDirichletDeviceOperator<mechanics_test::Space> A(
      fdec, rank, MPI_COMM_WORLD);
  REQUIRE(device_vcycle.coarse_unknowns() == 343);

  HostF host_rhs(fd, box, 1), host_out(fd, box, 1);
  DevF device_rhs(fd, box, 1), device_out(fd, box, 1);
  for (int trial : {0, 1}) {
    fill_both(host_rhs, device_rhs, [trial](int gx, int gy, int gz) {
      return value(40 + trial, gx, gy, gz, 3301u);
    });
    host_vcycle.apply(host_rhs, host_out);
    device_vcycle.apply(device_rhs, device_out);
    INFO("Dirichlet GPU two-level V-cycle, ranks " << ranks << ", trial " << trial);
    REQUIRE(rel_diff(host_out, device_out) < 2e-10);
  }

  // A fixed SPD preconditioner must be symmetric and positive for
  // independent fields, not just produce a small residual after PCG.
  pfc::solvers::FieldOperations<mechanics_test::Space> ops(
      fd, box, 1, 1, MPI_COMM_WORLD);
  auto u = ops.make(ops.size()), v = ops.make(ops.size());
  auto mu = ops.make(ops.size()), mv = ops.make(ops.size());
  HostF hu(fd, box, 1), hv(fd, box, 1);
  fill_both(hu, u[0], [](int x, int y, int z) {
    return value(61, x, y, z, 5001u);
  });
  fill_both(hv, v[0], [](int x, int y, int z) {
    return value(62, x, y, z, 6001u);
  });
  device_vcycle.apply(u[0], mu[0]);
  device_vcycle.apply(v[0], mv[0]);
  const double uMv = ops.dot(u, mv), vMu = ops.dot(v, mu);
  REQUIRE(std::abs(uMv - vMu) <=
          1e-8 * std::max({1.0, std::abs(uMv), std::abs(vMu)}));
  REQUIRE(ops.dot(u, mu) > 0.0);
  REQUIRE(ops.dot(v, mv) > 0.0);

  // This independently checks the preconditioner contract in the actual
  // CUDA/HIP FieldOperations and OpenPFC CG implementation, including
  // admission by the original unpreconditioned true global residual.
  auto rhs = ops.make(ops.size());
  auto exact = ops.make(ops.size());
  auto solution = ops.make(ops.size());
  HostF host_exact(fd, box, 1);
  fill_both(host_exact, exact[0], [](int gx, int gy, int gz) {
    return value(53, gx, gy, gz, 4001u);
  });
  A.apply_operator(exact[0], rhs[0]);
  pfc::solvers::KrylovControls ctrl;
  ctrl.relative_tolerance = 1e-10;
  ctrl.maximum_iterations = 80;
  const auto result = pfc::solvers::cg(
      ops,
      [&](const auto &u, auto &out) { A.apply_operator(u[0], out[0]); },
      rhs, solution, ctrl,
      [&](const auto &r, auto &z, int) {
        device_vcycle.apply(r[0], z[0]);
      });
  INFO("two-level Dirichlet PCG on " << ranks
                                    << " ranks, iterations " << result.iterations);
  REQUIRE(result.converged());
  REQUIRE(result.residual <= 1e-10 * result.rhs_norm);
  REQUIRE(rel_diff(host_exact, solution[0]) < 1e-7);
}



TEST_CASE("Dirichlet Q1 three-level device V-cycle equals host and preconditions CG",
          "[nodal_dirichlet][device]") {
  if (!mechanics_test::available()) SKIP("GPU unavailable");

  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 1 && ranks != 2 && ranks != 4)
    SKIP("nested GPU V-cycle gate covers 1/2/4 ranks");

  using Vcycle = pfc::solvers::NodalDirichletDeviceVcycle<
      mechanics_test::Space>;
  using Ops = pfc::solvers::FieldOperations<mechanics_test::Space>;

  for (const bool elongated : {false, true}) {
    if (elongated && ranks == 4) continue;
    const pfc::Int3 nf = elongated ? pfc::Int3{31, 31, 63}
                                   : pfc::Int3{31, 31, 31};
    const pfc::Int3 nm = elongated ? pfc::Int3{15, 15, 31}
                                   : pfc::Int3{15, 15, 15};
    const pfc::Int3 nc = elongated ? pfc::Int3{7, 7, 15}
                                   : pfc::Int3{7, 7, 7};
    const pfc::Real3 hf{1.0 / 32, 1.3 / 32,
                        elongated ? 0.7 / 64 : 0.7 / 32};
    const pfc::Real3 hm{2 * hf[0], 2 * hf[1], 2 * hf[2]};
    const pfc::Real3 hc{2 * hm[0], 2 * hm[1], 2 * hm[2]};
    const auto fd = pfc::domain::with_spacing(nf, hf, {false, false, false});
    const auto md = pfc::domain::with_spacing(nm, hm, {false, false, false});
    const auto cd = pfc::domain::with_spacing(nc, hc, {false, false, false});
    const pfc::Int3 proc = ranks == 4 ? pfc::Int3{2, 2, 1}
                                     : pfc::Int3{1, 1, ranks};
    const auto fdec = pfc::decomposition::create(fd, proc);
    const auto mdec = pfc::decomposition::create(md, proc);
    const auto cdec = pfc::decomposition::create(cd, proc);
    const auto box = pfc::decomposition::local_box(fdec, rank);

    // Old two-level API is retained, while the same implementation now
    // owns one extra GPU-resident intermediate level by recursion.
    Vcycle vcycle(fdec, mdec, cdec, rank, MPI_COMM_WORLD);
    REQUIRE(vcycle.levels() == 3);
    REQUIRE(vcycle.coarse_unknowns() == (elongated ? 735u : 343u));
    pfc::solvers::NodalDirichletMultigrid reference(
        fdec, rank, MPI_COMM_WORLD);
    pfc::solvers::NodalDirichletDeviceOperator<mechanics_test::Space> A(
        fdec, rank, MPI_COMM_WORLD);

    HostF host_rhs(fd, box, 1), host_out(fd, box, 1);
    DevF dev_rhs(fd, box, 1), dev_out(fd, box, 1);
    for (int case_id : {0, 1}) {
      fill_both(host_rhs, dev_rhs, [case_id](int x, int y, int z) {
        return value(70 + case_id, x, y, z, 719u);
      });
      reference.apply(host_rhs, host_out);
      vcycle.apply(dev_rhs, dev_out);
      INFO("GPU three-level Q1, elongated " << elongated
            << ", ranks " << ranks << ", input " << case_id);
      REQUIRE(rel_diff(host_out, dev_out) <= 4e-10);
    }

    Ops ops(fd, box, 1, 1, MPI_COMM_WORLD);
    auto u = ops.make(ops.size()), v = ops.make(ops.size());
    auto mu = ops.make(ops.size()), mv = ops.make(ops.size());
    HostF hu(fd, box, 1), hv(fd, box, 1);
    fill_both(hu, u[0], [](int x, int y, int z) {
      return value(84, x, y, z, 811u);
    });
    fill_both(hv, v[0], [](int x, int y, int z) {
      return value(85, x, y, z, 977u);
    });
    vcycle.apply(u[0], mu[0]);
    vcycle.apply(v[0], mv[0]);
    const double uMv = ops.dot(u, mv), vMu = ops.dot(v, mu);
    REQUIRE(std::abs(uMv - vMu) <=
            2e-8 * std::max({1.0, std::abs(uMv), std::abs(vMu)}));
    REQUIRE(ops.dot(u, mu) > 0.0);
    REQUIRE(ops.dot(v, mv) > 0.0);

    auto exact = ops.make(ops.size()), rhs = ops.make(ops.size());
    auto solution = ops.make(ops.size());
    HostF host_exact(fd, box, 1);
    fill_both(host_exact, exact[0], [](int x, int y, int z) {
      return value(91, x, y, z, 1103u);
    });
    A.apply_operator(exact[0], rhs[0]);
    pfc::solvers::KrylovControls controls;
    controls.relative_tolerance = 1e-10;
    controls.maximum_iterations = 120;
    const auto result = pfc::solvers::cg(
        ops, [&](const auto &q, auto &out) {
          A.apply_operator(q[0], out[0]);
        }, rhs, solution, controls,
        [&](const auto &r, auto &z, int) { vcycle.apply(r[0], z[0]); });
    INFO("GPU Q1 three-level true-residual PCG, rank count " << ranks
         << ", elongated " << elongated << ", iterations " << result.iterations);
    REQUIRE(result.converged());
    REQUIRE(result.residual <= 1e-10 * result.rhs_norm);
    REQUIRE(rel_diff(host_exact, solution[0]) <= 1e-7);
  }
}


TEST_CASE("CUDA/HIP four-level Dirichlet Q1 hierarchy matches independent host",
          "[nodal_dirichlet][device]") {
  if (!mechanics_test::available()) SKIP("GPU unavailable");
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 1 && ranks != 2 && ranks != 4)
    SKIP("four-level GPU Q1 admission uses 1, 2 or 4 ranks");

  using Vcycle = pfc::solvers::NodalDirichletDeviceVcycle<
      mechanics_test::Space>;
  using Ops = pfc::solvers::FieldOperations<mechanics_test::Space>;

  for (bool tall : {false, true}) {
    const pfc::Int3 nn{63, 63, tall ? 127 : 63};
    const pfc::Real3 h{1.0 / 64, 1.3 / 64,
                        tall ? 0.7 / 128 : 0.7 / 64};
    const auto domain = pfc::domain::create(
        pfc::GridSize(nn), pfc::PhysicalOrigin(h), pfc::GridSpacing(h),
        pfc::Bool3{false, false, false});
    const pfc::Int3 proc = ranks == 4 ? pfc::Int3{2, 2, 1}
                                      : pfc::Int3{1, 1, ranks};
    const auto dec = pfc::decomposition::create(domain, proc);
    const auto levels = Vcycle::make_hierarchy(dec);
    REQUIRE(levels.size() == 4);
    Vcycle gpu(levels, rank, MPI_COMM_WORLD);
    REQUIRE(gpu.levels() == 4);
    REQUIRE(gpu.coarse_unknowns() == (tall ? 735u : 343u));

    pfc::solvers::NodalDirichletMultigrid host(dec, rank, MPI_COMM_WORLD);
    REQUIRE(host.levels() == gpu.levels());
    const auto box = pfc::decomposition::local_box(dec, rank);
    HostF hrhs(domain, box, 0), hcorrected(domain, box, 0);
    DevF drhs(domain, box, 0), dcorrected(domain, box, 0);
    for (int sample : {0, 1}) {
      fill_both(hrhs, drhs, [sample](int x, int y, int z) {
        return value(101 + sample, x, y, z, 2903u);
      });
      host.apply(hrhs, hcorrected);
      gpu.apply(drhs, dcorrected);
      INFO("four-level GPU Dirichlet, tall " << tall
           << ", ranks " << ranks << ", input " << sample);
      REQUIRE(rel_diff(hcorrected, dcorrected) <= 8e-10);
    }

    Ops ops(domain, box, 1, 1, MPI_COMM_WORLD);
    auto u = ops.make(ops.size()), v = ops.make(ops.size());
    auto mu = ops.make(ops.size()), mv = ops.make(ops.size());
    HostF hu(domain, box, 1), hv(domain, box, 1);
    fill_both(hu, u[0], [](int x, int y, int z) {
      return value(106, x, y, z, 3109u);
    });
    fill_both(hv, v[0], [](int x, int y, int z) {
      return value(107, x, y, z, 3119u);
    });
    gpu.apply(u[0], mu[0]);
    gpu.apply(v[0], mv[0]);
    const double uMv = ops.dot(u, mv), vMu = ops.dot(v, mu);
    REQUIRE(std::abs(uMv - vMu) <=
            2e-8 * std::max({1.0, std::abs(uMv), std::abs(vMu)}));
    REQUIRE(ops.dot(u, mu) > 0.0);
    REQUIRE(ops.dot(v, mv) > 0.0);

    pfc::solvers::NodalDirichletDeviceOperator<mechanics_test::Space> A(
        dec, rank, MPI_COMM_WORLD);
    auto exact = ops.make(ops.size()), rhs = ops.make(ops.size());
    auto solution = ops.make(ops.size());
    HostF host_exact(domain, box, 1);
    fill_both(host_exact, exact[0], [](int x, int y, int z) {
      return value(115, x, y, z, 4441u);
    });
    A.apply_operator(exact[0], rhs[0]);
    pfc::solvers::KrylovControls controls;
    controls.relative_tolerance = 1e-10;
    controls.maximum_iterations = 120;
    const auto result = pfc::solvers::cg(
        ops, [&](const auto &q, auto &out) {
          A.apply_operator(q[0], out[0]);
        }, rhs, solution, controls,
        [&](const auto &r, auto &z, int) { gpu.apply(r[0], z[0]); });
    INFO("GPU Q1 four-level true-residual PCG, tall " << tall
         << ", ranks " << ranks << ", iterations " << result.iterations);
    REQUIRE(result.converged());
    REQUIRE(result.residual <= 1e-10 * result.rhs_norm);
    REQUIRE(rel_diff(host_exact, solution[0]) <= 1e-7);
  }
}


int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // On device-less hosted CI the Catch2 tests may SKIP. Hardware acceptance
  // must NEVER report a green result when any requested MPI rank lacks a
  // working CUDA/HIP device. The explicit strict mode fails before Catch2.
  const char *required = std::getenv("OPENPFC_REQUIRE_GPU_TESTS");
  if (required && std::string_view(required) == "1") {
    int available = mechanics_test::available() ? 1 : 0;
    int all_available = 0;
    MPI_Allreduce(&available, &all_available, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (!all_available) {
      if (rank == 0)
        std::cerr << "FAIL: required CUDA/HIP Q1 numerical tests have "
                     "no usable GPU on at least one MPI rank\n";
      MPI_Finalize();
      return 2;
    }
  }
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
