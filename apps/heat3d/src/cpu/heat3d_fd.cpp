// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_fd.cpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — compact
 *        educational FD driver, **everything in one file**.
 *
 * @details
 * Single self-contained translation unit you can read top-to-bottom:
 * MPI lifecycle, CLI parsing, geometry, halo exchanger, residual field,
 * the Euler time loop, and the L2-vs-analytic report all live here.
 * The five lines under `// 6. Time loop` are the math; everything else
 * is the setup the math depends on.
 *
 * The kernel pieces it composes (and does **not** re-implement):
 *
 *  - `pfc::field::LocalField<double>` — owned bulk storage with halos.
 *  - `pfc::halo::FaceHalos<double>` + `pfc::SparseHaloExchanger<double>`
 *    built by `pfc::halo::make_structured_halos<double>(...)` — non-blocking,
 *    axis-aligned, 6-face MPI halo exchange (with `pfc::halo::copy_to_face_layout`
 *    refilling the array-of-six face buffers the Laplacian kernels read).
 *  - `pfc::field::FdGradient<HeatGrads>` (built by
 *    `pfc::field::create<HeatGrads>(u, order)`) — per-point central FD
 *    evaluator that fills exactly the slots `HeatGrads` declares.
 *  - `pfc::sim::DuField<G, Eval>` — `du.apply(rhs)` triggers a halo
 *    exchange (captured below) and runs the per-point `for_each_interior`
 *    loop into an internal residual buffer; `dt * du` returns a
 *    `ScaledField` proxy that `LocalField::operator+=` axpys back into u.
 *
 * The companion drivers (`heat3d_fd_scratch`, `heat3d_fd_manual`) walk
 * the same physics with progressively more plumbing on the page; the
 * `apps/heat3d/README.md` ladder explains what each one teaches.
 */

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <ostream>
#include <utility>

#include <mpi.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/sparse_halo_exchange.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/simulation/du_field.hpp>
#include <openpfc/runtime/common/cpu_affinity.hpp>

#include <heat3d/heat_model.hpp>

using heat3d::HeatGrads;
using heat3d::HeatModel;
using heat3d::kD;

// =============================================================================
// CLI PART STARTS HERE — argument parsing for `heat3d_fd <N> <n_steps> <dt>
// <fd_order>`. Pure, no MPI, no OpenPFC. Skip past this block to read the
// solver if the CLI is not what you came for.
// =============================================================================

namespace {

struct RunConfig {
  int N{32};
  int n_steps{100};
  double dt{0.01};
  int fd_order{2};
};

void print_usage(std::ostream &os, const char *exe) {
  os << "Usage:\n  " << exe << " <N> <n_steps> <dt> <fd_order>\n"
     << "  fd_order: even 2,4,...,20 (central Laplacian; halo width = order/2)\n";
}

std::optional<RunConfig> parse_cli(int argc, char **argv) {
  if (argc < 5) return std::nullopt;
  RunConfig c;
  c.N = std::atoi(argv[1]);
  c.n_steps = std::atoi(argv[2]);
  c.dt = std::atof(argv[3]);
  c.fd_order = std::atoi(argv[4]);
  if (c.N < 8 || c.n_steps < 1 || c.dt <= 0.0) return std::nullopt;
  if (c.fd_order < 2 || c.fd_order > 20 || (c.fd_order % 2) != 0)
    return std::nullopt;
  return c;
}

// =============================================================================
// SOLVER — geometry + storage + time loop + L2 report. The math lives in
// step 6; the rest is the per-rank scaffolding it depends on.
// =============================================================================

void run_fd(const RunConfig &cfg, int rank, int nproc) {
  // 1. Physics. `HeatModel::initial_condition` is a `(x,y,z) -> u` lambda;
  //    `heat3d::HeatGrads` is the per-point grads aggregate the kernel
  //    evaluators introspect to know which derivatives to fill.
  HeatModel model;

  // 2. Global world + per-rank decomposition.
  const auto world = pfc::world::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                                        pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                        pfc::GridSpacing({1.0, 1.0, 1.0}));
  const auto decomp = pfc::decomposition::create(world, nproc);

  // 3. Storage + halo exchanger. Halo width = order/2 so the central
  //    stencil's most distant neighbour is a halo cell on either side.
  const int hw = cfg.fd_order / 2;
  auto u = pfc::field::LocalField<double>::from_subdomain(decomp, rank, hw);
  auto face_halos = pfc::halo::allocate_face_halos<double>(decomp, rank, hw);
  pfc::SparseHaloExchanger<double> exchanger(
      MPI_COMM_WORLD, rank,
      pfc::halo::make_structured_halos<double>(decomp, rank, hw));

  // 4. Residual field. The captured lambda is the "prepare parent" hook
  //    that `du.apply(...)` calls before evaluating the RHS — for FD this
  //    is the MPI halo exchange; for spectral it would be a no-op.
  auto eval = pfc::field::create<HeatGrads>(u, cfg.fd_order);
  pfc::sim::DuField<HeatGrads, decltype(eval)> du(u.size(), std::move(eval), [&]() {
    exchanger.exchange_halos(u.data(), u.size());
    pfc::halo::copy_to_face_layout(exchanger, face_halos);
  });

  // 5. Initial condition: \f$u(x,y,z,0) = \exp(-|x|^2/(4D))\f$ from the model.
  u.apply(model.initial_condition);

  // 6. Time loop — explicit Euler, point-wise RHS. The three-line body is
  //    the full algorithm; `du.apply` hides the halo exchange + stencil
  //    sweep, the operators hide the axpy. Wall time is the slowest rank.
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step) {
    du.apply([](const HeatGrads &g) { return kD * (g.xx + g.yy + g.zz); });
    u += cfg.dt * du;
    t += cfg.dt;
  }
  const double local_elapsed = MPI_Wtime() - t_start;
  double max_elapsed = 0.0;
  MPI_Allreduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  (void)t; // autonomous heat equation; the running clock is reported by t_final

  // 7. L2-vs-analytic report. Closed form on \f$\mathbb{R}^3\f$:
  //    \f$u(x,t) = (1+t)^{-3/2}\,\exp(-|x|^2/(4D(1+t)))\f$ for the same IC.
  double sum_err2 = 0.0;
  const double t_final = static_cast<double>(cfg.n_steps) * cfg.dt;
  const double s_t = 1.0 + t_final;
  u.for_each_interior([&](double x, double y, double z, double v) {
    const double r2 = x * x + y * y + z * z;
    const double u_exact = std::pow(s_t, -1.5) * std::exp(-r2 / (4.0 * kD * s_t));
    const double e = v - u_exact;
    sum_err2 += e * e;
  });
  double g_err2 = 0.0;
  MPI_Reduce(&sum_err2, &g_err2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const double rms =
        std::sqrt(g_err2 / static_cast<double>(cfg.N * cfg.N * cfg.N));
    std::cout << "heat3d method=fd N=" << cfg.N << " n_steps=" << cfg.n_steps
              << " dt=" << cfg.dt << " D=" << kD << " mpi_ranks=" << nproc
              << " fd_order=" << cfg.fd_order;
#if defined(_OPENMP)
    std::cout << " omp_max_threads=" << omp_get_max_threads()
              << " omp_get_num_procs()=" << omp_get_num_procs();
#endif
    std::cout << "\ntiming_s=" << max_elapsed << " avg_step_time_s="
              << max_elapsed / static_cast<double>(cfg.n_steps)
              << " (MPI_MAX across ranks)\n"
              << "l2_error_vs_R3_analytic_rms=" << rms
              << " (periodic; interior L2 vs infinite-domain Gaussian)\n";
  }
}

} // namespace

// =============================================================================
// MPI ENTRY — open MPI, parse, run, close. Any std::exception escaping
// `run_fd` is logged and aborted via MPI_Abort so peer ranks don't hang
// in subsequent collectives.
// =============================================================================

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // Single-rank Linux affinity rescue: lets OpenMP use every online CPU
  // when `mpirun -n 1` would otherwise inherit a one-CPU mask. No-op for
  // multi-rank jobs and on non-Linux platforms.
  pfc::runtime::reset_cpu_affinity_if_single_mpi_rank(nproc);

  const auto cfg = parse_cli(argc, argv);
  if (!cfg) {
    if (rank == 0) print_usage(std::cerr, argc >= 1 ? argv[0] : "heat3d_fd");
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  try {
    run_fd(*cfg, rank, nproc);
  } catch (const std::exception &e) {
    std::cerr << "(rank " << rank << "): " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
