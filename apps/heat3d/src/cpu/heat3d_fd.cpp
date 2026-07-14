// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_fd.cpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — compact
 *        educational FD driver, **everything in one file**.
 *
 * @details
 * Single self-contained translation unit you can read top-to-bottom:
 * MPI lifecycle, CLI parsing, geometry, halo exchanger, residual brick,
 * the Euler time loop, and the L2-vs-analytic report all live here.
 * The seven lines under `// 6. Time loop` are the math; everything else
 * is the setup the math depends on.
 *
 * The driver is intentionally explicit about the **three separate
 * concerns** that drive an FD step — halo exchange, gradient evaluation,
 * and iteration — so a reader can see where each one lives:
 *
 *  - `pfc::field::PaddedBrick<double>` — one contiguous owned-plus-halo-ring
 *    buffer (`[-hw, nx+hw)` indexing along each axis). Storage only;
 *    MPI-unaware. `u` holds the state, `du` holds the residual; both are
 *    plain bricks with the same decomposition / rank / halo width.
 *  - `pfc::communication::PaddedHaloExchanger<double>` — non-blocking
 *    six-face MPI exchange into the brick's halo ring. Constructed from
 *    `u` directly, then driven via `pfc::communication::exchange(halo)`
 *    (blocking one-shot) or `start_exchange` / `finish_exchange` when
 *    overlapping with inner work — no buffer pointer or halo width to
 *    mismatch.
 *  - **This file** defines `HeatGrads` (below) — the three second-
 *    derivative slots `FDGradient` fills; `heat3d/heat_model.hpp` carries
 *    the same aggregate for the other binaries.
 *  - `pfc::gradient::FDGradient<HeatGrads>` — per-point central FD
 *    evaluator bound to `u`; consumed via the free
 *    `pfc::gradient::evaluate(grad, idx)` so the inner loop reads as
 *    "evaluate the gradient at this index, then write the residual".
 *  - `pfc::field::for_each(du, fn)` — sweep every owned cell of `du`,
 *    passing a `pfc::Int3{i, j, k}` to `fn`. The brick already carries
 *    its halo width, so the lambda can hand `idx` straight to `evaluate`.
 *  - `pfc::field::for_each_coords(brick, …)` — every owned cell with
 *    physical `(x, y, z)`; used for the initial condition and the L2
 *    report (Gaussian IC is centred at the origin, well inside the box).
 *
 * For an FFT-safe **unpadded** core plus separated face buffers, use
 * `pfc::sim::stacks::FdCpuStack` (see tests and `heat3d_spectral_pointwise.cpp`).
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
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/runtime/common/cpu_affinity.hpp>

// =============================================================================
// CLI PART STARTS HERE — argument parsing for `heat3d_fd <N> <n_steps> <dt>
// <fd_order>`. Pure, no MPI, no OpenPFC. Skip past this block to read the
// solver if the CLI is not what you came for.
// =============================================================================

namespace {

/**
 * @brief Per-point Laplacian channels for \f$\partial_t u = D\Delta u\f$.
 *
 * OpenPFC's `pfc::gradient::FDGradient<G>` inspects which members `G`
 * declares (`xx`, `yy`, `zz` here) and evaluates exactly those central
 * second derivatives — see `pfc::field::has_xx` / `grad_concepts.hpp`.
 * The shared header `heat3d/heat_model.hpp` defines the same three
 * fields as `heat3d::HeatGrads` so tests and the other drivers stay
 * aligned; we redeclare it here for an "everything in one file" read.
 */
struct HeatGrads {
  double xx{};
  double yy{};
  double zz{};
};

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
  // 1. Physics. This driver fixes D = 1 (same as `heat3d::kD` elsewhere) so the
  //    Gaussian IC and analytic reference match the other heat3d binaries.

  // 2. Global world + per-rank decomposition.
  const auto world = pfc::world::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                                        pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                        pfc::GridSpacing({1.0, 1.0, 1.0}));
  const auto decomp = pfc::decomposition::create(world, nproc);

  // 3. Storage. Two padded bricks share decomp, rank, and halo width so
  //    nothing downstream can disagree with the layout: `u` is the state,
  //    `du` is the residual we accumulate each step. Halo width = order/2
  //    so the central stencil's most distant neighbour is a halo cell on
  //    either side.
  const int hw = cfg.fd_order / 2;
  pfc::field::PaddedBrick<double> u(decomp, rank, hw);
  pfc::field::PaddedBrick<double> du(decomp, rank, hw);

  // 4. Halo exchanger and gradient evaluator, both bound to `u`. The
  //    exchanger reads `u`'s decomp / rank / halo width and captures
  //    `u.data()` once; the evaluator reads the same geometry directly
  //    so changing `cfg.fd_order` here cannot drift away from the brick.
  pfc::communication::PaddedHaloExchanger<double> halo(u, MPI_COMM_WORLD);
  pfc::gradient::FDGradient<HeatGrads> grad(u, cfg.fd_order);

  // 5. Initial condition: \f$u(x,y,z,0) = \exp(-|x|^2/(4D))\f$, D = 1.
  pfc::field::for_each_coords(u, [](double x, double y, double z, double &v) {
    v = std::exp(-(x * x + y * y + z * z) / 4.0);
  });

  // 6. Time loop — explicit Euler, point-wise RHS. Each iteration:
  //      a) refresh `u`'s halo ring (start + finish; no overlap here),
  //      b) sweep `du` and write `kD * Δu` per cell using the gradient
  //         evaluator,
  //      c) axpy `u += dt * du` over the padded buffer.
  //    Wall time is the slowest rank (MPI_MAX).
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step) {
    pfc::communication::exchange(halo);
    pfc::field::for_each(du, [&](const auto &idx) {
      const auto g = pfc::gradient::evaluate(grad, idx);
      du[idx] = g.xx + g.yy + g.zz;
    });
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
  pfc::field::for_each_coords(
      std::as_const(u), [&](double x, double y, double z, const double &v) {
        const double r2 = x * x + y * y + z * z;
        const double u_exact = std::pow(s_t, -1.5) * std::exp(-r2 / (4.0 * s_t));
        const double e = v - u_exact;
        sum_err2 += e * e;
      });
  double g_err2 = 0.0;
  MPI_Reduce(&sum_err2, &g_err2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const double rms =
        std::sqrt(g_err2 / (static_cast<double>(cfg.N) * cfg.N * cfg.N));
    std::cout << "heat3d method=fd N=" << cfg.N << " n_steps=" << cfg.n_steps
              << " dt=" << cfg.dt << " D=1 mpi_ranks=" << nproc
              << " fd_order=" << cfg.fd_order;
#if defined(_OPENMP)
    std::cout << " omp_max_threads=" << omp_get_max_threads()
              << " omp_get_num_procs()=" << omp_get_num_procs();
#endif
    std::cout << "\ntiming_s=" << max_elapsed << " avg_step_time_s="
              << max_elapsed / static_cast<double>(cfg.n_steps)
              << " (MPI_MAX across ranks)\n"
              << "l2_error_vs_R3_analytic_rms=" << rms
              << " (periodic; L2 vs infinite-domain Gaussian over owned cells)\n";
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
