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
 *  - `pfc::data::Field<double, HostSpace>` via `field_from_subdomain` —
 *    padded owned-plus-halo storage for `u` / `du`.
 *  - `pfc::comm::HaloExchange<HostSpace>` bound to that Field.
 *  - `pfc::gradient::FDGradient<HeatGrads>` bound to the same Field.
 *  - `Field::for_each_owned` / `Field::apply` for residual and IC/L2.
 *
 * For an FFT-safe **unpadded** core plus separated face buffers, use
 * `pfc::sim::stacks::FdCpuStack` (see tests and `heat3d_spectral_pointwise.cpp`).
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

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/scaled_field.hpp>
#include <openpfc/runtime/common/cpu_affinity.hpp>
#include <heat3d/reporting.hpp>

using pfc::field::operator*;
using pfc::field::operator+=;

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

/**
 * @brief Heat equation solver using explicit finite-difference forward Euler integration
 *
 * @details This implementation solves the three-dimensional heat equation
 * ∂T/∂t = α * ∇²T on a fully periodic 3D box using explicit forward Euler
 * time integration with a 7-point finite-difference Laplacian stencil.
 *
 * @par Integrator method
 * Concrete integrator: explicit forward Euler (first-order, conditionally stable).
 * Each time step computes: T_next = T + dt * α * ∇²T, where the Laplacian is
 * evaluated using central finite differences with configurable order (2, 4, ..., 20).
 * This is a self-contained implementation that does not inherit from the Simulator
 * base class but demonstrates the same time-integration concepts.
 *
 * @par Lifecycle stage ownership
 * This implementation owns the following lifecycle stages:
 * - Pre-step preparation: performs MPI halo exchange to synchronize ghost cell
 *   values across domain decomposition boundaries
 * - RHS evaluation (compute_rhs): computes the Laplacian using central finite
 *   differences via FDGradient evaluator
 * - Post-step updates: applies explicit Euler update: u += dt * α * ∇²u
 * - Output generation: computes L2 error against analytical solution (no VTK output)
 * - No checkpointing: this is a benchmark/educational solver without restart capability
 *
 * @par Boundary/halo synchronization
 * Boundary conditions and halo exchanges occur at:
 * - Pre-stage: MPI halo exchange via HaloExchange before each RHS evaluation
 * - The solver assumes periodic boundary conditions in all directions
 * - Halo width is automatically configured as fd_order/2 to accommodate the stencil
 * - HaloExchange handles six-face MPI communication for the halo ring
 *
 * @par Application-specific constraints
 * - Stability: the explicit forward Euler scheme requires dt ≤ dx²/(6α) in 3D for
 *   numerical stability (von Neumann analysis). Violating this constraint leads to
 *   exponential growth of numerical errors.
 * - Memory: uses two padded bricks (u and du) with halo regions for computation
 * - Spatial accuracy: finite-difference order is configurable (2, 4, ..., 20) via
 *   command line parameter; higher orders require wider halos and more computation
 * - Parallel: domain decomposition with MPI; must ensure global grid size is
 *   divisible by number of processes
 * - No adaptive time stepping: dt is fixed for the entire simulation
 *
 * @par Contract for substituting alternative integrators
 * To implement a different time-integration scheme in this code structure:
 * - Replace the explicit Euler update "u += cfg.dt * du" with the desired scheme
 * - For explicit multi-stage methods (e.g., Runge-Kutta), implement multiple RHS
 *   evaluations per time step with appropriate stage combinations
 * - For implicit methods, would need to solve linear systems and modify the
 *   halo exchange pattern accordingly
 * - Maintain the same pre-step halo exchange pattern for explicit schemes
 * - Preserve the L2 error calculation for verification purposes
 *
 * @note This is an educational implementation designed for clarity rather than
 *   production use. It demonstrates explicit connection between halo exchange,
 *   gradient evaluation, and time integration. For production thermal simulations,
 *   consider the SpectralHeatPropagator (implicit Euler) for unconditional stability.
 *
 * @see SpectralHeatPropagator for an implicit Euler implementation in Fourier space
 * @see Simulator for the base class contract on time-integration assumptions
 */
void run_fd(const RunConfig &cfg, int rank, int nproc) {
  // 1. Physics. This driver fixes D = 1 (same as `heat3d::kD` elsewhere) so the
  //    Gaussian IC and analytic reference match the other heat3d binaries.

  // 2. Global domain + per-rank decomposition.
  const auto domain = pfc::domain::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  const auto decomp = pfc::decomposition::create(domain, nproc);

  // 3. Storage. Two padded Fields share decomp, rank, and halo width so
  //    nothing downstream can disagree with the layout: `u` is the state,
  //    `du` is the residual we accumulate each step. Halo width = order/2
  //    so the central stencil's most distant neighbour is a halo cell on
  //    either side.
  const int hw = cfg.fd_order / 2;
  pfc::data::Field<double, pfc::HostSpace> u =
      pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  pfc::data::Field<double, pfc::HostSpace> du =
      pfc::data::field_from_subdomain<double>(decomp, rank, hw);

  // 4. Halo exchanger and gradient evaluator, both bound to `u`.
  pfc::comm::HaloExchange<pfc::HostSpace, double> halo(u, decomp, rank,
                                                       MPI_COMM_WORLD);
  pfc::gradient::FDGradient<HeatGrads> grad(u, cfg.fd_order);

  // 5. Initial condition: \f$u(x,y,z,0) = \exp(-|x|^2/(4D))\f$, D = 1.
  u.apply([](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / 4.0);
  });

  // 6. Time loop — explicit Euler, point-wise RHS.
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step) {
    halo.exchange();
    u.for_each_owned([&](int i, int j, int k) {
      const auto g = pfc::gradient::evaluate(grad, pfc::Int3{i, j, k});
      du(i, j, k) = g.xx + g.yy + g.zz;
    });
    u += cfg.dt * du;
    t += cfg.dt;
  }
  const double local_elapsed = MPI_Wtime() - t_start;
  double max_elapsed = 0.0;
  MPI_Allreduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  (void)t; // autonomous heat equation; the running clock is reported by t_final

  // 7. L2-vs-analytic report via shared reporting infrastructure.
  heat3d::RunConfig heat_cfg{cfg.N, cfg.n_steps, cfg.dt, cfg.fd_order};
  heat3d::report(rank, nproc, heat_cfg, "fd",
                 heat3d::fd_extra_metadata(heat_cfg), max_elapsed,
                 "(periodic; interior L2)", [&u, hw](auto &&cb) {
                   const auto sz = u.local_size();
                   for (int k = hw; k < sz[2] - hw; ++k) {
                     for (int j = hw; j < sz[1] - hw; ++j) {
                       for (int i = hw; i < sz[0] - hw; ++i) {
                         const auto p = u.coords(i, j, k);
                         cb(p[0], p[1], p[2], u(i, j, k));
                       }
                     }
                   }
                 });
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
