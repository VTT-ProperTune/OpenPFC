// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d.cpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — single-file app.
 *
 * @details
 * This is the entire `heat3d` application: physics model, command-line
 * parsing, reference solution, and per-method orchestration in one file. All
 * reusable mechanism (point-wise gradient evaluators, the explicit-Euler
 * stepper, the per-cell driver loop, coordinate-space initial-condition
 * helpers, the host CPU-affinity rescue) lives in OpenPFC; this file imports
 * those pieces and wires them to `HeatModel`.
 *
 * Three solvers are exposed via the CLI:
 *  - `fd`          — explicit Euler with even-order central FD Laplacian
 *                    (`pfc::field::FdGradient`) and a separated face-halo
 *                    exchange.
 *  - `spectral`    — implicit Euler in Fourier space (2 FFTs/step), specific
 *                    to the linear heat operator.
 *  - `spectral_pw` — explicit Euler with a point-wise RHS over materialized
 *                    second-derivative fields (`pfc::field::SpectralGradient`,
 *                    1 fwd + 3 inv FFTs/step). Same `HeatModel::rhs` as `fd`.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/separated_halo_exchange.hpp>
#include <openpfc/kernel/fft/fft.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/grad_point.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/runtime/common/cpu_affinity.hpp>

using namespace pfc;

// -----------------------------------------------------------------------------
// Physics: the only thing a physicist edits in this file.
// -----------------------------------------------------------------------------

/**
 * @brief Heat equation \f$\partial_t u = D \nabla^2 u\f$, self-contained.
 *
 * Carries the physical parameters (just `D`), the initial condition as a
 * runtime-swappable spatial lambda, an optional boundary-value provider for
 * future Dirichlet/Neumann support, and the per-point right-hand side as a
 * direct method `rhs(t, g)`. `rhs` is a regular `inline noexcept` method (not
 * `operator()` and not `std::function`) so the inner `for_each_interior` loop
 * inlines it as cleanly as a free function.
 */
struct HeatModel {
  /** Diffusion coefficient. */
  double D = 1.0;

  /** Initial condition \f$u(x,y,z,0)\f$ (default Gaussian). */
  field::PointFn initial_condition = [](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / 4.0);
  };

  /**
   * @brief Optional Dirichlet/Neumann boundary value \f$u_b(x,y,z,t)\f$.
   *
   * Empty by default — the discretization treats the domain as periodic
   * (FD freezes its halo region, spectral assumes periodicity).
   */
  field::PointFnT boundary_value{};

  /** Per-point right-hand side \f$\partial_t u = D\nabla^2 u\f$ (hot path). */
  inline double rhs(double /*t*/, const field::GradPoint &g) const noexcept {
    return D * (g.uxx + g.uyy + g.uzz);
  }
};

// -----------------------------------------------------------------------------
// App scaffolding: CLI, reference solution, per-method orchestration.
// -----------------------------------------------------------------------------

namespace {

enum class Method { Fd, Spectral, SpectralPointwise };

struct RunConfig {
  Method method = Method::Fd;
  int N = 32;
  int n_steps = 100;
  double dt = 0.01;
  double D = 1.0;
  /** Spatial order for FD: even 2, 4, …, 20 (ignored for spectral methods). */
  int fd_order = 2;
};

void print_usage(const char *exe) {
  std::cerr
      << "Usage:\n  " << exe << " fd <N> <n_steps> <dt> <D> <fd_order>\n  " << exe
      << " spectral <N> <n_steps> <dt> <D>\n  " << exe
      << " spectral_pw <N> <n_steps> <dt> <D>\n"
      << "  fd_order: even 2,4,...,20 (central Laplacian; halo width order/2)\n"
      << "  spectral:    implicit Euler in Fourier space (2 FFTs/step)\n"
      << "  spectral_pw: explicit Euler with point-wise RHS over materialized\n"
      << "               second-derivative fields (1 fwd + 3 inv FFTs/step)\n";
}

RunConfig parse_args(int argc, char **argv) {
  RunConfig c;
  if (argc < 2) {
    return c;
  }
  if (std::strcmp(argv[1], "fd") == 0) {
    c.method = Method::Fd;
    if (argc < 7) {
      return c;
    }
    c.fd_order = std::atoi(argv[6]);
  } else if (std::strcmp(argv[1], "spectral") == 0) {
    c.method = Method::Spectral;
    if (argc < 6) {
      return c;
    }
  } else if (std::strcmp(argv[1], "spectral_pw") == 0) {
    c.method = Method::SpectralPointwise;
    if (argc < 6) {
      return c;
    }
  } else {
    return c;
  }
  c.N = std::atoi(argv[2]);
  c.n_steps = std::atoi(argv[3]);
  c.dt = std::atof(argv[4]);
  c.D = std::atof(argv[5]);
  return c;
}

bool validate(const RunConfig &c) {
  if (c.N < 8 || c.n_steps < 1 || c.dt <= 0.0 || c.D <= 0.0) {
    return false;
  }
  if (c.method == Method::Fd) {
    if (c.fd_order < 2 || c.fd_order > 20 || (c.fd_order % 2) != 0) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Fundamental Gaussian solution on \f$\mathbb{R}^3\f$ for the heat
 *        equation with IC \f$u(x,0)=\exp(-|x|^2/(4D))\f$:
 *        \f$u(x,t)=(1+t)^{-3/2}\exp(-|x|^2/(4D(1+t)))\f$.
 */
double analytic_gaussian(double r2, double t, double D) {
  const double s = 1.0 + t;
  return std::pow(s, -1.5) * std::exp(-r2 / (4.0 * D * s));
}

void run_fd(const RunConfig &cfg, int rank, int nproc) {
  auto world = world::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, nproc);
  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

  const int hw = cfg.fd_order / 2;
  if (2 * hw >= nx || 2 * hw >= ny || 2 * hw >= nz) {
    if (rank == 0) {
      std::cerr << "heat3d: local subdomain " << nx << "x" << ny << "x" << nz
                << " too small for fd_order=" << cfg.fd_order << " (need > "
                << (2 * hw) << " points per dimension)\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const double dx = 1.0;
  const double inv_dx2 = 1.0 / (dx * dx);

  HeatModel model;
  model.D = cfg.D;
  model.initial_condition = [D = cfg.D](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / (4.0 * D));
  };

  std::vector<double> u;
  field::apply_subdomain(u, decomp, rank, model.initial_condition);

  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, hw);
  SeparatedFaceHaloExchanger<double> exchanger(decomp, rank, hw, MPI_COMM_WORLD);

  field::FdGradient grad(u.data(), nx, ny, nz, inv_dx2, inv_dx2, inv_dx2, hw,
                         cfg.fd_order);
  sim::steppers::EulerStepper stepper(grad, model, cfg.dt, nlocal);

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step) {
    exchanger.exchange_halos(u.data(), u.size(), face_halos);
    t = stepper.step(t, u);
  }
  const double t1 = MPI_Wtime();
  double elapsed = t1 - t0;
  double max_elapsed = 0.0;
  MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  double sum_err2 = 0.0;
  const double t_final = static_cast<double>(cfg.n_steps) * cfg.dt;
  field::for_each_interior_with_coords(
      u, decomp, rank, hw, [&](const Real3 &x, double u_val) {
        const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        const double uex = analytic_gaussian(r2, t_final, cfg.D);
        const double e = u_val - uex;
        sum_err2 += e * e;
      });
  double g_err2 = 0.0;
  MPI_Reduce(&sum_err2, &g_err2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
#if defined(_OPENMP)
    const int omp_nt = omp_get_max_threads();
#else
    const int omp_nt = 1;
#endif
    std::cout << "heat3d method=fd N=" << cfg.N << " n_steps=" << cfg.n_steps
              << " dt=" << cfg.dt << " D=" << cfg.D << " fd_order=" << cfg.fd_order
              << " mpi_ranks=" << nproc << " omp_max_threads=" << omp_nt;
#if defined(_OPENMP)
    std::cout << " omp_get_num_procs()=" << omp_get_num_procs();
#endif
    std::cout << "\n";
    std::cout << "timing_s=" << max_elapsed << " avg_step_time_s="
              << (max_elapsed / static_cast<double>(cfg.n_steps))
              << " (MPI_MAX across ranks)\n";
    const double rms =
        std::sqrt(g_err2 / static_cast<double>(cfg.N * cfg.N * cfg.N));
    std::cout
        << "l2_error_vs_R3_analytic_rms=" << rms
        << " (periodic domain; error dominated by boundaries for localized IC)\n";
  }
}

void run_spectral(const RunConfig &cfg, int rank, int nproc) {
  auto world = world::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, nproc);
  fft::CpuFft fft = fft::create(decomp, MPI_COMM_WORLD);

  std::vector<double> psi(fft.size_inbox());
  std::vector<std::complex<double>> psi_F(fft.size_outbox());
  std::vector<double> opL(fft.size_outbox());

  const auto &gw = decomposition::get_world(decomp);
  auto size = world::get_size(gw);
  auto origin = world::get_origin(gw);
  auto spacing = world::get_spacing(gw);

  auto ib = fft.get_inbox_bounds();
  int idx = 0;
  for (int k = ib.low[2]; k <= ib.high[2]; ++k) {
    for (int j = ib.low[1]; j <= ib.high[1]; ++j) {
      for (int i = ib.low[0]; i <= ib.high[0]; ++i) {
        const double x = origin[0] + static_cast<double>(i) * spacing[0];
        const double y = origin[1] + static_cast<double>(j) * spacing[1];
        const double z = origin[2] + static_cast<double>(k) * spacing[2];
        psi[static_cast<size_t>(idx)] =
            std::exp(-(x * x + y * y + z * z) / (4.0 * cfg.D));
        ++idx;
      }
    }
  }

  auto ob = fft.get_outbox_bounds();
  const double fx =
      2.0 * constants::pi / (spacing[0] * static_cast<double>(size[0]));
  const double fy =
      2.0 * constants::pi / (spacing[1] * static_cast<double>(size[1]));
  const double fz =
      2.0 * constants::pi / (spacing[2] * static_cast<double>(size[2]));
  idx = 0;
  for (int k = ob.low[2]; k <= ob.high[2]; ++k) {
    for (int j = ob.low[1]; j <= ob.high[1]; ++j) {
      for (int i = ob.low[0]; i <= ob.high[0]; ++i) {
        const double ki = (i <= size[0] / 2) ? static_cast<double>(i) * fx
                                             : static_cast<double>(i - size[0]) * fx;
        const double kj = (j <= size[1] / 2) ? static_cast<double>(j) * fy
                                             : static_cast<double>(j - size[1]) * fy;
        const double kk = (k <= size[2] / 2) ? static_cast<double>(k) * fz
                                             : static_cast<double>(k - size[2]) * fz;
        const double k_lap = -(ki * ki + kj * kj + kk * kk);
        opL[static_cast<size_t>(idx)] = 1.0 / (1.0 - cfg.dt * cfg.D * k_lap);
        ++idx;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  for (int step = 0; step < cfg.n_steps; ++step) {
    fft.forward(psi, psi_F);
    for (size_t k = 0; k < psi_F.size(); ++k) {
      psi_F[k] *= opL[static_cast<size_t>(k)];
    }
    fft.backward(psi_F, psi);
  }
  const double t1 = MPI_Wtime();
  double elapsed = t1 - t0;
  double max_elapsed = 0.0;
  MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  const double t_final = static_cast<double>(cfg.n_steps) * cfg.dt;
  double sum_err2 = 0.0;
  idx = 0;
  for (int k = ib.low[2]; k <= ib.high[2]; ++k) {
    for (int j = ib.low[1]; j <= ib.high[1]; ++j) {
      for (int i = ib.low[0]; i <= ib.high[0]; ++i) {
        const double x = origin[0] + static_cast<double>(i) * spacing[0];
        const double y = origin[1] + static_cast<double>(j) * spacing[1];
        const double z = origin[2] + static_cast<double>(k) * spacing[2];
        const double r2 = x * x + y * y + z * z;
        const double uex = analytic_gaussian(r2, t_final, cfg.D);
        const double e = psi[static_cast<size_t>(idx)] - uex;
        sum_err2 += e * e;
        ++idx;
      }
    }
  }
  double g_err2 = 0.0;
  MPI_Reduce(&sum_err2, &g_err2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const double rms =
        std::sqrt(g_err2 / static_cast<double>(cfg.N * cfg.N * cfg.N));
    std::cout << "heat3d method=spectral N=" << cfg.N << " n_steps=" << cfg.n_steps
              << " dt=" << cfg.dt << " D=" << cfg.D << " mpi_ranks=" << nproc
              << "\n";
    std::cout << "timing_s=" << max_elapsed << " avg_step_time_s="
              << (max_elapsed / static_cast<double>(cfg.n_steps))
              << " (MPI_MAX across ranks)\n";
    std::cout << "l2_error_vs_R3_analytic_rms=" << rms
              << " (periodic spectral vs infinite-domain reference)\n";
  }
}

void run_spectral_pointwise(const RunConfig &cfg, int rank, int nproc) {
  auto world = world::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, nproc);
  fft::CpuFft fft = fft::create(decomp, MPI_COMM_WORLD);

  std::vector<double> u(fft.size_inbox());

  const auto &gw = decomposition::get_world(decomp);
  auto size = world::get_size(gw);
  auto origin = world::get_origin(gw);
  auto spacing = world::get_spacing(gw);
  auto ib = fft.get_inbox_bounds();

  HeatModel model;
  model.D = cfg.D;
  model.initial_condition = [D = cfg.D](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / (4.0 * D));
  };
  field::apply(u, gw, fft, [&ic = model.initial_condition](const Real3 &r) {
    return ic(r[0], r[1], r[2]);
  });

  field::SpectralGradient grad(fft, u, size, spacing, ib, fft.get_outbox_bounds());
  sim::steppers::EulerStepper stepper(grad, model, cfg.dt, u.size());

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step) {
    t = stepper.step(t, u);
  }
  const double t1 = MPI_Wtime();
  double elapsed = t1 - t0;
  double max_elapsed = 0.0;
  MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  const double t_final = static_cast<double>(cfg.n_steps) * cfg.dt;
  double sum_err2 = 0.0;
  std::size_t err_idx = 0;
  for (int k = ib.low[2]; k <= ib.high[2]; ++k) {
    for (int j = ib.low[1]; j <= ib.high[1]; ++j) {
      for (int i = ib.low[0]; i <= ib.high[0]; ++i) {
        const double x = origin[0] + static_cast<double>(i) * spacing[0];
        const double y = origin[1] + static_cast<double>(j) * spacing[1];
        const double z = origin[2] + static_cast<double>(k) * spacing[2];
        const double r2 = x * x + y * y + z * z;
        const double uex = analytic_gaussian(r2, t_final, cfg.D);
        const double e = u[err_idx] - uex;
        sum_err2 += e * e;
        ++err_idx;
      }
    }
  }
  double g_err2 = 0.0;
  MPI_Reduce(&sum_err2, &g_err2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const double rms =
        std::sqrt(g_err2 / static_cast<double>(cfg.N * cfg.N * cfg.N));
    std::cout << "heat3d method=spectral_pw N=" << cfg.N
              << " n_steps=" << cfg.n_steps << " dt=" << cfg.dt << " D=" << cfg.D
              << " mpi_ranks=" << nproc << "\n";
    std::cout << "timing_s=" << max_elapsed << " avg_step_time_s="
              << (max_elapsed / static_cast<double>(cfg.n_steps))
              << " (MPI_MAX across ranks)\n";
    std::cout << "l2_error_vs_R3_analytic_rms=" << rms
              << " (point-wise spectral RHS, explicit Euler)\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  pfc::runtime::reset_cpu_affinity_if_single_mpi_rank(nproc);

  const RunConfig cfg = parse_args(argc, argv);
  const bool args_ok = (cfg.method == Method::Fd && argc >= 7) ||
                       (cfg.method == Method::Spectral && argc >= 6) ||
                       (cfg.method == Method::SpectralPointwise && argc >= 6);
  if (!args_ok || !validate(cfg)) {
    if (rank == 0) {
      print_usage(argv[0]);
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  if (cfg.method == Method::Fd) {
    run_fd(cfg, rank, nproc);
  } else if (cfg.method == Method::Spectral) {
    run_spectral(cfg, rank, nproc);
  } else {
    run_spectral_pointwise(cfg, rank, nproc);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
