// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d.cpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — application driver.
 *
 * @details
 * Application driver for the `heat3d` example: command-line parsing,
 * reference solution, and per-method orchestration. The physics model
 * itself lives in **`heat3d/heat_model.hpp`** — the single header a
 * physicist edits — and is unit-tested in isolation under
 * `apps/heat3d/tests/test_heat3d.cpp`. All reusable mechanism (point-wise
 * gradient evaluators, the explicit-Euler stepper, the per-cell driver
 * loop, coordinate-space initial-condition helpers, the host CPU-affinity
 * rescue) lives in OpenPFC; this file imports those pieces and wires
 * them to `HeatModel`.
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
#include <exception>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <sstream>
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
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/runtime/common/cpu_affinity.hpp>
#include <openpfc/runtime/common/mpi_timer.hpp>

#include <heat3d/heat_model.hpp>

using namespace pfc;
using heat3d::HeatModel;

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

/**
 * @brief Run-end RMS-against-analytic reduction + rank-0 reporting, shared
 *        by all three solvers.
 *
 * Iterates rank-local owned cells (the visitor decides which set: FD
 * `for_each_interior`, point-wise spectral `for_each_owned`, or a manual
 * loop over an FFT inbox), accumulates `(u - u_exact)^2`, MPI-reduces to
 * rank 0, and prints the canonical three-line `method / timing / l2`
 * summary.
 *
 * @param method_tag       Value after `method=` in the metadata line.
 * @param extra_metadata   Appended after `mpi_ranks=...` (e.g. `fd_order=`,
 *                         OpenMP info); pass an empty string for none.
 * @param max_elapsed      Wall-clock max of the time-stepping loop across
 *                         ranks (already reduced by the caller).
 * @param l2_note          Parenthetical context appended to the L2 line.
 * @param visit_owned_cells   Callable `void(cb)` that calls
 *                            `cb(double x, double y, double z, double u_val)`
 *                            once per rank-local owned cell.
 */
template <class Visitor>
void report_l2_and_timing(int rank, int nproc, const RunConfig &cfg,
                          const char *method_tag, const std::string &extra_metadata,
                          double max_elapsed, const char *l2_note,
                          Visitor &&visit_owned_cells) {
  double sum_err2 = 0.0;
  const double t_final = static_cast<double>(cfg.n_steps) * cfg.dt;
  visit_owned_cells([&](double x, double y, double z, double u_val) {
    const double r2 = x * x + y * y + z * z;
    const double uex = analytic_gaussian(r2, t_final, cfg.D);
    const double e = u_val - uex;
    sum_err2 += e * e;
  });
  double g_err2 = 0.0;
  MPI_Reduce(&sum_err2, &g_err2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "heat3d method=" << method_tag << " N=" << cfg.N
              << " n_steps=" << cfg.n_steps << " dt=" << cfg.dt << " D=" << cfg.D
              << " mpi_ranks=" << nproc;
    if (!extra_metadata.empty()) {
      std::cout << " " << extra_metadata;
    }
    std::cout << "\n";
    std::cout << "timing_s=" << max_elapsed << " avg_step_time_s="
              << (max_elapsed / static_cast<double>(cfg.n_steps))
              << " (MPI_MAX across ranks)\n";
    const double rms =
        std::sqrt(g_err2 / static_cast<double>(cfg.N * cfg.N * cfg.N));
    std::cout << "l2_error_vs_R3_analytic_rms=" << rms << " " << l2_note << "\n";
  }
}

/**
 * @brief Build the FD-specific extra metadata string: `fd_order=...` plus
 *        OpenMP thread info (when `_OPENMP` is defined).
 */
std::string fd_extra_metadata(const RunConfig &cfg) {
  std::ostringstream os;
  os << "fd_order=" << cfg.fd_order;
#if defined(_OPENMP)
  os << " omp_max_threads=" << omp_get_max_threads()
     << " omp_get_num_procs()=" << omp_get_num_procs();
#else
  os << " omp_max_threads=1";
#endif
  return os.str();
}

void run_fd(const RunConfig &cfg, int rank, int nproc) {
  auto world = world::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, nproc);
  const int hw = cfg.fd_order / 2;
  field::LocalField<double> u =
      field::LocalField<double>::from_subdomain(decomp, rank, hw);

  HeatModel model;
  model.D = cfg.D;
  u.apply(model.initial_condition);

  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, hw);
  SeparatedFaceHaloExchanger<double> exchanger(decomp, rank, hw, MPI_COMM_WORLD);

  auto grad = field::create(u, cfg.fd_order);
  auto stepper = sim::steppers::create(u, grad, model, cfg.dt);

  runtime::MpiTimer timer{MPI_COMM_WORLD};
  runtime::tic(timer);
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step) {
    exchanger.exchange_halos(u.data(), u.size(), face_halos);
    t = stepper.step(t, u.vec());
  }
  const double max_elapsed = runtime::toc(timer);

  report_l2_and_timing(
      rank, nproc, cfg, "fd", fd_extra_metadata(cfg), max_elapsed,
      "(periodic domain; error dominated by boundaries for localized IC)",
      [&u](auto &&cb) { u.for_each_interior(cb); });
}

void run_spectral(const RunConfig &cfg, int rank, int nproc) {
  auto world = world::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, nproc);
  fft::CpuFft fft = fft::create(decomp, MPI_COMM_WORLD);

  const auto &gw = decomposition::get_world(decomp);
  field::LocalField<double> u =
      field::LocalField<double>::from_inbox(gw, fft.get_inbox_bounds());

  HeatModel model;
  model.D = cfg.D;
  u.apply(model.initial_condition);

  // Implicit-Euler symbol in Fourier space: 1 / (1 - dt * D * k_lap).
  std::vector<std::complex<double>> psi_F(fft.size_outbox());
  std::vector<double> opL(fft.size_outbox());
  const auto size = u.global_size();
  const auto spacing = u.spacing();
  const auto ob = fft.get_outbox_bounds();
  const double fx =
      2.0 * constants::pi / (spacing[0] * static_cast<double>(size[0]));
  const double fy =
      2.0 * constants::pi / (spacing[1] * static_cast<double>(size[1]));
  const double fz =
      2.0 * constants::pi / (spacing[2] * static_cast<double>(size[2]));
  std::size_t idx = 0;
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
        opL[idx++] = 1.0 / (1.0 - cfg.dt * cfg.D * k_lap);
      }
    }
  }

  runtime::MpiTimer timer{MPI_COMM_WORLD};
  runtime::tic(timer);
  for (int step = 0; step < cfg.n_steps; ++step) {
    fft.forward(u.vec(), psi_F);
    for (std::size_t k = 0; k < psi_F.size(); ++k) {
      psi_F[k] *= opL[k];
    }
    fft.backward(psi_F, u.vec());
  }
  const double max_elapsed = runtime::toc(timer);

  report_l2_and_timing(rank, nproc, cfg, "spectral", "", max_elapsed,
                       "(periodic spectral vs infinite-domain reference)",
                       [&u](auto &&cb) { u.for_each_owned(cb); });
}

void run_spectral_pointwise(const RunConfig &cfg, int rank, int nproc) {
  auto world = world::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, nproc);
  fft::CpuFft fft = fft::create(decomp, MPI_COMM_WORLD);

  const auto &gw = decomposition::get_world(decomp);
  field::LocalField<double> u =
      field::LocalField<double>::from_inbox(gw, fft.get_inbox_bounds());

  HeatModel model;
  model.D = cfg.D;
  u.apply(model.initial_condition);

  auto grad = field::create(u, fft);
  auto stepper = sim::steppers::create(u, grad, model, cfg.dt);

  runtime::MpiTimer timer{MPI_COMM_WORLD};
  runtime::tic(timer);
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step) {
    t = stepper.step(t, u.vec());
  }
  const double max_elapsed = runtime::toc(timer);

  report_l2_and_timing(rank, nproc, cfg, "spectral_pw", "", max_elapsed,
                       "(point-wise spectral RHS, explicit Euler)",
                       [&u](auto &&cb) { u.for_each_owned(cb); });
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

  try {
    if (cfg.method == Method::Fd) {
      run_fd(cfg, rank, nproc);
    } else if (cfg.method == Method::Spectral) {
      run_spectral(cfg, rank, nproc);
    } else {
      run_spectral_pointwise(cfg, rank, nproc);
    }
  } catch (const std::exception &e) {
    std::cerr << "heat3d (rank " << rank << "): " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
