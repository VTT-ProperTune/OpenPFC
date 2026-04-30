// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <heat3d/common.hpp>
#include <heat3d/discretization.hpp>
#include <heat3d/heat_model.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/separated_halo_exchange.hpp>
#include <openpfc/kernel/fft/fft.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>

using namespace pfc;

namespace {

/**
 * Open MPI (and some other launchers) pin a single rank to one logical CPU, so
 * OpenMP sees `omp_get_num_procs()==1` and all threads share one core. For
 * **exactly one MPI rank** on Linux, reset the process affinity mask to all
 * online CPUs so OpenMP can scale. Multi-rank jobs are unchanged (each rank
 * keeps the launcher mask). Opt out with `HEAT3D_NO_RESET_AFFINITY` set.
 */
void heat3d_reset_cpu_affinity_if_single_mpi_rank(int nproc) {
#if defined(__linux__)
  if (nproc != 1) {
    return;
  }
  if (std::getenv("HEAT3D_NO_RESET_AFFINITY") != nullptr) {
    return;
  }
  const long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpus <= 1) {
    return;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  for (long i = 0; i < ncpus; ++i) {
    CPU_SET(static_cast<int>(i), &set);
  }
  (void)sched_setaffinity(0, sizeof(set), &set);
#endif
}

} // namespace

static void run_fd(const heat3d::RunConfig &cfg, int rank, int nproc) {
  auto world = world::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, nproc);
  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  auto local_lower = world::get_lower(local_world);
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

  std::vector<double> u(nlocal);
  std::vector<double> du(nlocal, 0.0);
  heat3d::fill_gaussian_subdomain(&u, decomp, rank, cfg.D);

  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, hw);
  SeparatedFaceHaloExchanger<double> exchanger(decomp, rank, hw, MPI_COMM_WORLD);

  heat3d::FdGradient grad(u.data(), nx, ny, nz, inv_dx2, inv_dx2, inv_dx2, hw,
                          cfg.fd_order);
  heat3d::HeatRhs rhs{heat3d::HeatModel{cfg.D}};

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  const ptrdiff_t nptr = static_cast<ptrdiff_t>(nlocal);
  const int imin = hw;
  const int imax = nx - hw;
  const int jmin = hw;
  const int jmax = ny - hw;
  const int kmin = hw;
  const int kmax = nz - hw;
  const int sxy = nx * ny;
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step) {
    exchanger.exchange_halos(u.data(), u.size(), face_halos);
    heat3d::for_each_interior(grad, du.data(), t, rhs);
    for (ptrdiff_t li = 0; li < nptr; ++li) {
      u[static_cast<size_t>(li)] += cfg.dt * du[static_cast<size_t>(li)];
    }
    t += cfg.dt;
  }
  const double t1 = MPI_Wtime();
  double elapsed = t1 - t0;
  double max_elapsed = 0.0;
  MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  double sum_err2 = 0.0;
  const double t_final = static_cast<double>(cfg.n_steps) * cfg.dt;
  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      for (int ix = imin; ix < imax; ++ix) {
        const int gi = local_lower[0] + ix;
        const int gj = local_lower[1] + iy;
        const int gk = local_lower[2] + iz;
        const double x = static_cast<double>(gi);
        const double y = static_cast<double>(gj);
        const double z = static_cast<double>(gk);
        const double r2 = x * x + y * y + z * z;
        const size_t c = static_cast<size_t>(ix) +
                         static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                         static_cast<size_t>(iz) * static_cast<size_t>(sxy);
        const double uex = heat3d::analytic_gaussian(r2, t_final, cfg.D);
        const double e = u[c] - uex;
        sum_err2 += e * e;
      }
    }
  }
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

static void run_spectral(const heat3d::RunConfig &cfg, int rank, int nproc) {
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
        const double uex = heat3d::analytic_gaussian(r2, t_final, cfg.D);
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

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  heat3d_reset_cpu_affinity_if_single_mpi_rank(nproc);

  const heat3d::RunConfig cfg = heat3d::parse_args(argc, argv);
  const bool args_ok = (cfg.method == heat3d::Method::Fd && argc >= 7) ||
                       (cfg.method == heat3d::Method::Spectral && argc >= 6);
  if (!args_ok || !heat3d::validate(cfg)) {
    if (rank == 0) {
      heat3d::print_usage(argv[0]);
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  if (cfg.method == heat3d::Method::Fd) {
    run_fd(cfg, rank, nproc);
  } else {
    run_spectral(cfg, rank, nproc);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
