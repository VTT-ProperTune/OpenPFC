// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kobayashi_fd_manual.cpp
 * @brief Kobayashi phase-field + temperature coupling — manual FD matching the Julia
 *        `kobayashi_v1` script (Biner-style discretisation, explicit Euler).
 *
 * Periodic boundaries in x and y via `PaddedHaloExchanger` on an nz=1 slab.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <numbers>
#include <string>
#include <vector>

#include <kobayashi/cli.hpp>
#include <kobayashi/defaults.hpp>

#include <openpfc/frontend/io/png_writer.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

namespace {

using pfc::field::PaddedBrick;
using pfc::field::for_each_owned;

void pack_owned_xy0(const PaddedBrick<double> &b, std::vector<double> &out) {
  const int nx = b.nx();
  const int ny = b.ny();
  out.resize(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      out[static_cast<std::size_t>(i) +
          static_cast<std::size_t>(j) * static_cast<std::size_t>(nx)] = b(i, j, 0);
    }
  }
}

void write_phi_png(int rank, const pfc::decomposition::Decomposition &decomp,
                   const PaddedBrick<double> &phi, const std::string &path) {
  std::vector<double> local;
  pack_owned_xy0(phi, local);
  pfc::io::write_mpi_scalar_field_png_xy(MPI_COMM_WORLD, decomp, rank, local, path,
                                         0.0, 1.0);
}

/**
 * @brief Gather one z-layer of owned cell values onto rank 0 in **global**
 *        row-major order `global[gx + gy * nx_glob]` (same layout as PNG gather).
 */
void gather_global_xy_rank0(const pfc::decomposition::Decomposition &decomp, int rank,
                            int nproc, MPI_Comm comm,
                            const std::vector<double> &local_owned_xy, int nx_glob,
                            int ny_glob, std::vector<double> &global_out) {
  const int my_count = static_cast<int>(local_owned_xy.size());
  std::vector<int> counts(static_cast<std::size_t>(nproc));
  MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

  std::vector<int> displs(static_cast<std::size_t>(nproc));
  int total = 0;
  for (int r = 0; r < nproc; ++r) {
    displs[static_cast<std::size_t>(r)] = total;
    total += counts[static_cast<std::size_t>(r)];
  }

  std::vector<double> gathered;
  if (rank == 0) {
    gathered.resize(static_cast<std::size_t>(total));
  }

  MPI_Gatherv(const_cast<double *>(local_owned_xy.data()), my_count, MPI_DOUBLE,
              rank == 0 ? gathered.data() : nullptr, counts.data(), displs.data(),
              MPI_DOUBLE, 0, comm);

  if (rank != 0) {
    return;
  }

  global_out.assign(static_cast<std::size_t>(nx_glob) * static_cast<std::size_t>(ny_glob),
                    std::numeric_limits<double>::quiet_NaN());

  std::size_t offset = 0;
  for (int r = 0; r < nproc; ++r) {
    const auto &sw = pfc::decomposition::get_subworld(decomp, r);
    auto lo = pfc::world::get_lower(sw);
    auto sz = pfc::world::get_size(sw);
    const int nx = sz[0];
    const int ny = sz[1];
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const std::size_t li =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx);
        const int gx = lo[0] + ix;
        const int gy = lo[1] + iy;
        global_out[static_cast<std::size_t>(gx) +
                     static_cast<std::size_t>(gy) * static_cast<std::size_t>(nx_glob)] =
            gathered[offset + li];
      }
    }
    offset += static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  }
}

struct FieldStats {
  double sum = 0.0;
  double sumsq = 0.0;
  double min_v = 0.0;
  double max_v = 0.0;
};

/** Lexicographic \((g_x, g_y)\) accumulation — deterministic given identical data. */
FieldStats stats_global_ordered(const std::vector<double> &global_xy, int nx_glob,
                                int ny_glob) {
  FieldStats s{};
  s.min_v = std::numeric_limits<double>::infinity();
  s.max_v = -std::numeric_limits<double>::infinity();
  for (int gy = 0; gy < ny_glob; ++gy) {
    for (int gx = 0; gx < nx_glob; ++gx) {
      const double v =
          global_xy[static_cast<std::size_t>(gx) +
                    static_cast<std::size_t>(gy) * static_cast<std::size_t>(nx_glob)];
      s.sum += v;
      s.sumsq += v * v;
      s.min_v = std::min(s.min_v, v);
      s.max_v = std::max(s.max_v, v);
    }
  }
  return s;
}

void run_kobayashi(const kobayashi::RunConfig &cfg, int rank, int nproc) {
  const double dx = cfg.dx;
  const double dy = dx;
  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;
  const double inv_lap_den = 1.0 / (dx * dy);

  const auto world = pfc::world::create(pfc::GridSize({cfg.Nx, cfg.Ny, 1}),
                                        pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                        pfc::GridSpacing({dx, dy, 1.0}));
  const auto decomp = pfc::decomposition::create(world, nproc);

  constexpr int hw = 1;
  PaddedBrick<double> phi(decomp, rank, hw);
  PaddedBrick<double> tempr(decomp, rank, hw);
  PaddedBrick<double> lap_phi(decomp, rank, hw);
  PaddedBrick<double> lap_t(decomp, rank, hw);
  PaddedBrick<double> phidx(decomp, rank, hw);
  PaddedBrick<double> phidy(decomp, rank, hw);
  PaddedBrick<double> epsilon(decomp, rank, hw);
  PaddedBrick<double> epsilon_deriv(decomp, rank, hw);

  const int Nx = cfg.Nx;
  const int Ny = cfg.Ny;
  const int ci = Nx / 2;
  const int cj = Ny / 2;

  for_each_owned(phi, [&](int i, int j, int k) {
    (void)k;
    const auto g = phi.global(i, j, 0);
    const int gi = g[0];
    const int gj = g[1];
    const double ddx = static_cast<double>(gi - ci);
    const double ddy = static_cast<double>(gj - cj);
    phi(i, j, 0) = (ddx * ddx + ddy * ddy < kobayashi::kSeed) ? 1.0 : 0.0;
  });
  for_each_owned(tempr, [&](int i, int j, int k) { tempr(i, j, k) = 0.0; });

  pfc::PaddedHaloExchanger<double> halo_phi(decomp, rank, hw, MPI_COMM_WORLD, 0);
  pfc::PaddedHaloExchanger<double> halo_t(decomp, rank, hw, MPI_COMM_WORLD, 20);
  pfc::PaddedHaloExchanger<double> halo_eps(decomp, rank, hw, MPI_COMM_WORLD, 40);
  pfc::PaddedHaloExchanger<double> halo_epsd(decomp, rank, hw, MPI_COMM_WORLD, 60);
  pfc::PaddedHaloExchanger<double> halo_phidx(decomp, rank, hw, MPI_COMM_WORLD, 80);
  pfc::PaddedHaloExchanger<double> halo_phidy(decomp, rank, hw, MPI_COMM_WORLD, 100);

  const bool skip_png = std::getenv("OPENPFC_KOBAYASHI_SKIP_PNG") != nullptr;
  const bool quiet = std::getenv("OPENPFC_KOBAYASHI_QUIET") != nullptr;
  const int nprint_eff = quiet ? 0 : cfg.nprint;

  if (rank == 0) {
    std::filesystem::create_directories(cfg.output_dir);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "KOBAYASHI_MPI_COMM_WORLD_SIZE=" << nproc
              << " (must match srun/mpirun task count; if not, tasks are not sharing one "
                 "MPI_COMM_WORLD)\n";
  }

  int filenum = 0;
  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(), filenum);
    if (rank == 0) {
      std::cout << "saving step 0/" << cfg.n_steps << " to file " << path << "\n";
    }
    write_phi_png(rank, decomp, phi, path);
    ++filenum;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_loop0 = MPI_Wtime();

  for (int istep = 1; istep <= cfg.n_steps; ++istep) {
    halo_phi.exchange_halos(phi.data(), phi.size());
    halo_t.exchange_halos(tempr.data(), tempr.size());

    for_each_owned(phi, [&](int i, int j, int k) {
      const double hne = phi(i + 1, j, k);
      const double hnw = phi(i - 1, j, k);
      const double hns = phi(i, j - 1, k);
      const double hnn = phi(i, j + 1, k);
      const double hnc = phi(i, j, k);
      lap_phi(i, j, k) = (hne + hnw + hns + hnn - 4.0 * hnc) * inv_lap_den;

      const double Tne = tempr(i + 1, j, k);
      const double Tnw = tempr(i - 1, j, k);
      const double Tns = tempr(i, j - 1, k);
      const double Tnn = tempr(i, j + 1, k);
      const double Tnc = tempr(i, j, k);
      lap_t(i, j, k) = (Tne + Tnw + Tns + Tnn - 4.0 * Tnc) * inv_lap_den;

      const double dpx = (phi(i + 1, j, k) - phi(i - 1, j, k)) * inv_dx;
      const double dpy = (phi(i, j + 1, k) - phi(i, j - 1, k)) * inv_dy;
      phidx(i, j, k) = dpx;
      phidy(i, j, k) = dpy;

      const double theta = std::atan2(dpy, dpx);
      epsilon(i, j, k) =
          kobayashi::kEpsilonb *
          (1.0 + kobayashi::kDelta * std::cos(kobayashi::kAniso * (theta - kobayashi::kTheta0)));
      epsilon_deriv(i, j, k) = -kobayashi::kEpsilonb * kobayashi::kAniso *
                               kobayashi::kDelta *
                               std::sin(kobayashi::kAniso * (theta - kobayashi::kTheta0));
    });

    halo_eps.exchange_halos(epsilon.data(), epsilon.size());
    halo_epsd.exchange_halos(epsilon_deriv.data(), epsilon_deriv.size());
    halo_phidx.exchange_halos(phidx.data(), phidx.size());
    halo_phidy.exchange_halos(phidy.data(), phidy.size());

    for_each_owned(phi, [&](int i, int j, int k) {
      const double phiold = phi(i, j, k);

      const double term1 = (epsilon(i, j + 1, k) * epsilon_deriv(i, j + 1, k) *
                                  phidx(i, j + 1, k) -
                              epsilon(i, j - 1, k) * epsilon_deriv(i, j - 1, k) *
                                  phidx(i, j - 1, k)) *
                             inv_dy;

      const double term2 = -(epsilon(i + 1, j, k) * epsilon_deriv(i + 1, j, k) *
                                phidy(i + 1, j, k) -
                            epsilon(i - 1, j, k) * epsilon_deriv(i - 1, j, k) *
                                phidy(i - 1, j, k)) *
                           inv_dx;

      const double ep = epsilon(i, j, k);
      const double term3 = ep * ep * lap_phi(i, j, k);

      const double m =
          kobayashi::kAlpha / std::numbers::pi *
          std::atan(kobayashi::kGamma * (kobayashi::kTeq - tempr(i, j, k)));
      const double term4 = phiold * (1.0 - phiold) * (phiold - 0.5 + m);

      phi(i, j, k) = phiold + (cfg.dt / kobayashi::kTau) * (term1 + term2 + term3 + term4);

      tempr(i, j, k) =
          tempr(i, j, k) + cfg.dt * lap_t(i, j, k) + kobayashi::kKappa * (phi(i, j, k) - phiold);
    });

    if (nprint_eff > 0 && istep % nprint_eff == 0 && rank == 0) {
      std::cout << "step " << istep << "/" << cfg.n_steps << " done\n";
    }

    if (!skip_png && cfg.nsave > 0 && istep % cfg.nsave == 0) {
      char path[4096];
      std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(),
                    filenum);
      if (rank == 0) {
        std::cout << "saving step " << istep << "/" << cfg.n_steps << " to file " << path
                  << "\n";
      }
      write_phi_png(rank, decomp, phi, path);
      ++filenum;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_loop1 = MPI_Wtime();
  const double wall_local = t_loop1 - t_loop0;
  double wall_max = 0.0;
  MPI_Reduce(&wall_local, &wall_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_final.png", cfg.output_dir.c_str());
    if (rank == 0) {
      std::cout << "saving final field to " << path << "\n";
    }
    write_phi_png(rank, decomp, phi, path);
  }

  std::vector<double> loc_phi;
  std::vector<double> loc_T;
  pack_owned_xy0(phi, loc_phi);
  pack_owned_xy0(tempr, loc_T);

  std::vector<double> g_phi;
  std::vector<double> g_T;
  gather_global_xy_rank0(decomp, rank, nproc, MPI_COMM_WORLD, loc_phi, Nx, Ny, g_phi);
  gather_global_xy_rank0(decomp, rank, nproc, MPI_COMM_WORLD, loc_T, Nx, Ny, g_T);

  if (rank == 0) {
    const FieldStats sp = stats_global_ordered(g_phi, Nx, Ny);
    const FieldStats sT = stats_global_ordered(g_T, Nx, Ny);
    const double l2_phi = std::sqrt(sp.sumsq);
    const double l2_T = std::sqrt(sT.sumsq);
    std::cout << std::setprecision(17);
    std::cout << "KOBAYASHI_VERIFY"
              << " wall_loop_max_s=" << wall_max << " nproc=" << nproc << " Nx=" << Nx
              << " Ny=" << Ny << " steps=" << cfg.n_steps << " dt=" << cfg.dt
              << " dx=" << cfg.dx << " sum_phi=" << sp.sum << " sumsq_phi=" << sp.sumsq
              << " l2_phi=" << l2_phi << " min_phi=" << sp.min_v << " max_phi=" << sp.max_v
              << " sum_T=" << sT.sum << " sumsq_T=" << sT.sumsq << " l2_T=" << l2_T
              << " min_T=" << sT.min_v << " max_T=" << sT.max_v << "\n";
    std::cout << "KOBAYASHI_VERIFY_HEX"
              << " sum_phi=" << std::hexfloat << sp.sum << std::defaultfloat
              << " sumsq_phi=" << std::hexfloat << sp.sumsq << std::defaultfloat
              << " sum_T=" << std::hexfloat << sT.sum << std::defaultfloat
              << " sumsq_T=" << std::hexfloat << sT.sumsq << "\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg = kobayashi::parse_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) {
          return EXIT_FAILURE;
        }
        run_kobayashi(*cfg, rank, nproc);
        return EXIT_SUCCESS;
      });
}
