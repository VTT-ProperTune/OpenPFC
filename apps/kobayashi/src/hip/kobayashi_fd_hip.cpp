// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kobayashi_fd_hip.cpp
 * @brief MPI + HIP Kobayashi FD driver: one MPI rank binds one GPU (local rank mod device count).
 *
 * Halos are exchanged on the host (same as `allen_cahn_hip`) — portable without GPU-aware MPI.
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "kobayashi_fd_hip requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <kobayashi/cli.hpp>
#include <kobayashi/defaults.hpp>
#include <kobayashi/device_step_hip.hpp>

#include <openpfc/frontend/io/png_writer.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

namespace {

using pfc::field::PaddedBrick;
using pfc::field::for_each_owned;

void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

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

void sync_padded_d2h(const double *dev, PaddedBrick<double> &host) {
  hip_check(hipMemcpy(host.data(), dev, host.size() * sizeof(double),
                      hipMemcpyDeviceToHost),
            "sync_padded_d2h");
}

void sync_padded_h2d(const PaddedBrick<double> &host, double *dev) {
  hip_check(hipMemcpy(dev, host.data(), host.size() * sizeof(double),
                      hipMemcpyHostToDevice),
            "sync_padded_h2d");
}

void run_kobayashi_hip(const kobayashi::RunConfig &cfg, int rank, int nproc) {
  MPI_Comm node_comm{};
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
  int local_rank = 0;
  if (node_comm != MPI_COMM_NULL) {
    MPI_Comm_rank(node_comm, &local_rank);
  }

  int n_dev = 0;
  hip_check(hipGetDeviceCount(&n_dev), "hipGetDeviceCount");
  if (n_dev < 1) {
    throw std::runtime_error("No HIP devices visible to this rank");
  }
  const int dev_id = local_rank % n_dev;
  hip_check(hipSetDevice(dev_id), "hipSetDevice");

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
  PaddedBrick<double> phi_h(decomp, rank, hw);
  PaddedBrick<double> tempr_h(decomp, rank, hw);
  PaddedBrick<double> eps_h(decomp, rank, hw);
  PaddedBrick<double> epsd_h(decomp, rank, hw);
  PaddedBrick<double> px_h(decomp, rank, hw);
  PaddedBrick<double> py_h(decomp, rank, hw);

  const int Nx = cfg.Nx;
  const int Ny = cfg.Ny;
  const int nx = phi_h.nx();
  const int ny = phi_h.ny();
  const int nz = phi_h.nz();

  const int ci = Nx / 2;
  const int cj = Ny / 2;

  for_each_owned(phi_h, [&](int i, int j, int k) {
    (void)k;
    const auto g = phi_h.global(i, j, 0);
    const int gi = g[0];
    const int gj = g[1];
    const double ddx = static_cast<double>(gi - ci);
    const double ddy = static_cast<double>(gj - cj);
    phi_h(i, j, 0) = (ddx * ddx + ddy * ddy < kobayashi::kSeed) ? 1.0 : 0.0;
  });
  for_each_owned(tempr_h, [&](int i, int j, int k) { tempr_h(i, j, k) = 0.0; });

  const std::size_t padded_elems = phi_h.size();
  const std::size_t padded_bytes = padded_elems * sizeof(double);

  double *phi_d = nullptr;
  double *tempr_d = nullptr;
  double *lap_phi_d = nullptr;
  double *lap_t_d = nullptr;
  double *phidx_d = nullptr;
  double *phidy_d = nullptr;
  double *epsilon_d = nullptr;
  double *epsilon_deriv_d = nullptr;

  hip_check(hipMalloc(reinterpret_cast<void **>(&phi_d), padded_bytes), "hipMalloc phi");
  hip_check(hipMalloc(reinterpret_cast<void **>(&tempr_d), padded_bytes), "hipMalloc tempr");
  hip_check(hipMalloc(reinterpret_cast<void **>(&lap_phi_d), padded_bytes), "hipMalloc lap_phi");
  hip_check(hipMalloc(reinterpret_cast<void **>(&lap_t_d), padded_bytes), "hipMalloc lap_t");
  hip_check(hipMalloc(reinterpret_cast<void **>(&phidx_d), padded_bytes), "hipMalloc phidx");
  hip_check(hipMalloc(reinterpret_cast<void **>(&phidy_d), padded_bytes), "hipMalloc phidy");
  hip_check(hipMalloc(reinterpret_cast<void **>(&epsilon_d), padded_bytes), "hipMalloc epsilon");
  hip_check(hipMalloc(reinterpret_cast<void **>(&epsilon_deriv_d), padded_bytes),
            "hipMalloc epsilon_deriv");

  sync_padded_h2d(phi_h, phi_d);
  sync_padded_h2d(tempr_h, tempr_d);

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
    std::cout << "KOBAYASHI_HIP device_id=" << dev_id << " visible_devices=" << n_dev
              << " local_rank=" << local_rank << "\n";
    std::cout << "KOBAYASHI_MPI_COMM_WORLD_SIZE=" << nproc
              << " (must match srun task count)\n";
  }

  int filenum = 0;
  if (!skip_png) {
    sync_padded_d2h(phi_d, phi_h);
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(), filenum);
    if (rank == 0) {
      std::cout << "saving step 0/" << cfg.n_steps << " to file " << path << "\n";
    }
    write_phi_png(rank, decomp, phi_h, path);
    ++filenum;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_loop0 = MPI_Wtime();

  for (int istep = 1; istep <= cfg.n_steps; ++istep) {
    sync_padded_d2h(phi_d, phi_h);
    halo_phi.exchange_halos(phi_h.data(), phi_h.size());
    sync_padded_h2d(phi_h, phi_d);

    sync_padded_d2h(tempr_d, tempr_h);
    halo_t.exchange_halos(tempr_h.data(), tempr_h.size());
    sync_padded_h2d(tempr_h, tempr_d);

    kobayashi::kobayashi_stage_a_hip(phi_d, tempr_d, lap_phi_d, lap_t_d, phidx_d, phidy_d,
                                     epsilon_d, epsilon_deriv_d, nx, ny, nz, hw, inv_dx,
                                     inv_dy, inv_lap_den);

    sync_padded_d2h(epsilon_d, eps_h);
    halo_eps.exchange_halos(eps_h.data(), eps_h.size());
    sync_padded_h2d(eps_h, epsilon_d);

    sync_padded_d2h(epsilon_deriv_d, epsd_h);
    halo_epsd.exchange_halos(epsd_h.data(), epsd_h.size());
    sync_padded_h2d(epsd_h, epsilon_deriv_d);

    sync_padded_d2h(phidx_d, px_h);
    halo_phidx.exchange_halos(px_h.data(), px_h.size());
    sync_padded_h2d(px_h, phidx_d);

    sync_padded_d2h(phidy_d, py_h);
    halo_phidy.exchange_halos(py_h.data(), py_h.size());
    sync_padded_h2d(py_h, phidy_d);

    kobayashi::kobayashi_stage_b_hip(phi_d, tempr_d, lap_phi_d, lap_t_d, epsilon_d,
                                     epsilon_deriv_d, phidx_d, phidy_d, nx, ny, nz, hw,
                                     inv_dx, inv_dy, cfg.dt);

    if (nprint_eff > 0 && istep % nprint_eff == 0 && rank == 0) {
      std::cout << "step " << istep << "/" << cfg.n_steps << " done\n";
    }

    if (!skip_png && cfg.nsave > 0 && istep % cfg.nsave == 0) {
      sync_padded_d2h(phi_d, phi_h);
      char path[4096];
      std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(),
                    filenum);
      if (rank == 0) {
        std::cout << "saving step " << istep << "/" << cfg.n_steps << " to file " << path
                  << "\n";
      }
      write_phi_png(rank, decomp, phi_h, path);
      ++filenum;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_loop1 = MPI_Wtime();
  const double wall_local = t_loop1 - t_loop0;
  double wall_max = 0.0;
  MPI_Reduce(&wall_local, &wall_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  sync_padded_d2h(phi_d, phi_h);
  sync_padded_d2h(tempr_d, tempr_h);

  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_final.png", cfg.output_dir.c_str());
    if (rank == 0) {
      std::cout << "saving final field to " << path << "\n";
    }
    write_phi_png(rank, decomp, phi_h, path);
  }

  hip_check(hipFree(phi_d), "hipFree(phi_d)");
  hip_check(hipFree(tempr_d), "hipFree(tempr_d)");
  hip_check(hipFree(lap_phi_d), "hipFree(lap_phi_d)");
  hip_check(hipFree(lap_t_d), "hipFree(lap_t_d)");
  hip_check(hipFree(phidx_d), "hipFree(phidx_d)");
  hip_check(hipFree(phidy_d), "hipFree(phidy_d)");
  hip_check(hipFree(epsilon_d), "hipFree(epsilon_d)");
  hip_check(hipFree(epsilon_deriv_d), "hipFree(epsilon_deriv_d)");

  if (node_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&node_comm);
  }

  std::vector<double> loc_phi;
  std::vector<double> loc_T;
  pack_owned_xy0(phi_h, loc_phi);
  pack_owned_xy0(tempr_h, loc_T);

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
        try {
          run_kobayashi_hip(*cfg, rank, nproc);
        } catch (const std::exception &e) {
          if (rank == 0) {
            std::cerr << "kobayashi_fd_hip: " << e.what() << "\n";
          }
          return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
      });
}
