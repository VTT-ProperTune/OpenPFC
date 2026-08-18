// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kobayashi_fd_hip.cpp
 * @brief MPI + HIP Kobayashi FD driver: one MPI rank binds one GPU (local rank mod device count).
 *
 * Halos use `pfc::comm::HaloExchange<HipSpace>` on device-resident Fields
 * (same two groups as the CPU driver; `Axes2D()` skips ±Z on the nz=1 slab).
 * PNG / verify still stage \(\phi\) and \(T\) to host after the timed loop
 * (and at `nsave` snapshots).
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "kobayashi_fd_hip requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
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
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>

#include <kobayashi/verification_utilities.hpp>

namespace {



using HostField = pfc::data::Field<double, pfc::HostSpace>;
using DevField = pfc::data::Field<double, pfc::HipSpace>;
using pfc::data::field_from_subdomain;

class mpi_comm_guard {
private:
    MPI_Comm comm_;

public:
    explicit mpi_comm_guard(MPI_Comm comm = MPI_COMM_NULL) : comm_(comm) {}
    ~mpi_comm_guard() noexcept {
        if (comm_ != MPI_COMM_NULL && comm_ != MPI_COMM_WORLD) {
            (void)MPI_Comm_free(&comm_);  // Discard error to preserve no-throw guarantee
        }
    }
    mpi_comm_guard(const mpi_comm_guard&) = delete;
    mpi_comm_guard& operator=(const mpi_comm_guard&) = delete;
    mpi_comm_guard(mpi_comm_guard&& other) noexcept : comm_(other.comm_) {
        other.comm_ = MPI_COMM_NULL;
    }
    mpi_comm_guard& operator=(mpi_comm_guard&& other) noexcept {
        if (this != &other) {
            if (comm_ != MPI_COMM_NULL && comm_ != MPI_COMM_WORLD) {
                (void)MPI_Comm_free(&comm_);
            }
            comm_ = other.comm_;
            other.comm_ = MPI_COMM_NULL;
        }
        return *this;
    }

    operator MPI_Comm() const { return comm_; }
    MPI_Comm get() const { return comm_; }
    MPI_Comm release() {
        MPI_Comm tmp = comm_;
        comm_ = MPI_COMM_NULL;
        return tmp;
    }
};

void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

DevField make_dev_field(const pfc::decomposition::Decomposition &decomp, int rank,
                        int hw) {
  return DevField(pfc::decomposition::domain(decomp),
                  pfc::decomposition::local_box(decomp, rank), hw);
}

void copy_host_to_device(const HostField &host, DevField &dev) {
  if (host.size() != dev.size()) {
    throw std::runtime_error("copy_host_to_device: size mismatch");
  }
  dev.with_host_view([&](double *data, std::size_t n) {
    std::copy(host.data(), host.data() + n, data);
  });
  dev.sync_to_device();
}

void copy_device_to_host(DevField &dev, HostField &host) {
  if (host.size() != dev.size()) {
    throw std::runtime_error("copy_device_to_host: size mismatch");
  }
  dev.with_host_view([&](double *data, std::size_t n) {
    std::copy(data, data + n, host.data());
  });
  // Read-only pull: keep the device buffer as the source of truth.
  dev.note_device_write();
}

void run_kobayashi_hip(const kobayashi::RunConfig &cfg, int rank, int nproc) {
  MPI_Comm temp_node_comm{};
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &temp_node_comm);
  mpi_comm_guard node_comm(temp_node_comm);
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

  const auto domain = pfc::domain::create(pfc::GridSize({cfg.Nx, cfg.Ny, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({dx, dy, 1.0}));
  const auto decomp = pfc::decomposition::create(domain, nproc);

  constexpr int hw = 1;
  auto phi_h = field_from_subdomain<double>(decomp, rank, hw);
  auto tempr_h = field_from_subdomain<double>(decomp, rank, hw);

  auto phi = make_dev_field(decomp, rank, hw);
  auto tempr = make_dev_field(decomp, rank, hw);
  auto lap_phi = make_dev_field(decomp, rank, hw);
  auto lap_t = make_dev_field(decomp, rank, hw);
  auto phidx = make_dev_field(decomp, rank, hw);
  auto phidy = make_dev_field(decomp, rank, hw);
  auto epsilon = make_dev_field(decomp, rank, hw);
  auto epsilon_deriv = make_dev_field(decomp, rank, hw);

  const int Nx = cfg.Nx;
  const int Ny = cfg.Ny;
  const int nx = phi.local_size()[0];
  const int ny = phi.local_size()[1];
  const int nz = phi.local_size()[2];

  const int ci = Nx / 2;
  const int cj = Ny / 2;

  phi_h.for_each_owned([&](int i, int j, int k) {
    (void)k;
    const auto g = phi_h.global(i, j, 0);
    const int gi = g[0];
    const int gj = g[1];
    const double ddx = static_cast<double>(gi - ci);
    const double ddy = static_cast<double>(gj - cj);
    phi_h(i, j, 0) = (ddx * ddx + ddy * ddy < kobayashi::kSeed) ? 1.0 : 0.0;
  });
  tempr_h.for_each_owned([&](int i, int j, int k) { tempr_h(i, j, k) = 0.0; });

  copy_host_to_device(phi_h, phi);
  copy_host_to_device(tempr_h, tempr);

  pfc::comm::HaloExchangeOptions state_opt;
  state_opt.directions = pfc::halo::presets::Axes2D();
  pfc::comm::HaloExchange<pfc::HipSpace, double> halo_state(
      {&phi, &tempr}, decomp, rank, MPI_COMM_WORLD, state_opt);
  pfc::comm::HaloExchangeOptions aux_opt;
  aux_opt.exchange_base = 2;
  aux_opt.directions = pfc::halo::presets::Axes2D();
  pfc::comm::HaloExchange<pfc::HipSpace, double> halo_aux(
      {&epsilon, &epsilon_deriv, &phidx, &phidy}, decomp, rank, MPI_COMM_WORLD,
      aux_opt);

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
    std::cout << "KOBAYASHI_HIP_HALO_MODE=device"
              << " gpu_aware=" << (halo_state.uses_gpu_aware_mpi() ? 1 : 0)
              << " contiguous=" << (halo_state.uses_contiguous_device_mpi() ? 1 : 0)
              << "\n";
  }

  int filenum = 0;
  if (!skip_png) {
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
    halo_state.exchange();

    kobayashi::kobayashi_stage_a_hip(phi.data(), tempr.data(), lap_phi.data(),
                                     lap_t.data(), phidx.data(), phidy.data(),
                                     epsilon.data(), epsilon_deriv.data(), nx, ny,
                                     nz, hw, inv_dx, inv_dy, inv_lap_den);
    epsilon.note_device_write();
    epsilon_deriv.note_device_write();
    phidx.note_device_write();
    phidy.note_device_write();

    halo_aux.exchange();

    kobayashi::kobayashi_stage_b_hip(phi.data(), tempr.data(), lap_phi.data(),
                                     lap_t.data(), epsilon.data(),
                                     epsilon_deriv.data(), phidx.data(),
                                     phidy.data(), nx, ny, nz, hw, inv_dx, inv_dy,
                                     cfg.dt);
    phi.note_device_write();
    tempr.note_device_write();

    if (nprint_eff > 0 && istep % nprint_eff == 0 && rank == 0) {
      std::cout << "step " << istep << "/" << cfg.n_steps << " done\n";
    }

    if (!skip_png && cfg.nsave > 0 && istep % cfg.nsave == 0) {
      copy_device_to_host(phi, phi_h);
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

  copy_device_to_host(phi, phi_h);
  copy_device_to_host(tempr, tempr_h);

  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_final.png", cfg.output_dir.c_str());
    if (rank == 0) {
      std::cout << "saving final field to " << path << "\n";
    }
    write_phi_png(rank, decomp, phi_h, path);
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
        run_kobayashi_hip(*cfg, rank, nproc);
        return EXIT_SUCCESS;
      });
}
