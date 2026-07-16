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
#include <memory>
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

#include <kobayashi/verification_utilities.hpp>

namespace {



// RAII wrappers for HIP device memory and MPI communicators
struct hip_buffer_deleter {
  void operator()(void* ptr) const noexcept {
    if (ptr) {
      (void)hipFree(ptr);  // Discard error to preserve no-throw guarantee
    }
  }
};

template<typename T>
using hip_buffer_guard = std::unique_ptr<T, hip_buffer_deleter>;

class mpi_comm_guard {
private:
  MPI_Comm comm_;
  struct mpi_comm_deleter {
    void operator()(MPI_Comm* comm) const noexcept {
      if (comm && *comm != MPI_COMM_NULL && *comm != MPI_COMM_WORLD) {
        (void)MPI_Comm_free(comm);  // Discard error to preserve no-throw guarantee
      }
    }
  };
  std::unique_ptr<MPI_Comm, mpi_comm_deleter> holder_;

public:
  explicit mpi_comm_guard(MPI_Comm comm = MPI_COMM_NULL)
      : comm_(comm) {
    if (comm != MPI_COMM_NULL) {
      holder_ = std::unique_ptr<MPI_Comm, mpi_comm_deleter>(new MPI_Comm(comm), mpi_comm_deleter{});
    }
  }

  ~mpi_comm_guard() noexcept = default;

  // Disable copy, enable move
  mpi_comm_guard(const mpi_comm_guard&) = delete;
  mpi_comm_guard& operator=(const mpi_comm_guard&) = delete;
  mpi_comm_guard(mpi_comm_guard&& other) noexcept
      : comm_(other.comm_), holder_(std::move(other.holder_)) {
    other.comm_ = MPI_COMM_NULL;
  }

  mpi_comm_guard& operator=(mpi_comm_guard&& other) noexcept {
    if (this != &other) {
      comm_ = other.comm_;
      holder_ = std::move(other.holder_);
      other.comm_ = MPI_COMM_NULL;
    }
    return *this;
  }

  // Implicit conversion to raw communicator for seamless integration
  operator MPI_Comm() const { return comm_; }
  MPI_Comm get() const { return comm_; }
  MPI_Comm release() {
    MPI_Comm tmp = comm_;
    comm_ = MPI_COMM_NULL;
    if (holder_) {
      holder_.release();  // Release ownership without freeing
    }
    return tmp;
  }
};
using pfc::field::PaddedBrick;
using pfc::field::for_each_owned;

void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
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

  double *temp_phi_d = nullptr;
  hip_check(hipMalloc(reinterpret_cast<void **>(&temp_phi_d), padded_bytes), "hipMalloc phi");
  hip_buffer_guard<double> phi_d(temp_phi_d);

  double *temp_tempr_d = nullptr;
  hip_check(hipMalloc(reinterpret_cast<void **>(&temp_tempr_d), padded_bytes), "hipMalloc tempr");
  hip_buffer_guard<double> tempr_d(temp_tempr_d);

  double *temp_lap_phi_d = nullptr;
  hip_check(hipMalloc(reinterpret_cast<void **>(&temp_lap_phi_d), padded_bytes), "hipMalloc lap_phi");
  hip_buffer_guard<double> lap_phi_d(temp_lap_phi_d);

  double *temp_lap_t_d = nullptr;
  hip_check(hipMalloc(reinterpret_cast<void **>(&temp_lap_t_d), padded_bytes), "hipMalloc lap_t");
  hip_buffer_guard<double> lap_t_d(temp_lap_t_d);

  double *temp_phidx_d = nullptr;
  hip_check(hipMalloc(reinterpret_cast<void **>(&temp_phidx_d), padded_bytes), "hipMalloc phidx");
  hip_buffer_guard<double> phidx_d(temp_phidx_d);

  double *temp_phidy_d = nullptr;
  hip_check(hipMalloc(reinterpret_cast<void **>(&temp_phidy_d), padded_bytes), "hipMalloc phidy");
  hip_buffer_guard<double> phidy_d(temp_phidy_d);

  double *temp_epsilon_d = nullptr;
  hip_check(hipMalloc(reinterpret_cast<void **>(&temp_epsilon_d), padded_bytes), "hipMalloc epsilon");
  hip_buffer_guard<double> epsilon_d(temp_epsilon_d);

  double *temp_epsilon_deriv_d = nullptr;
  hip_check(hipMalloc(reinterpret_cast<void **>(&temp_epsilon_deriv_d), padded_bytes), "hipMalloc epsilon_deriv");
  hip_buffer_guard<double> epsilon_deriv_d(temp_epsilon_deriv_d);

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
