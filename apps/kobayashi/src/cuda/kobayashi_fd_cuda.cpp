// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kobayashi_fd_cuda.cpp
 * @brief MPI + CUDA Kobayashi FD driver: one MPI rank binds one GPU (local rank mod
 * device count).
 *
 * Halos use `pfc::cuda::PaddedDeviceHaloExchanger`: GPU-aware MPI on device buffers
 * when available, otherwise narrow face pack/unpack + MPI on pinned host (see
 * `OPENPFC_CUDA_FORCE_PACKED_HALO`).
 *
 * **Performance:** set **`OPENPFC_KOBAYASHI_PERF=1`** to print
 * **`KOBAYASHI_PERF_LOOP`** (wall around the six halo calls per step, both CUDA
 * stages, PNG cadence inside the loop, and residual time vs `wall_loop_max_s`) and,
 * when **`nproc > 1`**, to enable
 * **`OPENPFC_CUDA_PROFILE_HALO`** and print **`OPENPFC_CUDA_PROFILE_HALO_SUMMARY`**
 * from `pfc::cuda::print_cuda_halo_exchange_cpu_timers`.
 */

#if !defined(OpenPFC_ENABLE_CUDA)
#error "kobayashi_fd_cuda requires CUDA (configure with -DOpenPFC_ENABLE_CUDA=ON)"
#endif

#include <cuda_runtime.h>

#include <cmath>
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
#include <kobayashi/device_step_cuda.hpp>

#include <openpfc/frontend/io/png_writer.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/cuda/padded_device_halo_exchange.hpp>

#include "kobayashi_batched_halo.hpp"

namespace {

using pfc::field::for_each_owned;
using pfc::field::PaddedBrick;

void cuda_check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

/// Pin host staging bricks for faster H↔D copies (otherwise memcpy is
/// pageable-limited).
struct HostPinnedRegistration {
  void *ptr = nullptr;
  HostPinnedRegistration(void *p, std::size_t nbytes) : ptr(p) {
    if (nbytes == 0) {
      ptr = nullptr;
      return;
    }
    cuda_check(cudaHostRegister(ptr, nbytes, cudaHostRegisterDefault),
               "cudaHostRegister Kobayashi staging buffer");
  }
  ~HostPinnedRegistration() {
    if (ptr != nullptr) {
      [[maybe_unused]] const cudaError_t u = cudaHostUnregister(ptr);
      (void)u;
    }
  }
  HostPinnedRegistration(const HostPinnedRegistration &) = delete;
  HostPinnedRegistration &operator=(const HostPinnedRegistration &) = delete;
};

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

void gather_global_xy_rank0(const pfc::decomposition::Decomposition &decomp,
                            int rank, int nproc, MPI_Comm comm,
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

  global_out.assign(static_cast<std::size_t>(nx_glob) *
                        static_cast<std::size_t>(ny_glob),
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
                   static_cast<std::size_t>(gy) *
                       static_cast<std::size_t>(nx_glob)] = gathered[offset + li];
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
      const double v = global_xy[static_cast<std::size_t>(gx) +
                                 static_cast<std::size_t>(gy) *
                                     static_cast<std::size_t>(nx_glob)];
      s.sum += v;
      s.sumsq += v * v;
      s.min_v = std::min(s.min_v, v);
      s.max_v = std::max(s.max_v, v);
    }
  }
  return s;
}

void sync_padded_d2h(const double *dev, PaddedBrick<double> &host) {
  cuda_check(cudaMemcpy(host.data(), dev, host.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "sync_padded_d2h");
}

void sync_padded_h2d(const PaddedBrick<double> &host, double *dev) {
  cuda_check(cudaMemcpy(dev, host.data(), host.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "sync_padded_h2d");
}

void run_kobayashi_cuda(const kobayashi::RunConfig &cfg, int rank, int nproc) {
  MPI_Comm node_comm{};
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &node_comm);
  int local_rank = 0;
  if (node_comm != MPI_COMM_NULL) {
    MPI_Comm_rank(node_comm, &local_rank);
  }

  int n_dev = 0;
  cuda_check(cudaGetDeviceCount(&n_dev), "cudaGetDeviceCount");
  if (n_dev < 1) {
    throw std::runtime_error("No CUDA devices visible to this rank");
  }
  const int dev_id = local_rank % n_dev;
  cuda_check(cudaSetDevice(dev_id), "cudaSetDevice");

  const double dx = cfg.dx;
  const double dy = dx;
  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;
  const double inv_lap_den = 1.0 / (dx * dy);

  const auto world = pfc::world::create(pfc::GridSize({cfg.Nx, cfg.Ny, 1}),
                                        pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                        pfc::GridSpacing({dx, dy, 1.0}));
  const auto decomp = pfc::decomposition::create(world, nproc);

  // **Halo width:** `hw=1` is enough for a stage_a stencil that reads i+/-1 and
  // exchanges between stages. **Extended-halo mode (KOBAYASHI_HALO_EXTENDED=1)**
  // grows the halo of all fields to `hw=2` so stage_a can write the eps/eps_d/
  // phidx/phidy 1-cell ring locally and the **second halo exchange of the step
  // disappears** (see loop below). Only meaningful for `nproc > 1`; the device
  // periodic-halo helper used at `nproc == 1` only supports `hw == 1`.
  bool halo_extended_req = std::getenv("KOBAYASHI_HALO_EXTENDED") != nullptr &&
                           std::getenv("KOBAYASHI_HALO_EXTENDED")[0] == '1';
  if (halo_extended_req && nproc == 1) {
    if (rank == 0) {
      std::cout << "KOBAYASHI_HALO_EXTENDED ignored at nproc=1 (single-rank uses "
                   "device-only "
                   "periodic halos with hw=1)\n";
    }
    halo_extended_req = false;
  }
  const bool halo_extended = halo_extended_req;
  const int hw = halo_extended ? 2 : 1;
  const int stage_a_extend = halo_extended ? 1 : 0;
  PaddedBrick<double> phi_h(decomp, rank, hw);
  PaddedBrick<double> tempr_h(decomp, rank, hw);

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

  const HostPinnedRegistration pin_phi(phi_h.data(), phi_h.size() * sizeof(double));
  const HostPinnedRegistration pin_tempr(tempr_h.data(),
                                         tempr_h.size() * sizeof(double));

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

  cuda_check(cudaMalloc(reinterpret_cast<void **>(&phi_d), padded_bytes),
             "cudaMalloc phi");
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&tempr_d), padded_bytes),
             "cudaMalloc tempr");
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&lap_phi_d), padded_bytes),
             "cudaMalloc lap_phi");
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&lap_t_d), padded_bytes),
             "cudaMalloc lap_t");
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&phidx_d), padded_bytes),
             "cudaMalloc phidx");
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&phidy_d), padded_bytes),
             "cudaMalloc phidy");
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&epsilon_d), padded_bytes),
             "cudaMalloc epsilon");
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&epsilon_deriv_d), padded_bytes),
             "cudaMalloc epsilon_deriv");

  sync_padded_h2d(phi_h, phi_d);
  sync_padded_h2d(tempr_h, tempr_d);

  // Single-rank periodic torus: MPI halos only burn CPU (MPI progress) + CUDA global
  // sync; use device-side periodic copies instead (see
  // `kobayashi_periodic_halos_xy_cuda`).
  // **Batched mode (KOBAYASHI_HALO_BATCH=1):** post all fields' halos in one MPI
  // round per exchange point (two per step), reducing MPI_Waitall + sync count 6x ->
  // 2x.
  // **Extended mode (KOBAYASHI_HALO_EXTENDED=1):** stage_a writes its outputs into
  // the 1-cell ring outside the interior, so only {phi, tempr} need to be exchanged
  // per step (one batched call) -- the pre-stage-B exchange is gone entirely.
  std::unique_ptr<pfc::cuda::PaddedDeviceHaloExchanger> halo_phi;
  std::unique_ptr<pfc::cuda::PaddedDeviceHaloExchanger> halo_t;
  std::unique_ptr<pfc::cuda::PaddedDeviceHaloExchanger> halo_eps;
  std::unique_ptr<pfc::cuda::PaddedDeviceHaloExchanger> halo_epsd;
  std::unique_ptr<pfc::cuda::PaddedDeviceHaloExchanger> halo_phidx;
  std::unique_ptr<pfc::cuda::PaddedDeviceHaloExchanger> halo_phidy;
  std::unique_ptr<kobayashi::cuda::BatchedPaddedDeviceHalo> halo_pre_a;
  std::unique_ptr<kobayashi::cuda::BatchedPaddedDeviceHalo> halo_pre_b;
  const bool perf_k = std::getenv("OPENPFC_KOBAYASHI_PERF") != nullptr;
  if (perf_k && nproc > 1 && std::getenv("OPENPFC_CUDA_PROFILE_HALO") == nullptr) {
    (void)setenv("OPENPFC_CUDA_PROFILE_HALO", "1", 1);
  }
  const bool halo_batch = std::getenv("KOBAYASHI_HALO_BATCH") != nullptr &&
                          std::getenv("KOBAYASHI_HALO_BATCH")[0] == '1';
  // Extended mode always batches {phi, tempr} (only 1 exchange per step, 2 fields).
  const bool use_batched_pre_a = halo_batch || halo_extended;

  // Kobayashi FD is a 2D slab problem (`GridSize({Nx, Ny, 1})`); the CUDA
  // stage_a / stage_b kernels hardcode `iz = 0` and never read `k±1`. Use the
  // 2D in-plane axes preset so the exchangers skip ±Z entirely — this avoids
  // packing or MPI-to-self transferring the full nx*ny ±Z face slabs (which
  // would be ~128 MiB per message at 4096^2). The legacy
  // `kobayashi_periodic_halos_z_edges_hw1_kernel` is no longer needed in the
  // multi-rank path; tracked as a follow-up cleanup.
  const auto halo_dirs = pfc::halo::presets::Axes2D();

  if (nproc > 1) {
    if (halo_extended) {
      // corner_fill=true: ±X MPI first, then self ±Y pack with widened X so that
      // the X-Y corner halos used by extended stage_a's 5-point stencil at
      // (-1, 0) / (-1, ny-1) etc. are populated correctly.
      halo_pre_a = std::make_unique<kobayashi::cuda::BatchedPaddedDeviceHalo>(
          decomp, rank, hw, MPI_COMM_WORLD, /*n_fields=*/2, halo_dirs,
          /*base_tag=*/0, /*corner_fill=*/true);
      // No halo_pre_b: stage_a writes the 1-cell ring of its outputs directly.
    } else if (use_batched_pre_a) {
      // Pre-stage-A batch: phi, tempr (base_tag=0). Pre-stage-B batch: eps,
      // eps_deriv, phidx, phidy (base_tag=200) -- well separated from the
      // pre-A range (2 fields x 6 face slots = 12 tags below 200).
      halo_pre_a = std::make_unique<kobayashi::cuda::BatchedPaddedDeviceHalo>(
          decomp, rank, hw, MPI_COMM_WORLD, /*n_fields=*/2, halo_dirs,
          /*base_tag=*/0);
      halo_pre_b = std::make_unique<kobayashi::cuda::BatchedPaddedDeviceHalo>(
          decomp, rank, hw, MPI_COMM_WORLD, /*n_fields=*/4, halo_dirs,
          /*base_tag=*/200);
    } else {
      halo_phi = std::make_unique<pfc::cuda::PaddedDeviceHaloExchanger>(
          decomp, rank, hw, MPI_COMM_WORLD, halo_dirs, 0);
      halo_t = std::make_unique<pfc::cuda::PaddedDeviceHaloExchanger>(
          decomp, rank, hw, MPI_COMM_WORLD, halo_dirs, 20);
      halo_eps = std::make_unique<pfc::cuda::PaddedDeviceHaloExchanger>(
          decomp, rank, hw, MPI_COMM_WORLD, halo_dirs, 40);
      halo_epsd = std::make_unique<pfc::cuda::PaddedDeviceHaloExchanger>(
          decomp, rank, hw, MPI_COMM_WORLD, halo_dirs, 60);
      halo_phidx = std::make_unique<pfc::cuda::PaddedDeviceHaloExchanger>(
          decomp, rank, hw, MPI_COMM_WORLD, halo_dirs, 80);
      halo_phidy = std::make_unique<pfc::cuda::PaddedDeviceHaloExchanger>(
          decomp, rank, hw, MPI_COMM_WORLD, halo_dirs, 100);
    }
  }

  double perf_sum_exchange = 0.0;
  double perf_sum_stage_a = 0.0;
  double perf_sum_stage_b = 0.0;
  double perf_sum_png_loop = 0.0;

  auto exchange_padded = [&](double *d_padded,
                             pfc::cuda::PaddedDeviceHaloExchanger *ex) {
    const double t0 = perf_k ? MPI_Wtime() : 0.0;
    if (nproc == 1) {
      kobayashi::kobayashi_periodic_halos_xy_cuda(d_padded, nx, ny, nz, hw);
    } else {
      ex->exchange_halos_device(d_padded, padded_elems, nullptr);
    }
    if (perf_k) {
      perf_sum_exchange += MPI_Wtime() - t0;
    }
  };

  auto exchange_batch = [&](kobayashi::cuda::BatchedPaddedDeviceHalo *ex,
                            std::initializer_list<double *> fields) {
    const double t0 = perf_k ? MPI_Wtime() : 0.0;
    if (nproc == 1) {
      for (double *p : fields) {
        kobayashi::kobayashi_periodic_halos_xy_cuda(p, nx, ny, nz, hw);
      }
    } else {
      ex->exchange(fields, nullptr);
    }
    if (perf_k) {
      perf_sum_exchange += MPI_Wtime() - t0;
    }
  };

  const bool skip_png = std::getenv("OPENPFC_KOBAYASHI_SKIP_PNG") != nullptr;
  const bool quiet = std::getenv("OPENPFC_KOBAYASHI_QUIET") != nullptr;
  const int nprint_eff = quiet ? 0 : cfg.nprint;

  if (rank == 0) {
    std::filesystem::create_directories(cfg.output_dir);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "KOBAYASHI_CUDA device_id=" << dev_id
              << " visible_devices=" << n_dev << " local_rank=" << local_rank
              << "\n";
    std::cout << "KOBAYASHI_MPI_COMM_WORLD_SIZE=" << nproc
              << " (must match srun task count)\n";
    if (nproc == 1) {
      std::cout << "KOBAYASHI_CUDA_HALO_MODE=device_periodic_local hw=" << hw
                << "\n";
    } else {
      const bool gpu_aware = use_batched_pre_a ? halo_pre_a->uses_gpu_aware_mpi()
                                               : halo_phi->uses_gpu_aware_mpi();
      std::cout << "KOBAYASHI_CUDA_HALO_MODE="
                << (gpu_aware ? "gpu_aware_mpi" : "packed_faces_pcie")
                << " halo_batch=" << (halo_batch ? "on" : "off")
                << " halo_extended=" << (halo_extended ? "on" : "off")
                << " hw=" << hw << "\n";
    }
  }

  int filenum = 0;
  if (!skip_png) {
    sync_padded_d2h(phi_d, phi_h);
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(),
                  filenum);
    if (rank == 0) {
      std::cout << "saving step 0/" << cfg.n_steps << " to file " << path << "\n";
    }
    write_phi_png(rank, decomp, phi_h, path);
    ++filenum;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_loop0 = MPI_Wtime();

  for (int istep = 1; istep <= cfg.n_steps; ++istep) {
    if (use_batched_pre_a) {
      exchange_batch(halo_pre_a.get(), {phi_d, tempr_d});
    } else {
      exchange_padded(phi_d, halo_phi.get());
      exchange_padded(tempr_d, halo_t.get());
    }

    if (perf_k) {
      const double t0 = MPI_Wtime();
      kobayashi::kobayashi_stage_a_cuda(phi_d, tempr_d, lap_phi_d, lap_t_d, phidx_d,
                                        phidy_d, epsilon_d, epsilon_deriv_d, nx, ny,
                                        nz, hw, inv_dx, inv_dy, inv_lap_den,
                                        stage_a_extend);
      perf_sum_stage_a += MPI_Wtime() - t0;
    } else {
      kobayashi::kobayashi_stage_a_cuda(phi_d, tempr_d, lap_phi_d, lap_t_d, phidx_d,
                                        phidy_d, epsilon_d, epsilon_deriv_d, nx, ny,
                                        nz, hw, inv_dx, inv_dy, inv_lap_den,
                                        stage_a_extend);
    }

    if (halo_extended) {
      // No exchange: stage_a wrote the 1-cell ring of eps, eps_d, phidx, phidy.
    } else if (use_batched_pre_a) {
      exchange_batch(halo_pre_b.get(),
                     {epsilon_d, epsilon_deriv_d, phidx_d, phidy_d});
    } else {
      exchange_padded(epsilon_d, halo_eps.get());
      exchange_padded(epsilon_deriv_d, halo_epsd.get());
      exchange_padded(phidx_d, halo_phidx.get());
      exchange_padded(phidy_d, halo_phidy.get());
    }

    if (perf_k) {
      const double t0 = MPI_Wtime();
      kobayashi::kobayashi_stage_b_cuda(phi_d, tempr_d, lap_phi_d, lap_t_d,
                                        epsilon_d, epsilon_deriv_d, phidx_d, phidy_d,
                                        nx, ny, nz, hw, inv_dx, inv_dy, cfg.dt);
      perf_sum_stage_b += MPI_Wtime() - t0;
    } else {
      kobayashi::kobayashi_stage_b_cuda(phi_d, tempr_d, lap_phi_d, lap_t_d,
                                        epsilon_d, epsilon_deriv_d, phidx_d, phidy_d,
                                        nx, ny, nz, hw, inv_dx, inv_dy, cfg.dt);
    }

    if (nprint_eff > 0 && istep % nprint_eff == 0 && rank == 0) {
      std::cout << "step " << istep << "/" << cfg.n_steps << " done\n";
    }

    if (!skip_png && cfg.nsave > 0 && istep % cfg.nsave == 0) {
      const double t_png0 = perf_k ? MPI_Wtime() : 0.0;
      sync_padded_d2h(phi_d, phi_h);
      char path[4096];
      std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(),
                    filenum);
      if (rank == 0) {
        std::cout << "saving step " << istep << "/" << cfg.n_steps << " to file "
                  << path << "\n";
      }
      write_phi_png(rank, decomp, phi_h, path);
      ++filenum;
      if (perf_k) {
        perf_sum_png_loop += MPI_Wtime() - t_png0;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_loop1 = MPI_Wtime();
  const double wall_local = t_loop1 - t_loop0;
  double wall_max = 0.0;
  MPI_Reduce(&wall_local, &wall_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (perf_k) {
    const double accounted_local =
        perf_sum_exchange + perf_sum_stage_a + perf_sum_stage_b + perf_sum_png_loop;
    const double unaccounted_local = wall_local - accounted_local;
    double mx_ex = 0.0;
    double mx_a = 0.0;
    double mx_b = 0.0;
    double mx_png = 0.0;
    double mx_unacc = 0.0;
    MPI_Reduce(&perf_sum_exchange, &mx_ex, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&perf_sum_stage_a, &mx_a, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&perf_sum_stage_b, &mx_b, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&perf_sum_png_loop, &mx_png, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&unaccounted_local, &mx_unacc, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
      const double n = static_cast<double>(cfg.n_steps);
      std::cout << std::setprecision(17);
      std::cout << "KOBAYASHI_PERF_LOOP"
                << " n_steps=" << cfg.n_steps << " nproc=" << nproc
                << " exchange_driver_wall_s_max=" << mx_ex
                << " exchange_per_step_avg_s_max=" << (mx_ex / n)
                << " stage_a_wall_s_max=" << mx_a
                << " stage_a_per_step_avg_s_max=" << (mx_a / n)
                << " stage_b_wall_s_max=" << mx_b
                << " stage_b_per_step_avg_s_max=" << (mx_b / n)
                << " png_in_loop_wall_s_max=" << mx_png
                << " unaccounted_wall_s_max=" << mx_unacc
                << " wall_loop_max_s=" << wall_max
                << " (unaccounted = loop_wall - sum of measured buckets on slowest "
                   "rank)\n";
    }
    pfc::cuda::print_cuda_halo_exchange_cpu_timers(MPI_COMM_WORLD);
  }

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

  cuda_check(cudaFree(phi_d), "cudaFree(phi_d)");
  cuda_check(cudaFree(tempr_d), "cudaFree(tempr_d)");
  cuda_check(cudaFree(lap_phi_d), "cudaFree(lap_phi_d)");
  cuda_check(cudaFree(lap_t_d), "cudaFree(lap_t_d)");
  cuda_check(cudaFree(phidx_d), "cudaFree(phidx_d)");
  cuda_check(cudaFree(phidy_d), "cudaFree(phidy_d)");
  cuda_check(cudaFree(epsilon_d), "cudaFree(epsilon_d)");
  cuda_check(cudaFree(epsilon_deriv_d), "cudaFree(epsilon_deriv_d)");

  if (node_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&node_comm);
  }

  std::vector<double> loc_phi;
  std::vector<double> loc_T;
  pack_owned_xy0(phi_h, loc_phi);
  pack_owned_xy0(tempr_h, loc_T);

  std::vector<double> g_phi;
  std::vector<double> g_T;
  gather_global_xy_rank0(decomp, rank, nproc, MPI_COMM_WORLD, loc_phi, Nx, Ny,
                         g_phi);
  gather_global_xy_rank0(decomp, rank, nproc, MPI_COMM_WORLD, loc_T, Nx, Ny, g_T);

  if (rank == 0) {
    const FieldStats sp = stats_global_ordered(g_phi, Nx, Ny);
    const FieldStats sT = stats_global_ordered(g_T, Nx, Ny);
    const double l2_phi = std::sqrt(sp.sumsq);
    const double l2_T = std::sqrt(sT.sumsq);
    std::cout << std::setprecision(17);
    std::cout << "KOBAYASHI_VERIFY"
              << " wall_loop_max_s=" << wall_max << " nproc=" << nproc
              << " Nx=" << Nx << " Ny=" << Ny << " steps=" << cfg.n_steps
              << " dt=" << cfg.dt << " dx=" << cfg.dx << " sum_phi=" << sp.sum
              << " sumsq_phi=" << sp.sumsq << " l2_phi=" << l2_phi
              << " min_phi=" << sp.min_v << " max_phi=" << sp.max_v
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
          run_kobayashi_cuda(*cfg, rank, nproc);
        } catch (const std::exception &e) {
          if (rank == 0) {
            std::cerr << "kobayashi_fd_cuda: " << e.what() << "\n";
          }
          return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
      });
}
