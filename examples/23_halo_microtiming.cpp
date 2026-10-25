// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 23_halo_microtiming.cpp
 * @brief Timed `HaloExchange` loop that exports schema-v2 profiling JSON.
 *
 * Host Faces is the default. `--cuda` / `--hip` select the device facade.
 * Each timed iteration is one `wall_step` (barrier, exchange, barrier).
 *
 *   mpirun -n 16 ./examples/23_halo_microtiming --hip --nx 512 --ny 512 \\
 *     --nz 1024 --halo 1 --iters 50 --output results/halo/hip_2n.json
 *   OPENPFC_FD_PROC_GRID=2,2,4 mpirun -n 16 ...
 */

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#endif
#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

using namespace pfc;

namespace {

struct Options {
  bool use_cuda = false;
  bool use_hip = false;
  bool full = false;
  int nx = 128;
  int ny = 0;
  int nz = 0;
  int halo = 1;
  int iters = 50;
  int warmup = 5;
  std::string output;
};

[[noreturn]] void usage(int code) {
  std::cerr << "Usage: 23_halo_microtiming [--cuda|--hip] [--full] [--nx N] "
               "[--ny N] [--nz N] [--halo N] [--iters N] [--warmup N] "
               "--output PATH.json\n"
            << "  Ny/Nz default to Nx. OPENPFC_FD_PROC_GRID=gx,gy,gz selects "
               "the Cartesian split.\n";
  std::exit(code);
}

Options parse_args(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a{argv[i]};
    auto need = [&](int &dst) {
      if (i + 1 >= argc) {
        usage(2);
      }
      dst = std::atoi(argv[++i]);
    };
    if (a == "--help" || a == "-h") {
      usage(0);
    } else if (a == "--cuda") {
      opt.use_cuda = true;
    } else if (a == "--hip") {
      opt.use_hip = true;
    } else if (a == "--full") {
      opt.full = true;
    } else if (a == "--nx") {
      need(opt.nx);
    } else if (a == "--ny") {
      need(opt.ny);
    } else if (a == "--nz") {
      need(opt.nz);
    } else if (a == "--halo") {
      need(opt.halo);
    } else if (a == "--iters") {
      need(opt.iters);
    } else if (a == "--warmup") {
      need(opt.warmup);
    } else if (a == "--output") {
      if (i + 1 >= argc) {
        usage(2);
      }
      opt.output = argv[++i];
    } else {
      usage(2);
    }
  }
  if (opt.ny <= 0) {
    opt.ny = opt.nx;
  }
  if (opt.nz <= 0) {
    opt.nz = opt.nx;
  }
  if (opt.output.empty() || opt.nx <= 0 || opt.ny <= 0 || opt.nz <= 0 ||
      opt.halo < 0 || opt.iters <= 0 || opt.warmup < 0) {
    usage(2);
  }
  if (opt.use_cuda && opt.use_hip) {
    std::cerr << "23_halo_microtiming: choose at most one of --cuda / --hip\n";
    std::exit(2);
  }
  return opt;
}

template <typename Space>
void fill_owned(data::Field<double, Space> &u, double rank_val) {
  u.with_host_view([&](double *data, std::size_t) {
    const auto n = u.size3();
    for (int k = 0; k < n[2]; ++k) {
      for (int j = 0; j < n[1]; ++j) {
        for (int i = 0; i < n[0]; ++i) {
          data[u.idx(i, j, k)] = rank_val + 0.001 * static_cast<double>(i) +
                                 0.01 * static_cast<double>(j) +
                                 0.1 * static_cast<double>(k);
        }
      }
    }
  });
}

template <typename Space>
void run_timed([[maybe_unused]] data::Field<double, Space> &u,
               comm::HaloExchange<Space, double> &halo, const Options &opt,
               int rank) {
  using namespace pfc::profiling;
  for (int w = 0; w < opt.warmup; ++w) {
    halo.exchange();
  }
  MPI_Barrier(MPI_COMM_WORLD);

  ProfilingSession session(ProfilingMetricCatalog::with_defaults_and_extras({}),
                           ProfilingSession::openpfc_default_frame_metrics());
  ProfilingContextScope ctx(&session);

  for (int step = 0; step < opt.iters; ++step) {
    openpfc_begin_frame_with_step_and_rank(session, step, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    halo.exchange();
    MPI_Barrier(MPI_COMM_WORLD);
    openpfc_end_frame_step_wall_and_memory(session, MPI_Wtime() - t0, 0, 0, 0);
  }

  ProfilingExportOptions exp;
  exp.write_json = true;
  exp.json_path = opt.output;
  session.finalize_and_export(MPI_COMM_WORLD, exp);
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const Options opt = parse_args(argc, argv);

  try {
    if (opt.use_cuda || opt.use_hip) {
      pfc::runtime::gpu::bind_local_device();
    }
    auto domain = domain::create({opt.nx, opt.ny, opt.nz});
    auto decomp = decomposition::create(domain, nproc);
    const auto grid = decomposition::get_grid(decomp);
    const auto box = decomposition::local_box(decomp, rank);
    char host[256];
    if (gethostname(host, sizeof(host)) != 0) {
      std::snprintf(host, sizeof(host), "unknown");
    }
    int loc[3] = {box.size[0], box.size[1], box.size[2]};
    int loc_min[3] = {0, 0, 0};
    int loc_max[3] = {0, 0, 0};
    MPI_Allreduce(loc, loc_min, 3, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(loc, loc_max, 3, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    host[sizeof(host) - 1] = '\0';
    std::vector<char> hosts(static_cast<std::size_t>(nproc) * 256);
    MPI_Allgather(host, 256, MPI_CHAR, hosts.data(), 256, MPI_CHAR, MPI_COMM_WORLD);
    const int gx = grid[0];
    const int gy = grid[1];
    const int rx = rank % gx;
    const int ry = (rank / gx) % gy;
    const int rz = rank / (gx * gy);
    int gpu = -1;
#if defined(OpenPFC_ENABLE_HIP)
    if (opt.use_hip) {
      (void)hipGetDevice(&gpu);
    }
#endif
    int offnode = 0;
    const std::array<std::array<int, 3>, 6> faces = {{{{1, 0, 0}},
                                                      {{-1, 0, 0}},
                                                      {{0, 1, 0}},
                                                      {{0, -1, 0}},
                                                      {{0, 0, 1}},
                                                      {{0, 0, -1}}}};
    for (const auto &dir : faces) {
      const int peer =
          decomposition::get_neighbor_rank(decomp, rank, {dir[0], dir[1], dir[2]});
      if (peer < 0 || peer == rank) {
        continue;
      }
      if (std::strncmp(host, &hosts[static_cast<std::size_t>(peer) * 256], 256) !=
          0) {
        ++offnode;
      }
    }
    int off_sum = 0;
    int off_max = 0;
    MPI_Allreduce(&offnode, &off_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&offnode, &off_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    std::vector<int> gpus(static_cast<std::size_t>(nproc));
    std::vector<int> coords(static_cast<std::size_t>(nproc) * 4);
    const int local_meta[4] = {rx, ry, rz, offnode};
    MPI_Gather(&gpu, 1, MPI_INT, gpus.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_meta, 4, MPI_INT, coords.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "HALO_DECOMP proc_grid=" << grid[0] << "x" << grid[1] << "x"
                << grid[2] << " global=" << opt.nx << "x" << opt.ny << "x" << opt.nz
                << " local_min=" << loc_min[0] << "x" << loc_min[1] << "x"
                << loc_min[2] << " local_max=" << loc_max[0] << "x" << loc_max[1]
                << "x" << loc_max[2] << " halo=" << opt.halo << " nproc=" << nproc
                << " offnode_sum=" << off_sum << " offnode_max=" << off_max
                << std::endl;
      std::ofstream plc("halo_placement.txt");
      plc << "rank host gpu rx ry rz offnode_faces\n";
      for (int r = 0; r < nproc; ++r) {
        plc << r << " " << &hosts[static_cast<std::size_t>(r) * 256] << " "
            << gpus[r] << " " << coords[static_cast<std::size_t>(r) * 4] << " "
            << coords[static_cast<std::size_t>(r) * 4 + 1] << " "
            << coords[static_cast<std::size_t>(r) * 4 + 2] << " "
            << coords[static_cast<std::size_t>(r) * 4 + 3] << "\n";
      }
    }
    comm::HaloExchangeOptions hop;
    if (opt.full) {
      hop.connectivity = comm::HaloConnectivity::Full;
    }

    if (opt.use_cuda) {
#if defined(OpenPFC_ENABLE_CUDA)
      auto u = data::Field<double, CUDASpace>(decomposition::domain(decomp),
                                              decomposition::local_box(decomp, rank),
                                              opt.halo);
      fill_owned(u, static_cast<double>(rank));
      comm::HaloExchange<CUDASpace, double> halo(u, decomp, rank, MPI_COMM_WORLD,
                                                 hop);
      run_timed(u, halo, opt, rank);
#else
      throw std::runtime_error("23_halo_microtiming: --cuda requires CUDA build");
#endif
    } else if (opt.use_hip) {
#if defined(OpenPFC_ENABLE_HIP)
      auto u = data::Field<double, HIPSpace>(decomposition::domain(decomp),
                                             decomposition::local_box(decomp, rank),
                                             opt.halo);
      fill_owned(u, static_cast<double>(rank));
      comm::HaloExchange<HIPSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD,
                                                hop);
      if (rank == 0) {
        std::cout << "HALO_MODE gpu_aware=" << (halo.uses_gpu_aware_mpi() ? 1 : 0)
                  << " contiguous=" << (halo.uses_contiguous_device_mpi() ? 1 : 0)
                  << std::endl;
      }
      run_timed(u, halo, opt, rank);
#else
      throw std::runtime_error("23_halo_microtiming: --hip requires HIP build");
#endif
    } else {
      auto u = data::field_from_subdomain<double>(decomp, rank, opt.halo);
      fill_owned(u, static_cast<double>(rank));
      comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD,
                                                 hop);
      run_timed(u, halo, opt, rank);
    }
  } catch (const std::exception &ex) {
    if (rank == 0) {
      std::cerr << "23_halo_microtiming: " << ex.what() << '\n';
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (rank == 0) {
    std::cout << "halo microtiming wrote " << opt.output << " (nproc=" << nproc
              << " n=" << opt.nx << "x" << opt.ny << "x" << opt.nz
              << " iters=" << opt.iters << ")\n";
  }
  MPI_Finalize();
  return 0;
}
