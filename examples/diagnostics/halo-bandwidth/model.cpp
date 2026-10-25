// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file model.cpp
 * @brief Halo-only throughput of `pfc::comm::HaloExchange`. No physics.
 *
 * Prints an OSU-style table: face bytes, time, and GB/s (1e9 bytes/s).
 * The default exchanges only the ±X faces; `--axis` picks another pair and
 * `--faces` all six. Only faces between different ranks count as sent bytes.
 * The automatic two-rank grid splits z, so a ±X run there sends nothing;
 * read the printed grid, or select it with OPENPFC_FD_PROC_GRID.
 *
 *   mpirun -n 2 ./examples/25_halo_bandwidth --axis z --sizes 128,256,512
 *   mpirun -n 2 ./examples/25_halo_bandwidth --cuda --axis z --sizes 512,1024
 *   OPENPFC_FD_PROC_GRID=2,1,1 \
 *     mpirun -n 2 ./examples/25_halo_bandwidth --sizes 128,256,512
 *   OPENPFC_FD_PROC_GRID=8,1,1 \
 *     mpirun -n 8 ./examples/25_halo_bandwidth --hip --faces
 *
 * A CPU build of this file can be packaged with `openpfc bundle`. CUDA and
 * ROCm images are not qualified by that command.
 */

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
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

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_aware_mpi.hpp>
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
  bool faces = false;
  int axis = 0;
  int halo = 1;
  int iters = 50;
  int warmup = 5;
  std::vector<int> sizes{64, 128, 256, 512, 768, 1024, 1536, 2048};
};

[[noreturn]] void usage(int code) {
  std::cerr << "Usage: 25_halo_bandwidth [--cuda|--hip] [--faces] [--axis x|y|z] "
               "[--halo N] [--iters N] [--warmup N] [--sizes 64,128,256]\n"
            << "  Default directions are ±X. --axis picks one pair. --faces uses "
               "±X, ±Y and ±Z.\n"
            << "  Ny and Nz equal Nx. OPENPFC_FD_PROC_GRID=gx,gy,gz selects the "
               "split.\n"
            << "  Faces on an undivided axis wrap to the same rank and send no "
               "bytes.\n"
            << "  A z split keeps the exchanged face contiguous in the x-fastest "
               "field.\n"
            << "  Uni_GBps is remote send bytes / time / 1e9. Duplex_GBps is twice "
               "that.\n";
  std::exit(code);
}

int need_int(int argc, char **argv, int &i) {
  if (i + 1 >= argc) {
    usage(2);
  }
  return std::atoi(argv[++i]);
}

std::vector<int> parse_sizes(std::string_view text) {
  std::vector<int> sizes;
  std::string token;
  auto push = [&] {
    if (token.empty()) {
      return;
    }
    const int value = std::atoi(token.c_str());
    if (value <= 0) {
      usage(2);
    }
    sizes.push_back(value);
    token.clear();
  };
  for (const char c : text) {
    if (c == ',') {
      push();
    } else {
      token.push_back(c);
    }
  }
  push();
  if (sizes.empty()) {
    usage(2);
  }
  return sizes;
}

Options parse_args(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a{argv[i]};
    if (a == "--help" || a == "-h") {
      usage(0);
    } else if (a == "--cuda") {
      opt.use_cuda = true;
    } else if (a == "--hip") {
      opt.use_hip = true;
    } else if (a == "--faces") {
      opt.faces = true;
    } else if (a == "--axis") {
      if (i + 1 >= argc) {
        usage(2);
      }
      const std::string_view axis{argv[++i]};
      if (axis == "x") {
        opt.axis = 0;
      } else if (axis == "y") {
        opt.axis = 1;
      } else if (axis == "z") {
        opt.axis = 2;
      } else {
        usage(2);
      }
    } else if (a == "--halo") {
      opt.halo = need_int(argc, argv, i);
    } else if (a == "--iters") {
      opt.iters = need_int(argc, argv, i);
    } else if (a == "--warmup") {
      opt.warmup = need_int(argc, argv, i);
    } else if (a == "--sizes") {
      if (i + 1 >= argc) {
        usage(2);
      }
      opt.sizes = parse_sizes(argv[++i]);
    } else {
      usage(2);
    }
  }
  if (opt.halo <= 0 || opt.iters <= 0 || opt.warmup < 0 ||
      (opt.use_cuda && opt.use_hip)) {
    usage(2);
  }
  return opt;
}

halo::HaloDirectionSet directions_for(bool faces, int axis) {
  using Dir = halo::HaloDirectionSet::Int3;
  if (faces) {
    return halo::presets::Axes3D();
  }
  Dir positive{0, 0, 0};
  Dir negative{0, 0, 0};
  positive[axis] = 1;
  negative[axis] = -1;
  return halo::HaloDirectionSet(std::vector<Dir>{positive, negative});
}

const char *axis_name(bool faces, int axis) {
  if (faces) {
    return "faces";
  }
  if (axis == 1) {
    return "y";
  }
  if (axis == 2) {
    return "z";
  }
  return "x";
}

void bind_local_gpu() {
  MPI_Comm node = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int local = 0;
  MPI_Comm_rank(node, &local);
  MPI_Comm_free(&node);
#if defined(OpenPFC_ENABLE_CUDA)
  int n = 0;
  if (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) {
    cudaSetDevice(local % n);
  }
#endif
#if defined(OpenPFC_ENABLE_HIP)
  int n = 0;
  if (hipGetDeviceCount(&n) == hipSuccess && n > 0) {
    const hipError_t err = hipSetDevice(local % n);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("hipSetDevice: ") +
                               hipGetErrorString(err));
    }
  }
#endif
}

struct Traffic {
  unsigned long long remote_bytes = 0;
  unsigned long long self_bytes = 0;
  int remote_faces = 0;
  int self_faces = 0;
  int offnode_faces = 0;
};

// start snippet bytes
unsigned long long face_bytes(int axis, const int *n, int halo) {
  unsigned long long elems = static_cast<unsigned long long>(halo);
  for (int a = 0; a < 3; ++a) {
    if (a != axis) {
      elems *= static_cast<unsigned long long>(n[a]);
    }
  }
  return elems * sizeof(double);
}
// end snippet bytes

Traffic count_traffic(const decomposition::Decomposition &decomp, int rank, int halo,
                      const halo::HaloDirectionSet &dirs,
                      const std::vector<std::string> &hosts) {
  const auto box = decomposition::local_box(decomp, rank);
  const int n[3] = {box.size[0], box.size[1], box.size[2]};
  const std::array<std::array<int, 3>, 6> axes = {{{{1, 0, 0}},
                                                   {{-1, 0, 0}},
                                                   {{0, 1, 0}},
                                                   {{0, -1, 0}},
                                                   {{0, 0, 1}},
                                                   {{0, 0, -1}}}};
  Traffic traffic;
  for (const auto &dir : axes) {
    const halo::HaloDirectionSet::Int3 key{dir[0], dir[1], dir[2]};
    if (!dirs.contains(key)) {
      continue;
    }
    // start snippet peers
    const int peer = decomposition::get_neighbor_rank(decomp, rank, key);
    if (peer < 0) {
      continue;
    }
    const int axis = dir[0] != 0 ? 0 : (dir[1] != 0 ? 1 : 2);
    const unsigned long long bytes = face_bytes(axis, n, halo);
    if (peer == rank) {
      traffic.self_bytes += bytes;
      ++traffic.self_faces;
      continue;
    }
    traffic.remote_bytes += bytes;
    ++traffic.remote_faces;
    // end snippet peers
    if (hosts[static_cast<std::size_t>(peer)] !=
        hosts[static_cast<std::size_t>(rank)]) {
      ++traffic.offnode_faces;
    }
  }
  return traffic;
}

bool fits(const decomposition::Decomposition &decomp, int rank, int halo, bool gpu) {
  const auto box = decomposition::local_box(decomp, rank);
  const long long padded_x = static_cast<long long>(box.size[0]) + 2LL * halo;
  const long long padded_y = static_cast<long long>(box.size[1]) + 2LL * halo;
  const long long padded_z = static_cast<long long>(box.size[2]) + 2LL * halo;
  const long long bytes =
      padded_x * padded_y * padded_z * static_cast<long long>(sizeof(double));
  if (!gpu) {
    return bytes < (48LL << 30);
  }
#if defined(OpenPFC_ENABLE_CUDA)
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
    return static_cast<unsigned long long>(bytes) + (1ULL << 30) < free_bytes;
  }
#endif
#if defined(OpenPFC_ENABLE_HIP)
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
    return static_cast<unsigned long long>(bytes) + (1ULL << 30) < free_bytes;
  }
#endif
  return bytes < (40LL << 30);
}

template <typename Space>
void time_exchange(data::Field<double, Space> &field,
                   comm::HaloExchange<Space, double> &halo, const Options &opt,
                   double &seconds) {
  // start snippet timing
  for (int step = 0; step < opt.warmup; ++step) {
    halo.exchange();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  for (int step = 0; step < opt.iters; ++step) {
    halo.exchange();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  seconds = (MPI_Wtime() - t0) / static_cast<double>(opt.iters);
  // end snippet timing
  (void)field;
}

template <typename Space>
void run_size(const Options &opt, int n, int rank,
              const halo::HaloDirectionSet &dirs,
              const std::vector<std::string> &hosts, bool gpu) {
  auto domain = domain::create({n, n, n});
  auto decomp = decomposition::create(domain, static_cast<int>(hosts.size()));
  int ok = fits(decomp, rank, opt.halo, gpu) ? 1 : 0;
  int all_ok = 0;
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (all_ok == 0) {
    if (rank == 0) {
      std::cout << "# skip n=" << n << " (padded field does not fit)\n";
    }
    return;
  }

  const Traffic local = count_traffic(decomp, rank, opt.halo, dirs, hosts);
  comm::HaloExchangeOptions hop;
  hop.directions = dirs;
  double seconds = 0;
  if constexpr (std::is_same_v<Space, HostSpace>) {
    auto field = data::field_from_subdomain<double>(decomp, rank, opt.halo);
    comm::HaloExchange<HostSpace, double> halo(field, decomp, rank, MPI_COMM_WORLD,
                                               hop);
    time_exchange(field, halo, opt, seconds);
  } else {
    auto field =
        data::Field<double, Space>(decomposition::domain(decomp),
                                   decomposition::local_box(decomp, rank), opt.halo);
    comm::HaloExchange<Space, double> halo(field, decomp, rank, MPI_COMM_WORLD, hop);
    if (rank == 0 && n == opt.sizes.front()) {
      std::cout << "# exchanger gpu_aware " << (halo.uses_gpu_aware_mpi() ? 1 : 0)
                << " contiguous " << (halo.uses_contiguous_device_mpi() ? 1 : 0)
                << '\n';
    }
    time_exchange(field, halo, opt, seconds);
  }

  // start snippet rate
  double slowest = 0;
  MPI_Allreduce(&seconds, &slowest, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  unsigned long long send_min = 0;
  unsigned long long send_max = 0;
  int off_max = 0;
  int remote_faces = 0;
  int self_faces = 0;
  MPI_Allreduce(&local.remote_bytes, &send_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local.remote_bytes, &send_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local.offnode_faces, &off_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local.remote_faces, &remote_faces, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local.self_faces, &self_faces, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rank != 0) {
    return;
  }
  const double uni =
      slowest > 0.0 ? static_cast<double>(send_max) / slowest / 1.0e9 : 0.0;
  // end snippet rate
  std::cout << std::fixed << n << ' ' << send_max << ' ' << std::setprecision(1)
            << (slowest * 1.0e6) << ' ' << std::setprecision(3) << uni << ' '
            << (2.0 * uni) << ' ' << remote_faces << ' ' << self_faces << ' '
            << off_max;
  if (send_min != send_max) {
    std::cout << " send_min=" << send_min;
  }
  std::cout << '\n';
}

std::string device_label(bool cuda, bool hip) {
#if defined(OpenPFC_ENABLE_CUDA)
  if (cuda) {
    cudaDeviceProp prop{};
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
      return std::string(prop.name) + " id=" + std::to_string(dev);
    }
  }
#else
  (void)cuda;
#endif
#if defined(OpenPFC_ENABLE_HIP)
  if (hip) {
    hipDeviceProp_t prop{};
    int dev = 0;
    if (hipGetDevice(&dev) == hipSuccess &&
        hipGetDeviceProperties(&prop, dev) == hipSuccess) {
      return std::string(prop.name) + " id=" + std::to_string(dev);
    }
  }
#else
  (void)hip;
#endif
  return "host";
}

} // namespace

int main(int argc, char **argv) {
  const Options opt = parse_args(argc, argv);
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    return 1;
  }
  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  try {
    const bool gpu = opt.use_cuda || opt.use_hip;
    if (gpu) {
      bind_local_gpu();
    }
    char host[256];
    if (gethostname(host, sizeof(host)) != 0) {
      std::snprintf(host, sizeof(host), "unknown");
    }
    host[sizeof(host) - 1] = '\0';
    std::vector<char> packed(static_cast<std::size_t>(ranks) * 256);
    MPI_Allgather(host, 256, MPI_CHAR, packed.data(), 256, MPI_CHAR, MPI_COMM_WORLD);
    std::vector<std::string> hosts(static_cast<std::size_t>(ranks));
    for (int i = 0; i < ranks; ++i) {
      hosts[static_cast<std::size_t>(i)] =
          &packed[static_cast<std::size_t>(i) * 256];
    }
    int device_id = -1;
#if defined(OpenPFC_ENABLE_CUDA)
    if (opt.use_cuda) {
      cudaGetDevice(&device_id);
    }
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if (opt.use_hip) {
      hipGetDevice(&device_id);
    }
#endif
    std::vector<int> devices(static_cast<std::size_t>(ranks), -1);
    MPI_Gather(&device_id, 1, MPI_INT, devices.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);
    const auto dirs = directions_for(opt.faces, opt.axis);
    auto probe =
        domain::create({opt.sizes.front(), opt.sizes.front(), opt.sizes.front()});
    auto probe_decomp = decomposition::create(probe, ranks);
    const auto grid = decomposition::get_grid(probe_decomp);

    if (rank == 0) {
      int nodes = 0;
      for (int i = 0; i < ranks; ++i) {
        bool seen = false;
        for (int j = 0; j < i; ++j) {
          seen = seen || hosts[static_cast<std::size_t>(j)] ==
                             hosts[static_cast<std::size_t>(i)];
        }
        if (!seen) {
          ++nodes;
        }
      }
      std::string aware = "n/a";
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
      if (gpu) {
        const auto decision = gpu::decide_gpu_aware_mpi();
        aware = std::string(decision.enabled ? "1" : "0") + " " +
                gpu::gpu_aware_how_cstr(decision.how);
      }
#endif
      std::cout << "# OpenPFC halo bandwidth\n"
                << "# backend "
                << (opt.use_cuda ? "cuda" : (opt.use_hip ? "hip" : "host"))
                << " device " << device_label(opt.use_cuda, opt.use_hip) << '\n'
                << "# ranks " << ranks << " nodes " << nodes << " grid " << grid[0]
                << "x" << grid[1] << "x" << grid[2] << " directions "
                << axis_name(opt.faces, opt.axis) << " halo " << opt.halo
                << " iters " << opt.iters << " warmup " << opt.warmup << '\n'
                << "# gpu_aware " << aware << '\n'
                << "# rank_device";
      for (int i = 0; i < ranks; ++i) {
        std::cout << ' ' << i << ':' << devices[static_cast<std::size_t>(i)];
      }
      std::cout << '\n'
                << "# Uni_GBps = remote send bytes / time / 1e9; Duplex_GBps = 2 * "
                   "Uni_GBps\n"
                << "# N Send_B Time_us Uni_GBps Duplex_GBps RemoteFaces SelfFaces "
                   "OffNodeFaces\n";
    }

    for (const int n : opt.sizes) {
      if (opt.use_cuda) {
#if defined(OpenPFC_ENABLE_CUDA)
        run_size<CUDASpace>(opt, n, rank, dirs, hosts, true);
#else
        throw std::runtime_error("25_halo_bandwidth: --cuda requires a CUDA build");
#endif
      } else if (opt.use_hip) {
#if defined(OpenPFC_ENABLE_HIP)
        run_size<HIPSpace>(opt, n, rank, dirs, hosts, true);
#else
        throw std::runtime_error("25_halo_bandwidth: --hip requires a HIP build");
#endif
      } else {
        run_size<HostSpace>(opt, n, rank, dirs, hosts, false);
      }
    }
  } catch (const std::exception &ex) {
    if (rank == 0) {
      std::cerr << "25_halo_bandwidth: " << ex.what() << '\n';
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
