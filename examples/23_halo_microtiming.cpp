// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 23_halo_microtiming.cpp
 * @brief Timed `HaloExchange` loop that exports schema-v2 profiling JSON.
 *
 * Host Faces is the default. `--cuda` / `--hip` select the device facade.
 * Each timed iteration is one `wall_step` (barrier, exchange, barrier).
 *
 * Capture on tohtori:
 *   mpirun -n 2 ./examples/23_halo_microtiming --nx 128 --iters 50 \\
 *     --output results/halo/host_2rank.json
 *   mpirun -n 2 ./examples/23_halo_microtiming --cuda --nx 128 --iters 50 \\
 *     --output results/halo/cuda_2rank.json
 */

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>

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
  int halo = 1;
  int iters = 50;
  int warmup = 5;
  std::string output;
};

[[noreturn]] void usage(int code) {
  std::cerr << "Usage: 23_halo_microtiming [--cuda|--hip] [--full] [--nx N] "
               "[--halo N] [--iters N] [--warmup N] --output PATH.json\n";
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
  if (opt.output.empty() || opt.nx <= 0 || opt.halo < 0 || opt.iters <= 0 ||
      opt.warmup < 0) {
    usage(2);
  }
  if (opt.use_cuda && opt.use_hip) {
    std::cerr << "23_halo_microtiming: choose at most one of --cuda / --hip\n";
    std::exit(2);
  }
  return opt;
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
    hipSetDevice(local % n);
  }
#endif
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
      bind_local_gpu();
    }
    auto domain = domain::create({opt.nx, opt.nx, opt.nx});
    auto decomp = decomposition::create(domain, nproc);
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
              << " nx=" << opt.nx << " iters=" << opt.iters << ")\n";
  }
  MPI_Finalize();
  return 0;
}
