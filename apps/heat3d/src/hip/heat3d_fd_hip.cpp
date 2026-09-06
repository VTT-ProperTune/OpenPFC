// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_fd_hip.cpp
 * @brief 3D heat equation on HIP: device halo + stencil, one rank per GCD.
 *
 * Same CLI as `heat3d_fd`: `<N> <n_steps> <dt> <fd_order>`. Optional env:
 * `HEAT3D_PROFILE_JSON` writes a schema-v4 `wall_step` profile; `HEAT3D_WARMUP`
 * (default 1) drops that many frames from the profile.
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "heat3d_fd_hip requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <mpi.h>

#include <heat3d/cli.hpp>
#include <heat3d/device_step.hpp>
#include <heat3d/heat_model.hpp>
#include <heat3d/reporting.hpp>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <openpfc/runtime/gpu/fd_gpu_stack.hpp>

namespace {

using HostField = pfc::data::Field<double, pfc::HostSpace>;
using DevField = pfc::data::Field<double, pfc::HIPSpace>;

void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

void copy_host_to_device(const HostField &host, DevField &dev) {
  if (host.size() != dev.size()) {
    throw std::runtime_error("heat3d_fd_hip: host/device size mismatch");
  }
  dev.with_host_view([&](double *data, std::size_t n) {
    std::copy(host.data(), host.data() + n, data);
  });
  dev.sync_to_device();
}

void copy_device_to_host(DevField &dev, HostField &host) {
  if (host.size() != dev.size()) {
    throw std::runtime_error("heat3d_fd_hip: host/device size mismatch");
  }
  dev.with_host_view(
      [&](double *data, std::size_t n) { std::copy(data, data + n, host.data()); });
  dev.note_device_write();
}

int env_int(const char *name, int fallback) {
  const char *v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return fallback;
  }
  return std::atoi(v);
}

int run_heat3d_fd_hip(const heat3d::RunConfig &cfg, int rank, int nproc) {
  pfc::runtime::gpu::bind_local_device(MPI_COMM_WORLD);

  const int hw = cfg.fd_order / 2;
  const auto domain = pfc::domain::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::FDGPUStack<pfc::HIPSpace> stack(domain, hw, rank, nproc);
  const auto &decomp = stack.decomposition();

  auto u_h = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  u_h.apply([](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / (4.0 * heat3d::kD));
  });

  DevField &u = stack.u();
  DevField du = stack.make_field();
  copy_host_to_device(u_h, u);

  auto halo = stack.make_exchange({&u}, {});
  auto grad = stack.gradient<heat3d::HeatGrads>(cfg.fd_order);
  const auto owned = u.local_size();
  const int nx = owned[0];
  const int ny = owned[1];
  const int nz = owned[2];

  if (rank == 0) {
    std::cout << "HEAT3D_HIP_HALO_MODE=device"
              << " gpu_aware=" << (halo.uses_gpu_aware_mpi() ? 1 : 0)
              << " contiguous=" << (halo.uses_contiguous_device_mpi() ? 1 : 0)
              << " N=" << cfg.N << " ranks=" << nproc << " fd_order=" << cfg.fd_order
              << "\n";
  }

  const char *profile_path = std::getenv("HEAT3D_PROFILE_JSON");
  const int warmup = env_int("HEAT3D_WARMUP", 1);
  std::unique_ptr<pfc::profiling::ProfilingSession> prof;
  if (profile_path != nullptr && *profile_path != '\0') {
    using pfc::profiling::ProfilingMetricCatalog;
    using pfc::profiling::ProfilingSession;
    prof = std::make_unique<ProfilingSession>(
        ProfilingMetricCatalog::with_defaults_and_extras({}),
        ProfilingSession::openpfc_default_frame_metrics());
  }
  pfc::profiling::ProfilingContextScope prof_ctx(prof.get());

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  double t = 0.0;
  for (int step = 0; step < cfg.n_steps; ++step) {
    const bool record = prof && step >= warmup;
    if (record) {
      pfc::profiling::openpfc_begin_frame_with_step_and_rank(*prof, step, rank);
      hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
    }
    const double wall = pfc::profiling::measure_barriered(MPI_COMM_WORLD, [&] {
      halo.exchange();
      heat3d::fd_rhs_hip(grad, du.data(), t, nx, ny, nz);
      du.note_device_write();
      heat3d::euler_axpy_hip(u.data(), du.data(), cfg.dt, u.size());
      u.note_device_write();
      hip_check(hipDeviceSynchronize(), "heat3d step sync");
    });
    if (record) {
      pfc::profiling::openpfc_end_frame_step_wall_and_memory(*prof, wall, 0, 0, 0);
    }
    t += cfg.dt;
  }
  const double local_elapsed = MPI_Wtime() - t_start;
  double max_elapsed = 0.0;
  MPI_Allreduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  if (prof) {
    pfc::profiling::ProfilingExportOptions exp;
    exp.write_json = true;
    exp.json_path = profile_path;
    prof->finalize_and_export(MPI_COMM_WORLD, exp);
    if (rank == 0) {
      std::cout << "HEAT3D_PROFILE wrote " << profile_path << " warmup=" << warmup
                << " steps=" << cfg.n_steps << "\n";
    }
  }

  copy_device_to_host(u, u_h);

  double sum = 0.0;
  double sumsq = 0.0;
  u_h.for_each_owned([&](int i, int j, int k) {
    const double v = u_h(i, j, k);
    sum += v;
    sumsq += v * v;
  });
  double g_sum = 0.0;
  double g_sumsq = 0.0;
  MPI_Allreduce(&sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&sumsq, &g_sumsq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << std::setprecision(17);
    std::cout << "HEAT3D_HIP_CHECKSUM sum_u=" << g_sum << " sumsq_u=" << g_sumsq
              << " l2=" << std::sqrt(g_sumsq) << "\n";
    std::cout << "HEAT3D_HIP_CHECKSUM_HEX sum_u=" << std::hexfloat << g_sum
              << std::defaultfloat << " sumsq_u=" << std::hexfloat << g_sumsq
              << "\n";
  }

  heat3d::report(rank, nproc, cfg, "fd_hip", heat3d::fd_extra_metadata(cfg),
                 max_elapsed, "(periodic; interior L2)",
                 [&u_h, hw](auto &&cb) {
                   const auto sz = u_h.local_size();
                   for (int k = hw; k < sz[2] - hw; ++k) {
                     for (int j = hw; j < sz[1] - hw; ++j) {
                       for (int i = hw; i < sz[0] - hw; ++i) {
                         const auto p = u_h.coords(i, j, k);
                         cb(p[0], p[1], p[2], u_h(i, j, k));
                       }
                     }
                   }
                 });
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[]) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg =
            heat3d::parse_fd_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) {
          return EXIT_FAILURE;
        }
        return run_heat3d_fd_hip(*cfg, rank, nproc);
      });
}
