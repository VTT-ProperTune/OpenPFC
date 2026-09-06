// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_spectral_hip.cpp
 * @brief 3D heat equation on HIP: implicit Euler in Fourier space.
 *
 * HIP twin of `heat3d_spectral` (2 FFTs/step), not the 4-FFT point-wise
 * path. Same CLI: `<N> <n_steps> <dt>`. Optional env:
 * `HEAT3D_PROFILE_JSON` writes a schema-v4 `wall_step` profile;
 * `HEAT3D_WARMUP` (default 1) drops that many frames from the profile.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "heat3d_spectral_hip requires HIP spectral support (rocFFT HeFFTe)"
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
#include <heat3d/heat_model.hpp>
#include <heat3d/reporting.hpp>
#include <heat3d/spectral_heat_propagator_hip.hpp>

#include <openpfc/domain/create.hpp>
#include <openpfc/frontend/ui/from_json_heffte.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>

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
    throw std::runtime_error("heat3d_spectral_hip: host/device size mismatch");
  }
  dev.with_host_view([&](double *data, std::size_t n) {
    std::copy(host.data(), host.data() + n, data);
  });
  dev.sync_to_device();
}

void copy_device_to_host(DevField &dev, HostField &host) {
  if (host.size() != dev.size()) {
    throw std::runtime_error("heat3d_spectral_hip: host/device size mismatch");
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

int run_heat3d_spectral_hip(const heat3d::RunConfig &cfg, int rank, int nproc) {
  pfc::runtime::gpu::bind_local_device(MPI_COMM_WORLD);

  const auto domain = pfc::domain::create(pfc::GridSize({cfg.N, cfg.N, cfg.N}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto opts = pfc::sim::stacks::gpu_fft_for<pfc::HIPSpace>::default_plan_options();
  // Same overlay as docs/lumi_slurm/tungsten_hip_scaling.toml: 1D slabs even
  // on one node. Default rocFFT pencils made 2/4/8 GCD slower than 1 GCD.
  opts.use_pencils = false;
  opts.use_gpu_aware = true;
  opts.algorithm = heffte::reshape_algorithm::p2p_plined;
  pfc::ui::apply_heffte_comm_scale(opts, nproc);
  pfc::sim::stacks::HIPSpectralStack stack(domain, rank, nproc, MPI_COMM_WORLD,
                                           opts);

  heat3d::HeatModel model;
  auto u_h = pfc::data::field_from_inbox<double>(stack.domain(),
                                                 stack.fft().get_inbox_bounds());
  u_h.apply(model.initial_condition);
  copy_host_to_device(u_h, stack.u());

  heat3d::SpectralHeatPropagatorHIP prop(stack.fft(), stack.u(), heat3d::kD, cfg.dt);

  if (rank == 0) {
    std::cout << "HEAT3D_SPECTRAL_HIP"
              << " N=" << cfg.N << " ranks=" << nproc
              << " inbox=" << stack.fft().size_inbox()
              << " outbox=" << stack.fft().size_outbox()
              << " use_pencils=" << (opts.use_pencils ? 1 : 0)
              << " gpu_aware=" << (opts.use_gpu_aware ? 1 : 0) << "\n";
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
  for (int step = 0; step < cfg.n_steps; ++step) {
    const bool record = prof && step >= warmup;
    if (record) {
      pfc::profiling::openpfc_begin_frame_with_step_and_rank(*prof, step, rank);
      hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
    }
    const double wall = pfc::profiling::measure_barriered(MPI_COMM_WORLD, [&] {
      prop.step(stack.u());
      hip_check(hipDeviceSynchronize(), "heat3d spectral step sync");
    });
    if (record) {
      pfc::profiling::openpfc_end_frame_step_wall_and_memory(*prof, wall, 0, 0, 0);
    }
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

  copy_device_to_host(stack.u(), u_h);

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
    std::cout << "HEAT3D_SPECTRAL_HIP_CHECKSUM sum_u=" << g_sum
              << " sumsq_u=" << g_sumsq << " l2=" << std::sqrt(g_sumsq) << "\n";
    std::cout << "HEAT3D_SPECTRAL_HIP_CHECKSUM_HEX sum_u=" << std::hexfloat << g_sum
              << std::defaultfloat << " sumsq_u=" << std::hexfloat << g_sumsq
              << "\n";
  }

  heat3d::report(rank, nproc, cfg, "spectral_hip", "", max_elapsed,
                 "(periodic spectral vs infinite-domain reference)",
                 [&u_h](auto &&cb) { u_h.for_each_owned(cb); });
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[]) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg =
            heat3d::parse_spectral_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) {
          return EXIT_FAILURE;
        }
        return run_heat3d_spectral_hip(*cfg, rank, nproc);
      });
}
