// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_fd_hip.cpp
 * @brief 3D heat equation on HIP: device halo + stencil, one rank per GCD.
 *
 * CLI: `<N> <n_steps> <dt> <fd_order>` or
 * `<Nx> <Ny> <Nz> <n_steps> <dt> <fd_order>`. Optional env:
 * `HEAT3D_PROFILE_JSON` writes a schema-v4 `wall_step` profile; `HEAT3D_WARMUP`
 * (default 1) drops that many frames from the profile.
 * `OPENPFC_FD_PROC_GRID=gx,gy,gz` forces the Cartesian split.
 * `HEAT3D_REQUIRE_INTERIOR=nx,ny,nz` fails closed unless every rank's owned
 * interior matches. `HEAT3D_DIAG_TIMING=1` adds HIP-event / blocking-halo
 * attribution and must not replace the clean barriered `wall_step`.
 * `HEAT3D_HALO_OVERLAP` selects the device timestep via
 * `pfc::comm::GpuHaloOverlap`: `1` (default on more than one rank) is
 * Waitall (interior on a non-blocking compute stream, Faces pack/MPI on
 * the default stream, then `finish()` + boundary); `0` is blocking
 * `exchange()`; `2` is Testall (`MPI_Testall` until the interior event
 * completes). A single rank has no MPI halo, so the default falls back
 * to blocking; an explicit `1` or `2` still runs the split.
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "heat3d_fd_hip requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>
#include <openpfc/runtime/gpu/check_error.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include <mpi.h>

#include <heat3d/cli.hpp>
#include <heat3d/device_step.hpp>
#include <heat3d/heat_model.hpp>
#include <heat3d/reporting.hpp>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/halo_overlap.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/profiling/profiling.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <openpfc/runtime/gpu/fd_gpu_stack.hpp>
#include <openpfc/runtime/gpu/halo_overlap_gpu.hpp>

namespace {

using HostField = pfc::data::Field<double, pfc::HostSpace>;
using DevField = pfc::data::Field<double, pfc::HIPSpace>;

using pfc::runtime::gpu::hip_check;

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

// Two-stream Faces overlap is the production default after the
// standard-g 1/2/4-node Heat3D A/B (issue #48). `0` remains the
// blocking control.
constexpr int kDefaultHaloOverlap = 1;

int halo_overlap_mode(int nproc) {
  const char *v = std::getenv("HEAT3D_HALO_OVERLAP");
  const bool explicit_set = v != nullptr && v[0] != '\0';
  const int overlap = explicit_set ? std::atoi(v) : kDefaultHaloOverlap;
  if (overlap < 0 || overlap > 2) {
    throw std::runtime_error(
        "heat3d_fd_hip: HEAT3D_HALO_OVERLAP must be 0, 1, or 2");
  }
  // Single-rank has no MPI halo. Default two-stream was 8% slower on
  // 512³ 1 GCD (jobs 22162498 / 22162499). Honor an explicit 1 or 2 so
  // the inner/border split stays testable.
  if (!explicit_set && nproc == 1) {
    return 0;
  }
  return overlap;
}

bool env_flag(const char *name) {
  const char *v = std::getenv(name);
  return v != nullptr && v[0] == '1' && v[1] == '\0';
}

std::array<int, 3> parse_int3_env(const char *name) {
  const char *e = std::getenv(name);
  if (e == nullptr || e[0] == '\0') {
    return {0, 0, 0};
  }
  std::array<int, 3> g{0, 0, 0};
  const char *p = e;
  for (int i = 0; i < 3; ++i) {
    char *end = nullptr;
    const long v = std::strtol(p, &end, 10);
    if (end == p || v < 1) {
      return {0, 0, 0};
    }
    g[static_cast<std::size_t>(i)] = static_cast<int>(v);
    if (i < 2) {
      if (*end != ',' && *end != 'x' && *end != 'X') {
        return {0, 0, 0};
      }
      p = end + 1;
    }
  }
  return g;
}

double median_of(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  if (n % 2 == 1) {
    return v[n / 2];
  }
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct HipEvent {
  hipEvent_t e{};
  HipEvent() { hip_check(hipEventCreate(&e), "hipEventCreate"); }
  ~HipEvent() { hipEventDestroy(e); }
  HipEvent(const HipEvent &) = delete;
  HipEvent &operator=(const HipEvent &) = delete;
};

int run_heat3d_fd_hip(const heat3d::RunConfig &cfg, int rank, int nproc) {
  pfc::runtime::gpu::bind_local_device(MPI_COMM_WORLD);
  const int overlap = halo_overlap_mode(nproc);

  const int hw = cfg.fd_order / 2;
  const auto domain = pfc::domain::create(pfc::GridSize({cfg.Nx, cfg.Ny, cfg.Nz}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::FDGPUStack<pfc::HIPSpace> stack(domain, hw, rank, nproc);
  const auto &decomp = stack.decomposition();
  const auto grid = pfc::decomposition::get_grid(decomp);
  const auto owned_box = pfc::decomposition::local_box(decomp, rank);
  const int lx = owned_box.size[0];
  const int ly = owned_box.size[1];
  const int lz = owned_box.size[2];

  int loc_min[3] = {lx, ly, lz};
  int loc_max[3] = {lx, ly, lz};
  MPI_Allreduce(MPI_IN_PLACE, loc_min, 3, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, loc_max, 3, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  auto u_h = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  u_h.apply([](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / (4.0 * heat3d::kD));
  });

  DevField &u = stack.u();
  DevField du = stack.make_field();
  copy_host_to_device(u_h, u);

  auto halo = stack.make_exchange({&u}, {});
  auto grad = stack.gradient<heat3d::HeatGrads>(cfg.fd_order);
  const auto padded = u.local_size();
  const int nx = padded[0];
  const int ny = padded[1];
  const int nz = padded[2];

  int gpu = -1;
  hip_check(hipGetDevice(&gpu), "hipGetDevice");
  char host[256];
  if (gethostname(host, sizeof(host)) != 0) {
    std::snprintf(host, sizeof(host), "unknown");
  }
  host[sizeof(host) - 1] = '\0';

  const int gx = grid[0];
  const int gy = grid[1];
  const int rx = rank % gx;
  const int ry = (rank / gx) % gy;
  const int rz = rank / (gx * gy);
  // Allgather hostnames, then count off-node faces locally. Sequential
  // MPI_Sendrecv along +z on a 4-wide periodic ring deadlocks (the 1-node
  // 2x2x2 job is an involution on every axis, so it did not hang).
  std::vector<char> hosts(static_cast<std::size_t>(nproc) * 256);
  MPI_Allgather(host, 256, MPI_CHAR, hosts.data(), 256, MPI_CHAR, MPI_COMM_WORLD);
  int offnode = 0;
  const std::array<std::array<int, 3>, 6> faces = {{{{1, 0, 0}},
                                                    {{-1, 0, 0}},
                                                    {{0, 1, 0}},
                                                    {{0, -1, 0}},
                                                    {{0, 0, 1}},
                                                    {{0, 0, -1}}}};
  for (const auto &dir : faces) {
    const int peer = pfc::decomposition::get_neighbor_rank(decomp, rank,
                                                           {dir[0], dir[1], dir[2]});
    if (peer < 0 || peer == rank) {
      continue;
    }
    if (std::strncmp(host, &hosts[static_cast<std::size_t>(peer) * 256], 256) != 0) {
      ++offnode;
    }
  }
  std::vector<int> gpus(static_cast<std::size_t>(nproc));
  std::vector<int> coords(static_cast<std::size_t>(nproc) * 4);
  const int local_meta[4] = {rx, ry, rz, offnode};
  MPI_Gather(&gpu, 1, MPI_INT, gpus.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(local_meta, 4, MPI_INT, coords.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "HEAT3D_FD_DECOMP proc_grid=" << gx << "x" << gy << "x" << grid[2]
              << " global=" << cfg.Nx << "x" << cfg.Ny << "x" << cfg.Nz
              << " local_min=" << loc_min[0] << "x" << loc_min[1] << "x"
              << loc_min[2] << " local_max=" << loc_max[0] << "x" << loc_max[1]
              << "x" << loc_max[2] << " halo=" << hw
              << " gpu_aware=" << (halo.uses_gpu_aware_mpi() ? 1 : 0)
              << " contiguous=" << (halo.uses_contiguous_device_mpi() ? 1 : 0)
              << " ranks=" << nproc << " fd_order=" << cfg.fd_order
              << " halo_overlap=" << overlap << std::endl;
    std::ofstream plc("fd_placement.txt");
    plc << "rank host gpu rx ry rz offnode_faces\n";
    for (int r = 0; r < nproc; ++r) {
      plc << r << " " << &hosts[static_cast<std::size_t>(r) * 256] << " " << gpus[r]
          << " " << coords[static_cast<std::size_t>(r) * 4] << " "
          << coords[static_cast<std::size_t>(r) * 4 + 1] << " "
          << coords[static_cast<std::size_t>(r) * 4 + 2] << " "
          << coords[static_cast<std::size_t>(r) * 4 + 3] << "\n";
    }
  }

  const auto require = parse_int3_env("HEAT3D_REQUIRE_INTERIOR");
  if (require[0] > 0) {
    const bool ok = loc_min[0] == require[0] && loc_min[1] == require[1] &&
                    loc_min[2] == require[2] && loc_max[0] == require[0] &&
                    loc_max[1] == require[1] && loc_max[2] == require[2];
    if (!ok) {
      if (rank == 0) {
        std::cerr << "heat3d_fd_hip: owned interior " << loc_min[0] << "x"
                  << loc_min[1] << "x" << loc_min[2] << ".." << loc_max[0] << "x"
                  << loc_max[1] << "x" << loc_max[2] << " != required " << require[0]
                  << "x" << require[1] << "x" << require[2] << "\n";
      }
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }

  const char *profile_path = std::getenv("HEAT3D_PROFILE_JSON");
  const int warmup = env_int("HEAT3D_WARMUP", 1);
  const bool diag = env_flag("HEAT3D_DIAG_TIMING");
  std::unique_ptr<pfc::profiling::ProfilingSession> prof;
  if (profile_path != nullptr && *profile_path != '\0') {
    using pfc::profiling::ProfilingMetricCatalog;
    using pfc::profiling::ProfilingSession;
    prof = std::make_unique<ProfilingSession>(
        ProfilingMetricCatalog::with_defaults_and_extras({}),
        ProfilingSession::openpfc_default_frame_metrics());
  }
  pfc::profiling::ProfilingContextScope prof_ctx(prof.get());

  std::unique_ptr<HipEvent> rhs0, rhs1, upd0, upd1, inner0, inner1, bord0, bord1;
  std::vector<double> t_halo, t_rhs, t_upd, t_post, t_wait, t_inner, t_border;
  if (diag) {
    rhs0 = std::make_unique<HipEvent>();
    rhs1 = std::make_unique<HipEvent>();
    upd0 = std::make_unique<HipEvent>();
    upd1 = std::make_unique<HipEvent>();
    if (overlap != 0) {
      inner0 = std::make_unique<HipEvent>();
      inner1 = std::make_unique<HipEvent>();
      bord0 = std::make_unique<HipEvent>();
      bord1 = std::make_unique<HipEvent>();
    }
  }
  std::unique_ptr<pfc::comm::GpuHaloOverlap> overlap_helper;
  if (overlap != 0) {
    overlap_helper = std::make_unique<pfc::comm::GpuHaloOverlap>(
        pfc::comm::halo_overlap_mode_from_int(overlap));
  }

  double t = 0.0;
  auto overlap_rhs = [&] {
    overlap_helper->rhs(
        halo,
        [&](pfc::gpuStream_t stream) {
          heat3d::fd_rhs_inner_hip(grad, du.data(), t, nx, ny, nz, hw,
                                   /*sync=*/false, stream);
        },
        [&] {
          heat3d::fd_rhs_border_hip(grad, du.data(), t, nx, ny, nz, hw,
                                    /*sync=*/true);
          du.note_device_write();
        });
  };

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  for (int step = 0; step < cfg.n_steps; ++step) {
    const bool record = prof && step >= warmup;
    if (record) {
      pfc::profiling::openpfc_begin_frame_with_step_and_rank(*prof, step, rank);
      hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
    }
    const double wall = pfc::profiling::measure_barriered(MPI_COMM_WORLD, [&] {
      if (diag && overlap == 0) {
        hip_check(hipDeviceSynchronize(), "diag pre-halo sync");
        const double h0 = MPI_Wtime();
        halo.exchange();
        hip_check(hipDeviceSynchronize(), "diag post-halo sync");
        const double halo_s = MPI_Wtime() - h0;
        hip_check(hipEventRecord(rhs0->e, nullptr), "rhs0");
        heat3d::fd_rhs_hip(grad, du.data(), t, nx, ny, nz);
        du.note_device_write();
        hip_check(hipEventRecord(rhs1->e, nullptr), "rhs1");
        hip_check(hipEventSynchronize(rhs1->e), "rhs sync");
        float rhs_ms = 0.0f;
        hip_check(hipEventElapsedTime(&rhs_ms, rhs0->e, rhs1->e), "rhs elapsed");
        hip_check(hipEventRecord(upd0->e, nullptr), "upd0");
        heat3d::euler_axpy_hip(u.data(), du.data(), cfg.dt, u.size());
        u.note_device_write();
        hip_check(hipEventRecord(upd1->e, nullptr), "upd1");
        hip_check(hipEventSynchronize(upd1->e), "upd sync");
        float upd_ms = 0.0f;
        hip_check(hipEventElapsedTime(&upd_ms, upd0->e, upd1->e), "upd elapsed");
        if (step >= warmup) {
          t_halo.push_back(halo_s);
          t_rhs.push_back(static_cast<double>(rhs_ms) * 1.0e-3);
          t_upd.push_back(static_cast<double>(upd_ms) * 1.0e-3);
        }
      } else if (diag && overlap != 0) {
        const pfc::gpuStream_t compute = overlap_helper->compute_stream();
        hip_check(hipDeviceSynchronize(), "diag pre-overlap sync");
        hip_check(hipEventRecord(inner0->e, compute), "inner0");
        heat3d::fd_rhs_inner_hip(grad, du.data(), t, nx, ny, nz, hw, false, compute);
        hip_check(hipEventRecord(inner1->e, compute), "inner1");
        if (overlap == 2) {
          overlap_helper->record_inner_done();
        }
        const double p0 = MPI_Wtime();
        halo.start();
        const double post_s = MPI_Wtime() - p0;
        if (overlap == 2) {
          overlap_helper->pump(halo);
        }
        const double w0 = MPI_Wtime();
        halo.finish();
        const double wait_s = MPI_Wtime() - w0;
        hip_check(hipEventRecord(bord0->e, nullptr), "bord0");
        heat3d::fd_rhs_border_hip(grad, du.data(), t, nx, ny, nz, hw, true);
        du.note_device_write();
        hip_check(hipEventRecord(bord1->e, nullptr), "bord1");
        hip_check(hipEventSynchronize(inner1->e), "inner sync");
        hip_check(hipEventSynchronize(bord1->e), "border sync");
        float inner_ms = 0.0f;
        float bord_ms = 0.0f;
        hip_check(hipEventElapsedTime(&inner_ms, inner0->e, inner1->e),
                  "inner elapsed");
        hip_check(hipEventElapsedTime(&bord_ms, bord0->e, bord1->e),
                  "border elapsed");
        hip_check(hipEventRecord(upd0->e, nullptr), "upd0");
        heat3d::euler_axpy_hip(u.data(), du.data(), cfg.dt, u.size());
        u.note_device_write();
        hip_check(hipEventRecord(upd1->e, nullptr), "upd1");
        hip_check(hipEventSynchronize(upd1->e), "upd sync");
        float upd_ms = 0.0f;
        hip_check(hipEventElapsedTime(&upd_ms, upd0->e, upd1->e), "upd elapsed");
        if (step >= warmup) {
          t_post.push_back(post_s);
          t_wait.push_back(wait_s);
          t_inner.push_back(static_cast<double>(inner_ms) * 1.0e-3);
          t_border.push_back(static_cast<double>(bord_ms) * 1.0e-3);
          t_halo.push_back(post_s + wait_s);
          t_rhs.push_back(static_cast<double>(inner_ms + bord_ms) * 1.0e-3);
          t_upd.push_back(static_cast<double>(upd_ms) * 1.0e-3);
        }
      } else if (overlap != 0) {
        overlap_rhs();
        heat3d::euler_axpy_hip(u.data(), du.data(), cfg.dt, u.size());
        u.note_device_write();
        hip_check(hipDeviceSynchronize(), "heat3d step sync");
      } else {
        halo.exchange();
        heat3d::fd_rhs_hip(grad, du.data(), t, nx, ny, nz);
        du.note_device_write();
        heat3d::euler_axpy_hip(u.data(), du.data(), cfg.dt, u.size());
        u.note_device_write();
        hip_check(hipDeviceSynchronize(), "heat3d step sync");
      }
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

  if (diag) {
    const double local[3] = {median_of(t_halo), median_of(t_rhs), median_of(t_upd)};
    double gmax[3] = {0.0, 0.0, 0.0};
    double gmin[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local, gmax, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local, gmin, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
      // Headline is the max across rank-local medians so it is comparable
      // to the barriered clean wall_step (slowest rank). min/max is spread.
      std::cout << "HEAT3D_DIAG reduce=max_rank_median halo_s=" << gmax[0]
                << " rhs_s=" << gmax[1] << " update_s=" << gmax[2]
                << " halo_minmax=" << gmin[0] << "," << gmax[0]
                << " rhs_minmax=" << gmin[1] << "," << gmax[1]
                << " update_minmax=" << gmin[2] << "," << gmax[2] << std::endl;
    }
    if (overlap != 0 && !t_wait.empty()) {
      const double ol[4] = {median_of(t_post), median_of(t_wait), median_of(t_inner),
                            median_of(t_border)};
      double omax[4] = {0.0, 0.0, 0.0, 0.0};
      MPI_Allreduce(ol, omax, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (rank == 0) {
        std::cout << "HEAT3D_OVERLAP reduce=max_rank_median post_s=" << omax[0]
                  << " exposed_wait_s=" << omax[1] << " inner_s=" << omax[2]
                  << " border_s=" << omax[3] << " mode=" << overlap << std::endl;
      }
    }
  }

  if (prof) {
    pfc::profiling::ProfilingExportOptions exp;
    exp.write_json = true;
    exp.json_path = profile_path;
    prof->finalize_and_export(MPI_COMM_WORLD, exp);
    if (rank == 0) {
      std::cout << "HEAT3D_PROFILE wrote " << profile_path << " warmup=" << warmup
                << " steps=" << cfg.n_steps << " diag=" << (diag ? 1 : 0) << "\n";
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
                 max_elapsed, "(periodic; interior L2)", [&u_h, hw](auto &&cb) {
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
        const auto cfg = heat3d::parse_fd_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) {
          return EXIT_FAILURE;
        }
        return run_heat3d_fd_hip(*cfg, rank, nproc);
      });
}
