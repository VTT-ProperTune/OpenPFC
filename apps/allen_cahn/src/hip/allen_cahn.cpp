// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_HIP)
#error "allen_cahn_hip requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/comm_sparse_exchange_gpu.hpp>

#include <allen_cahn/common.hpp>
#include <allen_cahn/device_step.hpp>
#include <openpfc/frontend/io/png_writer.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>

namespace {

using DevField = pfc::data::Field<double, pfc::HipSpace>;

void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

void bind_local_gpu() {
  MPI_Comm node_comm = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &node_comm);
  int local_rank = 0;
  if (node_comm != MPI_COMM_NULL) {
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
  }
  int n_dev = 0;
  hip_check(hipGetDeviceCount(&n_dev), "hipGetDeviceCount");
  if (n_dev < 1) {
    throw std::runtime_error("No HIP devices visible to this rank");
  }
  hip_check(hipSetDevice(local_rank % n_dev), "hipSetDevice");
}

} // namespace

int main(int argc, char *argv[]) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        bind_local_gpu();
        const allen_cahn::RunConfig cfg = allen_cahn::parse_args(app_argc, app_argv);
        if (cfg.nx_glob < 4 || cfg.ny_glob < 4 || cfg.n_steps < 1) {
          if (rank == 0) {
            std::cerr << "Need nx, ny >= 4 and n_steps >= 1\n";
          }
          return EXIT_FAILURE;
        }
        auto domain = pfc::domain::create(pfc::GridSize({cfg.nx_glob, cfg.ny_glob, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
        auto decomp = pfc::decomposition::create(domain, nproc);
        const auto &local_box = pfc::decomposition::local_box(decomp, rank);
        auto local_size = local_box.size;
        const int nx = local_size[0];
        const int ny = local_size[1];
        const int nz = local_size[2];
        const std::size_t nlocal = static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny) *
                                   static_cast<std::size_t>(nz);
        const double dx = 1.0;
        const double inv_dx2 = 1.0 / (dx * dx);
        const double inv_dy2 = inv_dx2;
        const double inv_eps2 = 1.0 / (cfg.epsilon * cfg.epsilon);
        std::vector<double> u_host(nlocal);
        allen_cahn::fill_initial_condition(&u_host, decomp, rank);
        const std::int64_t n_local_initial = allen_cahn::count_cells_above(
            u_host, allen_cahn::RunConfig::kLevelSetThreshold);
        if (!cfg.png_output_initial.empty()) {
          pfc::io::write_mpi_scalar_field_png_xy(MPI_COMM_WORLD, decomp, rank, u_host,
                                                 cfg.png_output_initial, -1.0, 1.0);
          if (rank == 0) {
            std::cout << "Wrote initial-state PNG: " << cfg.png_output_initial << "\n";
          }
        }
        constexpr int halo_width = allen_cahn::RunConfig::kHaloWidth;
        DevField u(domain, local_box, /*storage_halo=*/0, /*iteration_halo=*/halo_width);
        u.with_host_view([&](double *data, std::size_t n) {
          if (n != u_host.size()) {
            throw std::runtime_error("Allen–Cahn HIP: field size mismatch");
          }
          std::copy(u_host.begin(), u_host.end(), data);
        });
        u.sync_to_device();

        pfc::comm::SparseExchange<pfc::HipSpace, double> exchanger(
            u, decomp, rank, MPI_COMM_WORLD);
        if (rank == 0) {
          std::cout << "ALLEN_CAHN_HIP_HALO_MODE=device"
                    << " gpu_aware=" << (exchanger.uses_gpu_aware_mpi() ? 1 : 0)
                    << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double step_t0 = MPI_Wtime();
        for (int step = 0; step < cfg.n_steps; ++step) {
          exchanger.exchange();
          const auto faces = exchanger.face_recv_ptrs();
          allen_cahn::allen_cahn_step_hip(u.data(), faces[0], faces[1], faces[2],
                                          faces[3], faces[4], faces[5], nx, ny, nz,
                                          halo_width, inv_dx2, inv_dy2, cfg.dt, cfg.M,
                                          inv_eps2, cfg.driving_force);
          u.note_device_write();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double step_elapsed_s = MPI_Wtime() - step_t0;
        u.with_host_view([&](double *data, std::size_t n) {
          u_host.assign(data, data + n);
        });
        if (!cfg.png_output.empty()) {
          pfc::io::write_mpi_scalar_field_png_xy(MPI_COMM_WORLD, decomp, rank, u_host,
                                                 cfg.png_output, -1.0, 1.0);
          if (rank == 0) {
            std::cout << "Wrote PNG: " << cfg.png_output << "\n";
          }
        }
        double sum_u = 0.0;
        for (double v : u_host) {
          sum_u += v;
        }
        double sum_global = 0.0;
        MPI_Reduce(&sum_u, &sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
          std::cout << "Allen–Cahn FD (HIP): grid " << cfg.nx_glob << "x" << cfg.ny_glob
                    << "x1, ranks=" << nproc << ", steps=" << cfg.n_steps << "\n";
          std::cout << "Bulk driving force: " << cfg.driving_force << "\n";
          std::cout << "Global sum(phi) after stepping: " << sum_global << "\n";
        }
        allen_cahn::report_step_timing(MPI_COMM_WORLD, rank, cfg.n_steps, step_elapsed_s);
        const std::int64_t n_local_final = allen_cahn::count_cells_above(
            u_host, allen_cahn::RunConfig::kLevelSetThreshold);
        const bool growth_ok = allen_cahn::verify_level_set_area_growth(
            MPI_COMM_WORLD, rank, n_local_initial, n_local_final,
            allen_cahn::RunConfig::kMinLevelSetAreaGrowthFactor,
            allen_cahn::RunConfig::kLevelSetThreshold);
        return growth_ok ? EXIT_SUCCESS : EXIT_FAILURE;
      });
}
