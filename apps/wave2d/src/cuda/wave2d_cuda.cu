// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_CUDA)
#error "wave2d_cuda requires CUDA (configure with -DOpenPFC_ENABLE_CUDA=ON)"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/kernel/data/model_types.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/comm_sparse_exchange_gpu.hpp>

#include <wave2d/cli.hpp>
#include <wave2d/device_step.hpp>
#include <wave2d/vtk_snapshot.hpp>
#include <wave2d/wave_model.hpp>

namespace {

using DevField = pfc::data::Field<double, pfc::CUDASpace>;

void cuda_check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
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
  cuda_check(cudaGetDeviceCount(&n_dev), "cudaGetDeviceCount");
  if (n_dev < 1) {
    throw std::runtime_error("No CUDA devices visible to this rank");
  }
  cuda_check(cudaSetDevice(local_rank % n_dev), "cudaSetDevice");
}

void copy_host_to_device(const std::vector<double> &host, DevField &dev) {
  dev.with_host_view([&](double *data, std::size_t n) {
    if (n != host.size()) {
      throw std::runtime_error("wave2d_cuda: field size mismatch");
    }
    std::copy(host.begin(), host.end(), data);
  });
  dev.sync_to_device();
}

void copy_device_to_host(DevField &dev, std::vector<double> &host) {
  dev.with_host_view([&](double *data, std::size_t n) {
    host.assign(data, data + n);
  });
  dev.note_device_write();
}

} // namespace

namespace {

int run_wave2d_cuda(const wave2d::RunConfig &cfg, int rank, int nproc) {
  bind_local_gpu();
  const int Nx = cfg.Nx;
  const int Ny = cfg.Ny;
  const int n_steps = cfg.n_steps;
  const double dt = cfg.dt;
  auto domain = pfc::domain::create(pfc::GridSize({Nx, Ny, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, nproc);

  const auto &local_box = pfc::decomposition::local_box(decomp, rank);
  auto local_size = local_box.size;
  const auto lower = local_box.low;
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const std::size_t nlocal = static_cast<std::size_t>(nx) *
                             static_cast<std::size_t>(ny) *
                             static_cast<std::size_t>(nz);

  const auto global_domain = pfc::decomposition::domain(decomp);
  const std::array<int, 3> global_vtk{global_domain.size[0], global_domain.size[1],
                                      global_domain.size[2]};
  const std::array<int, 3> local_vtk{nx, ny, nz};
  const std::array<int, 3> off_vtk{lower[0], lower[1], lower[2]};
  const std::array<double, 3> origin_vtk{global_domain.origin[0],
                                         global_domain.origin[1],
                                         global_domain.origin[2]};
  const std::array<double, 3> spacing_vtk{global_domain.spacing[0],
                                          global_domain.spacing[1],
                                          global_domain.spacing[2]};

  const double inv_dx2 = 1.0;
  const double inv_dy2 = 1.0;
  constexpr int halo_width = 1;
  const bool dirichlet = cfg.y_bc == wave2d::YBoundaryKind::Dirichlet;

  std::vector<double> u_host(nlocal);
  std::vector<double> v_host(nlocal, 0.0);
  const double xc = 0.5 * static_cast<double>(Nx - 1);
  const double yc = 0.5 * static_cast<double>(Ny - 1);
  const double sigma = 0.12 * static_cast<double>(std::min(Nx, Ny));
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int gx = lower[0] + ix;
        const int gy = lower[1] + iy;
        const double x = static_cast<double>(gx);
        const double y = static_cast<double>(gy);
        const double dx = x - xc;
        const double dy = y - yc;
        const std::size_t idx =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
        u_host[idx] = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
      }
    }
  }

  DevField u(domain, local_box, /*storage_halo=*/0, /*iteration_halo=*/halo_width);
  DevField v(domain, local_box, /*storage_halo=*/0, /*iteration_halo=*/halo_width);
  copy_host_to_device(u_host, u);
  copy_host_to_device(v_host, v);

  pfc::comm::SparseExchange<pfc::CUDASpace, double> exchanger(
      u, decomp, rank, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "WAVE2D_CUDA_HALO_MODE=device"
              << " gpu_aware=" << (exchanger.uses_gpu_aware_mpi() ? 1 : 0) << "\n";
  }

  pfc::RealField vtk_buf;
  std::unique_ptr<pfc::VTKWriter> vtk_writer;
  if (!cfg.vtk_pattern.empty()) {
    vtk_writer = std::make_unique<pfc::VTKWriter>(cfg.vtk_pattern);
    wave2d::vtk_configure_writer_owned_slab(*vtk_writer, global_vtk, local_vtk,
                                            off_vtk, origin_vtk, spacing_vtk);
    wave2d::mkdir_vtk_parent_rank0(cfg.vtk_pattern, rank);
    wave2d::vtk_write_u_owned_buffer(*vtk_writer, 0, u_host.data(), nx, ny, nz,
                                     vtk_buf);
  }

  for (int step = 0; step < n_steps; ++step) {
    exchanger.exchange();
    const auto faces = exchanger.face_recv_ptrs();
    wave2d::wave2d_patch_y_faces_cuda(u.data(), faces[2], faces[3], nx, ny, lower[1],
                                      Ny, dirichlet, cfg.u_wall);
    wave2d::wave2d_step_cuda(u.data(), v.data(), faces[0], faces[1], faces[2],
                             faces[3], faces[4], faces[5], nx, ny, nz, halo_width,
                             inv_dx2, inv_dy2, dt, wave2d::kC);
    if (dirichlet) {
      wave2d::wave2d_enforce_dirichlet_walls_cuda(u.data(), v.data(), nx, ny,
                                                  lower[1], Ny, cfg.u_wall);
    }
    u.note_device_write();
    v.note_device_write();

    if (vtk_writer && (step + 1) % cfg.vtk_every == 0) {
      copy_device_to_host(u, u_host);
      wave2d::vtk_write_u_owned_buffer(*vtk_writer, step + 1, u_host.data(), nx, ny,
                                       nz, vtk_buf);
    }
  }

  if (rank == 0) {
    std::cout << "wave2d_cuda: finished " << n_steps << " steps on " << Nx << "x"
              << Ny << " (ranks=" << nproc << ")\n";
  }
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[]) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg = wave2d::parse_manual_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) return EXIT_FAILURE;
        return run_wave2d_cuda(*cfg, rank, nproc);
      });
}
