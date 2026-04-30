// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_HIP)
#error "wave2d_hip requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>

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
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>

#include <wave2d/cli.hpp>
#include <wave2d/device_step.hpp>
#include <wave2d/vtk_snapshot.hpp>
#include <wave2d/wave_model.hpp>
#include <wave2d/wave_step_separated.hpp>

namespace {

void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

} // namespace

int main(int argc, char *argv[]) {
  const std::string exe = (argc > 0 && argv[0] != nullptr)
                              ? std::string(argv[0])
                              : std::string("wave2d_hip");
  const auto cfg_o = wave2d::parse_manual(argc, argv);
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (!cfg_o) {
    if (rank == 0) {
      wave2d::print_usage(std::cerr, exe.c_str(), false);
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  const wave2d::RunConfig &cfg = *cfg_o;
  const int Nx = cfg.Nx;
  const int Ny = cfg.Ny;
  const int n_steps = cfg.n_steps;
  const double dt = cfg.dt;

  auto world = pfc::world::create(pfc::GridSize({Nx, Ny, 1}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, nproc);

  const auto &local_world = pfc::decomposition::get_subworld(decomp, rank);
  auto local_size = pfc::world::get_size(local_world);
  const auto lower = pfc::world::get_lower(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const std::size_t nlocal = static_cast<std::size_t>(nx) *
                             static_cast<std::size_t>(ny) *
                             static_cast<std::size_t>(nz);

  const auto gw = pfc::world::get_size(world);
  const std::array<int, 3> global_vtk{gw[0], gw[1], gw[2]};
  const std::array<int, 3> local_vtk{nx, ny, nz};
  const std::array<int, 3> off_vtk{lower[0], lower[1], lower[2]};
  const auto worg = pfc::world::get_origin(world);
  const auto wsp = pfc::world::get_spacing(world);
  const std::array<double, 3> origin_vtk{worg[0], worg[1], worg[2]};
  const std::array<double, 3> spacing_vtk{wsp[0], wsp[1], wsp[2]};

  const double inv_dx2 = 1.0;
  const double inv_dy2 = 1.0;
  constexpr int halo_width = 1;

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

  auto face_halos_host =
      pfc::halo::allocate_face_halos<double>(decomp, rank, halo_width);
  pfc::SeparatedFaceHaloExchanger<double> exchanger(decomp, rank, halo_width,
                                                    MPI_COMM_WORLD);
  const auto counts = pfc::halo::face_halo_counts(decomp, rank, halo_width);

  double *u_dev = nullptr;
  double *v_dev = nullptr;
  hip_check(hipMalloc(reinterpret_cast<void **>(&u_dev), nlocal * sizeof(double)),
            "hipMalloc u");
  hip_check(hipMalloc(reinterpret_cast<void **>(&v_dev), nlocal * sizeof(double)),
            "hipMalloc v");
  std::array<double *, 6> face_dev{};
  for (int f = 0; f < 6; ++f) {
    const std::size_t n = counts.counts[static_cast<std::size_t>(f)];
    hip_check(
        hipMalloc(reinterpret_cast<void **>(&face_dev[static_cast<std::size_t>(f)]),
                  std::max<std::size_t>(n, 1u) * sizeof(double)),
        "hipMalloc face");
  }

  hip_check(hipMemcpy(u_dev, u_host.data(), nlocal * sizeof(double),
                      hipMemcpyHostToDevice),
            "H2D u");
  hip_check(hipMemcpy(v_dev, v_host.data(), nlocal * sizeof(double),
                      hipMemcpyHostToDevice),
            "H2D v");

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
    (void)step;
    hip_check(hipMemcpy(u_host.data(), u_dev, nlocal * sizeof(double),
                        hipMemcpyDeviceToHost),
              "D2H u");
    hip_check(hipMemcpy(v_host.data(), v_dev, nlocal * sizeof(double),
                        hipMemcpyDeviceToHost),
              "D2H v");
    exchanger.exchange_halos(u_host.data(), u_host.size(), face_halos_host);
    if (cfg.y_bc == wave2d::YBoundaryKind::Dirichlet) {
      wave2d::patch_y_face_halos_dirichlet_order2(
          u_host.data(), nx, ny, face_halos_host, lower, Ny, cfg.u_wall);
    } else {
      wave2d::patch_y_face_halos_neumann_order2(u_host.data(), nx, ny,
                                                face_halos_host, lower, Ny);
    }
    for (int f = 0; f < 6; ++f) {
      const std::size_t n = counts.counts[static_cast<std::size_t>(f)];
      if (n == 0) {
        continue;
      }
      hip_check(hipMemcpy(face_dev[static_cast<std::size_t>(f)],
                          face_halos_host[static_cast<std::size_t>(f)].data(),
                          n * sizeof(double), hipMemcpyHostToDevice),
                "H2D face");
    }
    wave2d::wave2d_step_hip(u_dev, v_dev, face_dev[0], face_dev[1], face_dev[2],
                            face_dev[3], face_dev[4], face_dev[5], nx, ny, nz,
                            halo_width, inv_dx2, inv_dy2, dt, wave2d::kC);
    if (cfg.y_bc == wave2d::YBoundaryKind::Dirichlet) {
      hip_check(hipMemcpy(u_host.data(), u_dev, nlocal * sizeof(double),
                          hipMemcpyDeviceToHost),
                "D2H u post");
      hip_check(hipMemcpy(v_host.data(), v_dev, nlocal * sizeof(double),
                          hipMemcpyDeviceToHost),
                "D2H v post");
      for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
          const int gy = lower[1] + iy;
          if (gy != 0 && gy != Ny - 1) {
            continue;
          }
          for (int ix = 0; ix < nx; ++ix) {
            const std::size_t idx =
                static_cast<std::size_t>(ix) +
                static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
            u_host[idx] = cfg.u_wall;
            v_host[idx] = 0.0;
          }
        }
      }
      hip_check(hipMemcpy(u_dev, u_host.data(), nlocal * sizeof(double),
                          hipMemcpyHostToDevice),
                "H2D u wall");
      hip_check(hipMemcpy(v_dev, v_host.data(), nlocal * sizeof(double),
                          hipMemcpyHostToDevice),
                "H2D v wall");
    }

    if (vtk_writer && (step + 1) % cfg.vtk_every == 0) {
      hip_check(hipMemcpy(u_host.data(), u_dev, nlocal * sizeof(double),
                          hipMemcpyDeviceToHost),
                "D2H u vtk");
      wave2d::vtk_write_u_owned_buffer(*vtk_writer, step + 1, u_host.data(), nx, ny,
                                       nz, vtk_buf);
    }
  }

  hipFree(u_dev);
  hipFree(v_dev);
  for (int f = 0; f < 6; ++f) {
    hipFree(face_dev[static_cast<std::size_t>(f)]);
  }

  if (rank == 0) {
    std::cout << "wave2d_hip: finished " << n_steps << " steps on " << Nx << "x"
              << Ny << " (ranks=" << nproc << ")\n";
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}
