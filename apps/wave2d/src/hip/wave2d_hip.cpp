// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_HIP)
#error "wave2d_hip requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>

#include <wave2d/device_step.hpp>
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
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (argc < 6) {
    if (rank == 0) {
      std::cerr << "Usage: " << argv[0]
                << " <Nx> <Ny> <n_steps> <dt> <y_bc> [u_wall]\n";
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  const int Nx = std::atoi(argv[1]);
  const int Ny = std::atoi(argv[2]);
  const int n_steps = std::atoi(argv[3]);
  const double dt = std::atof(argv[4]);
  const auto yb = wave2d::parse_y_bc(argv[5]);
  const double u_wall = (argc >= 7) ? std::atof(argv[6]) : 0.0;
  if (!yb || Nx < 4 || Ny < 4 || n_steps < 1 || dt <= 0.0) {
    if (rank == 0) {
      std::cerr << "Invalid arguments\n";
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

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

  for (int step = 0; step < n_steps; ++step) {
    (void)step;
    hip_check(hipMemcpy(u_host.data(), u_dev, nlocal * sizeof(double),
                        hipMemcpyDeviceToHost),
              "D2H u");
    hip_check(hipMemcpy(v_host.data(), v_dev, nlocal * sizeof(double),
                        hipMemcpyDeviceToHost),
              "D2H v");
    exchanger.exchange_halos(u_host.data(), u_host.size(), face_halos_host);
    if (*yb == wave2d::YBoundaryKind::Dirichlet) {
      wave2d::patch_y_face_halos_dirichlet_order2(
          u_host.data(), nx, ny, face_halos_host, lower, Ny, u_wall);
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
    if (*yb == wave2d::YBoundaryKind::Dirichlet) {
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
            u_host[idx] = u_wall;
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
