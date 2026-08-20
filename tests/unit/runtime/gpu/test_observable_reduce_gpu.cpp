// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OPENPFC_TEST_OBSERVABLE_HIP) &&                                   \
    !defined(OPENPFC_TEST_OBSERVABLE_CUDA)

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

#else

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/observable_reduce.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

#if defined(OPENPFC_TEST_OBSERVABLE_HIP)
using Space = pfc::HIPSpace;
#else
using Space = pfc::CUDASpace;
#endif

using Catch::Approx;

TEST_CASE("device observable Gaussian integral 1 rank",
          "[gpu][observable]") {
#if defined(OPENPFC_TEST_OBSERVABLE_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
#endif

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }

  constexpr int N = 32;
  constexpr double L = 6.0;
  const double dx = 2.0 * L / static_cast<double>(N);
  const double origin = -L + 0.5 * dx;
  auto domain = pfc::domain::create(
      pfc::GridSize({N, N, N}),
      pfc::PhysicalOrigin({origin, origin, origin}),
      pfc::GridSpacing({dx, dx, dx}));
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {N - 1, N - 1, N - 1});
  pfc::data::Field<double, Space> ones(domain, box, 0);
  ones.with_host_view([&](double *data, std::size_t n) {
    std::fill(data, data + n, 1.0);
  });
  const double vol = 8.0 * L * L * L;
  REQUIRE(pfc::sim::integrate_owned(ones, MPI_COMM_WORLD) ==
          Approx(vol).margin(1e-12));

  pfc::data::Field<double, Space> psi(domain, box, 0);
  psi.with_host_view([&](double *data, std::size_t) {
    const int nx = psi.box().size[0];
    const int ny = psi.box().size[1];
    const int nz = psi.box().size[2];
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const auto c = psi.coords(i, j, k);
          data[psi.idx(i, j, k)] =
              std::exp(-(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]));
        }
      }
    }
  });
  const double got = pfc::sim::integrate_owned(psi, MPI_COMM_WORLD);
  const double expect = pfc::pi * std::sqrt(pfc::pi);
  REQUIRE(got == Approx(expect).margin(1e-4));
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  return Catch::Session().run(argc, argv);
}

#endif
