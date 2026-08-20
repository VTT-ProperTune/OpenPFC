// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/simulation/observable_reduce.hpp>

using Catch::Approx;
using pfc::Box3i;
using pfc::Domain;
using pfc::data::Field;
using pfc::sim::integrate_owned;

namespace {

// Cell-centered grid covering [-L, L]^3.
Domain gaussian_domain(int n, double L) {
  const double dx = 2.0 * L / static_cast<double>(n);
  const double origin = -L + 0.5 * dx;
  return pfc::domain::create(pfc::GridSize({n, n, n}),
                             pfc::PhysicalOrigin({origin, origin, origin}),
                             pfc::GridSpacing({dx, dx, dx}));
}

void fill_gaussian(Field<double> &field) {
  field.apply([](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z));
  });
}

double analytic_gaussian_r3() {
  // ∫ exp(-r²) dV over R³ = π^{3/2}
  return pfc::pi * std::sqrt(pfc::pi);
}

} // namespace

TEST_CASE("observable constant field integral is the domain volume",
          "[observable][unit]") {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    SKIP("needs 1 rank");
  }
  constexpr int N = 16;
  constexpr double L = 6.0;
  const auto domain = gaussian_domain(N, L);
  const auto box = Box3i::from_bounds({0, 0, 0}, {N - 1, N - 1, N - 1});
  Field<double> psi(domain, box, 0);
  psi.apply([](double, double, double) { return 1.0; });
  const double vol = 8.0 * L * L * L;
  REQUIRE(integrate_owned(psi, MPI_COMM_WORLD) == Approx(vol).margin(1e-12));
}

TEST_CASE("observable Gaussian integral 1 rank", "[observable][unit]") {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    SKIP("needs 1 rank");
  }
  constexpr int N = 32;
  constexpr double L = 6.0;
  const auto domain = gaussian_domain(N, L);
  const auto box = Box3i::from_bounds({0, 0, 0}, {N - 1, N - 1, N - 1});
  Field<double> psi(domain, box, 0);
  fill_gaussian(psi);
  const double got = integrate_owned(psi, MPI_COMM_WORLD);
  REQUIRE(got == Approx(analytic_gaussian_r3()).margin(1e-4));
}

TEST_CASE("observable Gaussian integral 4 ranks matches 1-rank discrete sum",
          "[MPI][multiple][observable]") {
  int size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (size != 4) {
    SKIP("needs 4 ranks");
  }
  constexpr int N = 32;
  constexpr double L = 6.0;
  const auto domain = gaussian_domain(N, L);
  auto decomp = pfc::decomposition::create(domain, 4);
  const auto box = pfc::decomposition::local_box(decomp, rank);
  Field<double> psi(domain, box, 0);
  fill_gaussian(psi);
  const double got = integrate_owned(psi, MPI_COMM_WORLD);

  Field<double> full(domain,
                     Box3i::from_bounds({0, 0, 0}, {N - 1, N - 1, N - 1}), 0);
  fill_gaussian(full);
  const double ref = integrate_owned(full, MPI_COMM_SELF);
  REQUIRE(got == Approx(ref).margin(1e-12));
}

#if defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

TEST_CASE("observable Gaussian integral HIPSpace 1 rank",
          "[observable][unit][hip]") {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    SKIP("needs 1 rank");
  }
  constexpr int N = 32;
  constexpr double L = 6.0;
  const auto domain = gaussian_domain(N, L);
  const auto box = Box3i::from_bounds({0, 0, 0}, {N - 1, N - 1, N - 1});
  Field<double, pfc::HIPSpace> psi(domain, box, 0);
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
  const double got = integrate_owned(psi, MPI_COMM_WORLD);
  REQUIRE(got == Approx(analytic_gaussian_r3()).margin(1e-4));
}
#endif

#if defined(OpenPFC_ENABLE_CUDA)
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

TEST_CASE("observable Gaussian integral CUDASpace 1 rank",
          "[observable][unit][cuda]") {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    SKIP("needs 1 rank");
  }
  constexpr int N = 32;
  constexpr double L = 6.0;
  const auto domain = gaussian_domain(N, L);
  const auto box = Box3i::from_bounds({0, 0, 0}, {N - 1, N - 1, N - 1});
  Field<double, pfc::CUDASpace> psi(domain, box, 0);
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
  const double got = integrate_owned(psi, MPI_COMM_WORLD);
  REQUIRE(got == Approx(analytic_gaussian_r3()).margin(1e-4));
}
#endif
