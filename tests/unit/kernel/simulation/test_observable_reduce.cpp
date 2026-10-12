// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <span>
#include <vector>

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

namespace {

// Each rank contributes (rank + 1) * (i + 1); the sum over P ranks is
// (i + 1) * P * (P + 1) / 2.
void check_span_allreduce() {
  int size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> values(5);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<double>((rank + 1) * (i + 1));
  }
  pfc::sim::allreduce_sum(std::span<double>(values), MPI_COMM_WORLD);
  const double ranks = static_cast<double>(size) * (size + 1) / 2.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    REQUIRE(values[i] == static_cast<double>(i + 1) * ranks);
  }
  std::vector<double> empty;
  REQUIRE_NOTHROW(pfc::sim::allreduce_sum(std::span<double>(empty), MPI_COMM_WORLD));
}

} // namespace

TEST_CASE("allreduce_sum over a span sums element-wise", "[observable][unit]") {
  check_span_allreduce();
}

TEST_CASE("allreduce_sum over a span sums element-wise across ranks",
          "[MPI][observable]") {
  check_span_allreduce();
}

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

  Field<double> full(domain, Box3i::from_bounds({0, 0, 0}, {N - 1, N - 1, N - 1}),
                     0);
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

TEST_CASE("owned reductions exclude a halo and match a known field",
          "[observable][unit]") {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) SKIP("needs 1 rank");
  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = pfc::domain::index_box(domain);
  Field<double> u(domain, box, 1);
  u.with_host_view([&](double *data, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) data[i] = -99.0;
  });
  u(0, 0, 0) = 1.0;
  u(1, 0, 0) = 3.0;
  const auto r = pfc::sim::reduce_owned(u, MPI_COMM_WORLD);
  REQUIRE(r.count == 2);
  REQUIRE(r.min == Approx(1.0));
  REQUIRE(r.max == Approx(3.0));
  REQUIRE(r.sum == Approx(4.0));
  REQUIRE(r.mean == Approx(2.0));
  REQUIRE(r.l1 == Approx(4.0));
  REQUIRE(r.l2 == Approx(std::sqrt(10.0)));
  REQUIRE(r.rms == Approx(std::sqrt(5.0)));
  REQUIRE(r.variance == Approx(1.0));
  const auto squares =
      pfc::sim::reduce_owned(u, MPI_COMM_WORLD, [](double v) { return v * v; });
  REQUIRE(squares.sum == Approx(10.0));
  REQUIRE(squares.min == Approx(1.0));
}

TEST_CASE("owned reductions agree across two ranks", "[observable][MPI]") {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) SKIP("needs 2 ranks");
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto domain = pfc::domain::create({4, 1, 1});
  const auto local = Box3i::from_bounds({rank * 2, 0, 0}, {rank * 2 + 1, 0, 0});
  Field<double> u(domain, local, 0);
  u(0, 0, 0) = static_cast<double>(rank * 2 + 1);
  u(1, 0, 0) = static_cast<double>(rank * 2 + 2);
  const auto r = pfc::sim::reduce_owned(u, MPI_COMM_WORLD);
  REQUIRE(r.count == 4);
  REQUIRE(r.min == Approx(1.0));
  REQUIRE(r.max == Approx(4.0));
  REQUIRE(r.mean == Approx(2.5));
  REQUIRE(r.sum == Approx(10.0));
}

TEST_CASE("variance of a large baseline keeps the fluctuation",
          "[observable][unit]") {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) SKIP("needs 1 rank");
  constexpr double base = 1.0e12;
  const double devs[4] = {-2.0, -1.0, 1.0, 2.0};
  double naive_mean = 0.0;
  double naive_sq = 0.0;
  for (double d : devs) {
    const double x = base + d;
    naive_mean += x;
    naive_sq += x * x;
  }
  naive_mean /= 4.0;
  const double naive_var = naive_sq / 4.0 - naive_mean * naive_mean;
  REQUIRE(std::abs(naive_var - 2.5) > 1.0);

  const auto domain = pfc::domain::create({4, 1, 1});
  const auto box = pfc::domain::index_box(domain);
  Field<double> u(domain, box, 0);
  for (int i = 0; i < 4; ++i) u(i, 0, 0) = base + devs[i];
  const auto r = pfc::sim::reduce_owned(u, MPI_COMM_WORLD);
  REQUIRE(r.count == 4);
  REQUIRE(r.mean == Approx(base).margin(1e-3));
  REQUIRE(r.variance == Approx(2.5).margin(1e-8));
  REQUIRE(r.min == Approx(base - 2.0));
  REQUIRE(r.max == Approx(base + 2.0));
  REQUIRE(r.variance >= 0.0);
}

TEST_CASE("shifted variance agrees across two ranks", "[observable][MPI]") {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) SKIP("needs 2 ranks");
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  constexpr double base = 1.0e12;
  const auto domain = pfc::domain::create({4, 1, 1});
  const auto local = Box3i::from_bounds({rank * 2, 0, 0}, {rank * 2 + 1, 0, 0});
  Field<double> u(domain, local, 0);
  const double devs[4] = {-2.0, -1.0, 1.0, 2.0};
  u(0, 0, 0) = base + devs[rank * 2];
  u(1, 0, 0) = base + devs[rank * 2 + 1];
  const auto r = pfc::sim::reduce_owned(u, MPI_COMM_WORLD);
  REQUIRE(r.count == 4);
  REQUIRE(r.variance == Approx(2.5).margin(1e-6));
  REQUIRE(r.mean == Approx(base).margin(1e-2));
}
