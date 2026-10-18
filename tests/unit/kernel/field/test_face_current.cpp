// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Independent V1/V2/V3 checks for `pfc::field::fd::face_divergence`.
 *
 * V1 manufactured conservation, walls, periodic seams, 2-D non-cubic.
 * V2 split-domain and MPI x-slab independence.
 * V3 a supplied `a_face (p_hi-p_lo)/h` functor equals `divergence_separated`.
 *
 * Deliberate sign, orientation, volumetric, one-sided-face and wall
 * faults must fail the matching control. A global sign flip still
 * conserves; conservation is not a sign check.
 *
 * No material constitutive law lives here.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/field/face_current.hpp>
#include <openpfc/kernel/field/face_flux.hpp>

using pfc::field::fd::apply_walls;
using pfc::field::fd::average_face;
using pfc::field::fd::cell_index;
using pfc::field::fd::closed_walls;
using pfc::field::fd::divergence_separated;
using pfc::field::fd::face_divergence;
using pfc::field::fd::face_flux;
using pfc::field::fd::FaceAverage;
using pfc::field::fd::fill_faces;
using pfc::field::fd::model_walls;
using pfc::field::fd::n_cells;
using pfc::field::fd::n_xfaces;
using pfc::field::fd::n_yfaces;
using pfc::field::fd::n_zfaces;
using pfc::field::fd::outer_flux;
using pfc::field::fd::WallPolicy;
using pfc::field::fd::Walls;
using pfc::field::fd::xface_index;
using pfc::field::fd::yface_index;
using pfc::field::fd::zface_index;

namespace {

std::vector<double> zeros(std::size_t n) { return std::vector<double>(n, 0.0); }

double sum_of(std::span<const double> a) {
  double s = 0.0;
  for (double v : a) s += v;
  return s;
}

double max_abs_diff(std::span<const double> a, std::span<const double> b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

} // namespace

TEST_CASE("supplied face current telescopes on manufactured G", "[face_current]") {
  const int nx = 7, ny = 5, nz = 3;
  const double dx = 0.4, dy = 0.7, dz = 1.1;
  const std::array<bool, 3> active{true, true, true};
  auto gx = zeros(n_xfaces(nx, ny, nz));
  auto gy = zeros(n_yfaces(nx, ny, nz));
  auto gz = zeros(n_zfaces(nx, ny, nz));
  auto rhs = zeros(n_cells(nx, ny, nz));

  std::fill(gx.begin(), gx.end(), 1.25);
  std::fill(gy.begin(), gy.end(), -0.4);
  std::fill(gz.begin(), gz.end(), 0.9);
  apply_walls(gx, gy, gz, nx, ny, nz, active, closed_walls());
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
  REQUIRE(std::abs(sum_of(rhs) * dx * dy * dz) < 1.0e-14);
  REQUIRE(std::abs(outer_flux(gx, gy, gz, nx, ny, nz, active, dx, dy, dz)) <
          1.0e-15);

  auto flux = [](int axis, int i, int j, int k) {
    if (axis == 0) return 0.2 + 0.05 * (i + 1) + 0.01 * j + 0.002 * k;
    if (axis == 1) return -0.3 + 0.04 * i + 0.03 * (j + 1) + 0.001 * k;
    return 0.15 + 0.01 * i + 0.02 * j + 0.06 * (k + 1);
  };
  fill_faces(flux, gx, gy, gz, nx, ny, nz, active, model_walls());
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
  const double want_div = 0.05 / dx + 0.03 / dy + 0.06 / dz;
  double max_err = 0.0;
  for (double v : rhs) max_err = std::max(max_err, std::abs(v - want_div));
  REQUIRE(max_err < 1.0e-14);
  const double vol = dx * dy * dz;
  REQUIRE(std::abs(sum_of(rhs) * vol - outer_flux(gx, gy, gz, nx, ny, nz, active, dx,
                                                  dy, dz)) < 1.0e-13);

  std::mt19937 rng(385);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  for (double &v : gx) v = U(rng);
  for (double &v : gy) v = U(rng);
  for (double &v : gz) v = U(rng);
  apply_walls(gx, gy, gz, nx, ny, nz, active, closed_walls());
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
  REQUIRE(std::abs(sum_of(rhs) * vol) < 1.0e-13);
  REQUIRE(std::abs(outer_flux(gx, gy, gz, nx, ny, nz, active, dx, dy, dz)) <
          1.0e-15);
}

TEST_CASE("supplied face current 2-D non-cubic and periodic seam",
          "[face_current]") {
  {
    const int nx = 9, ny = 6, nz = 1;
    const double dx = 0.3, dy = 1.2, dz = 1.0;
    const std::array<bool, 3> active{true, true, false};
    auto gx = zeros(n_xfaces(nx, ny, nz));
    auto gy = zeros(n_yfaces(nx, ny, nz));
    std::span<double> gz{};
    auto rhs = zeros(n_cells(nx, ny, nz));
    auto flux = [](int axis, int i, int j, int) {
      return axis == 0 ? 0.4 * (i + 1) : -0.25 * (j + 1);
    };
    fill_faces(flux, gx, gy, gz, nx, ny, nz, active, model_walls());
    face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
    const double want = 0.4 / dx + (-0.25) / dy;
    double max_err = 0.0;
    for (double v : rhs) max_err = std::max(max_err, std::abs(v - want));
    REQUIRE(max_err < 1.0e-14);
    REQUIRE(std::abs(sum_of(rhs) * dx * dy -
                     outer_flux(gx, gy, gz, nx, ny, nz, active, dx, dy, dz)) <
            1.0e-13);
  }
  {
    const int nx = 8, ny = 4, nz = 1;
    const double h = 0.5;
    const std::array<bool, 3> active{true, true, false};
    auto gx = zeros(n_xfaces(nx, ny, nz));
    auto gy = zeros(n_yfaces(nx, ny, nz));
    std::span<double> gz{};
    auto rhs = zeros(n_cells(nx, ny, nz));
    auto flux = [nx, ny](int axis, int i, int j, int) {
      const double I = axis == 0 ? std::fmod(i + 1 + nx, nx) : i;
      const double J = axis == 1 ? std::fmod(j + 1 + ny, ny) : j;
      constexpr double twopi = 6.28318530717958647692;
      return std::sin(twopi * I / nx) + 0.3 * std::cos(twopi * J / ny);
    };
    fill_faces(flux, gx, gy, gz, nx, ny, nz, active, model_walls());
    double seam = 0.0;
    for (int j = 0; j < ny; ++j)
      seam = std::max(seam, std::abs(gx[xface_index(nx, ny, 0, j, 0)] -
                                     gx[xface_index(nx, ny, nx, j, 0)]));
    for (int i = 0; i < nx; ++i)
      seam = std::max(seam, std::abs(gy[yface_index(nx, ny, i, 0, 0)] -
                                     gy[yface_index(nx, ny, i, ny, 0)]));
    REQUIRE(seam == 0.0);
    face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, h, h, 1.0);
    REQUIRE(std::abs(sum_of(rhs)) < 1.0e-14);
  }
}

TEST_CASE("supplied face current prescribed-wall budget", "[face_current]") {
  const int nx = 5, ny = 4, nz = 2;
  const double dx = 0.2, dy = 0.5, dz = 0.8;
  const std::array<bool, 3> active{true, true, true};
  Walls w = closed_walls();
  w.xp = {WallPolicy::Prescribed, 0.35};
  w.ym = {WallPolicy::Prescribed, -0.12};
  auto gx = zeros(n_xfaces(nx, ny, nz));
  auto gy = zeros(n_yfaces(nx, ny, nz));
  auto gz = zeros(n_zfaces(nx, ny, nz));
  auto rhs = zeros(n_cells(nx, ny, nz));
  fill_faces([](int, int, int, int) { return 7.0; }, gx, gy, gz, nx, ny, nz, active,
             w);
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
  const double want = 0.35 * dy * dz * ny * nz - (-0.12) * dx * dz * nx * nz;
  REQUIRE(std::abs(sum_of(rhs) * dx * dy * dz - want) < 1.0e-13);
  REQUIRE(std::abs(outer_flux(gx, gy, gz, nx, ny, nz, active, dx, dy, dz) - want) <
          1.0e-13);
}

TEST_CASE("supplied a_face grad p functor equals divergence_separated",
          "[face_current]") {
  const int nx = 6, ny = 5, nz = 1;
  const double dx = 0.4, dy = 0.7, dz = 1.0;
  const std::array<bool, 3> active{true, true, false};
  std::vector<double> a(n_cells(nx, ny, nz)), p(a.size());
  std::mt19937 rng(143);
  std::uniform_real_distribution<double> U(0.2, 1.8);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const auto q = cell_index(nx, ny, i, j, k);
        a[q] = U(rng);
        p[q] = U(rng) - 1.0;
      }
  auto at = [&](const std::vector<double> &f, int i, int j) -> double {
    const int ii = (i + nx) % nx;
    const int jj = (j + ny) % ny;
    return f[cell_index(nx, ny, ii, jj, 0)];
  };
  auto gx = zeros(n_xfaces(nx, ny, nz));
  auto gy = zeros(n_yfaces(nx, ny, nz));
  std::span<double> gz{};
  auto flux = [&](int axis, int i, int j, int) {
    const int i1 = axis == 0 ? i + 1 : i;
    const int j1 = axis == 1 ? j + 1 : j;
    const double a_lo = at(a, i, j), a_hi = at(a, i1, j1);
    const double p_lo = at(p, i, j), p_hi = at(p, i1, j1);
    const double h = axis == 0 ? dx : dy;
    const double af = average_face(FaceAverage::Arithmetic, a_lo, a_hi);
    return face_flux(af, p_hi, p_lo, h);
  };
  fill_faces(flux, gx, gy, gz, nx, ny, nz, active, model_walls());
  auto rhs = zeros(n_cells(nx, ny, nz));
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);

  std::array<std::vector<double>, 6> ah, ph;
  ah[0].resize(ny);
  ah[1].resize(ny);
  ph[0].resize(ny);
  ph[1].resize(ny);
  ah[2].resize(nx);
  ah[3].resize(nx);
  ph[2].resize(nx);
  ph[3].resize(nx);
  for (int j = 0; j < ny; ++j) {
    ah[0][static_cast<std::size_t>(j)] = at(a, nx, j);
    ph[0][static_cast<std::size_t>(j)] = at(p, nx, j);
    ah[1][static_cast<std::size_t>(j)] = at(a, -1, j);
    ph[1][static_cast<std::size_t>(j)] = at(p, -1, j);
  }
  for (int i = 0; i < nx; ++i) {
    ah[2][static_cast<std::size_t>(i)] = at(a, i, ny);
    ph[2][static_cast<std::size_t>(i)] = at(p, i, ny);
    ah[3][static_cast<std::size_t>(i)] = at(a, i, -1);
    ph[3][static_cast<std::size_t>(i)] = at(p, i, -1);
  }
  std::array<const double *, 6> ap{}, pp{};
  for (int f = 0; f < 4; ++f) {
    ap[static_cast<std::size_t>(f)] = ah[static_cast<std::size_t>(f)].data();
    pp[static_cast<std::size_t>(f)] = ph[static_cast<std::size_t>(f)].data();
  }
  std::vector<double> opfc(a.size(), 0.0);
  divergence_separated(a, p, ap, pp, opfc, nx, ny, 1, active, dx, dy, dz,
                       FaceAverage::Arithmetic);
  REQUIRE(max_abs_diff(rhs, opfc) < 1.0e-14);
}

TEST_CASE("supplied face current split box matches the full domain",
          "[face_current]") {
  const int nx = 8, ny = 3, nz = 2, mid = 4;
  const double dx = 0.25, dy = 0.5, dz = 0.75;
  const std::array<bool, 3> active{true, true, true};
  auto flux = [](int axis, int i, int j, int k) {
    return 0.1 * (axis + 1) + 0.02 * i + 0.03 * j + 0.04 * k;
  };
  auto gx = zeros(n_xfaces(nx, ny, nz));
  auto gy = zeros(n_yfaces(nx, ny, nz));
  auto gz = zeros(n_zfaces(nx, ny, nz));
  auto full = zeros(n_cells(nx, ny, nz));
  fill_faces(flux, gx, gy, gz, nx, ny, nz, active, model_walls());
  face_divergence(gx, gy, gz, full, nx, ny, nz, active, dx, dy, dz);

  const int nL = mid, nR = nx - mid;
  auto gxL = zeros(n_xfaces(nL, ny, nz));
  auto gyL = zeros(n_yfaces(nL, ny, nz));
  auto gzL = zeros(n_zfaces(nL, ny, nz));
  auto gxR = zeros(n_xfaces(nR, ny, nz));
  auto gyR = zeros(n_yfaces(nR, ny, nz));
  auto gzR = zeros(n_zfaces(nR, ny, nz));
  auto fluxL = [&](int axis, int i, int j, int k) { return flux(axis, i, j, k); };
  auto fluxR = [&](int axis, int i, int j, int k) {
    return flux(axis, axis == 0 ? i + mid : i, j, k);
  };
  fill_faces(fluxL, gxL, gyL, gzL, nL, ny, nz, active, model_walls());
  fill_faces(fluxR, gxR, gyR, gzR, nR, ny, nz, active, model_walls());
  double max_shared = 0.0;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j) {
      const double a = gxL[xface_index(nL, ny, nL, j, k)];
      const double b = gxR[xface_index(nR, ny, 0, j, k)];
      max_shared = std::max(max_shared, std::abs(a - b));
    }
  REQUIRE(max_shared == 0.0);
  auto rL = zeros(n_cells(nL, ny, nz));
  auto rR = zeros(n_cells(nR, ny, nz));
  face_divergence(gxL, gyL, gzL, rL, nL, ny, nz, active, dx, dy, dz);
  face_divergence(gxR, gyR, gzR, rR, nR, ny, nz, active, dx, dy, dz);
  double max_cell = 0.0;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const double got = i < mid ? rL[cell_index(nL, ny, i, j, k)]
                                   : rR[cell_index(nR, ny, i - mid, j, k)];
        max_cell =
            std::max(max_cell, std::abs(got - full[cell_index(nx, ny, i, j, k)]));
      }
  REQUIRE(max_cell < 1.0e-15);
}

TEST_CASE("supplied face current faults fail the matching control",
          "[face_current]") {
  const int nx = 6, ny = 5, nz = 2;
  const double dx = 0.3, dy = 0.4, dz = 0.5;
  const std::array<bool, 3> active{true, true, true};
  auto gx = zeros(n_xfaces(nx, ny, nz));
  auto gy = zeros(n_yfaces(nx, ny, nz));
  auto gz = zeros(n_zfaces(nx, ny, nz));
  auto rhs = zeros(n_cells(nx, ny, nz));
  auto flux = [](int axis, int i, int j, int k) {
    return 0.2 + 0.03 * i + 0.04 * j + 0.05 * k + 0.1 * axis;
  };
  fill_faces(flux, gx, gy, gz, nx, ny, nz, active, model_walls());
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
  auto ref = rhs;

  for (double &v : gx) v = -v;
  for (double &v : gy) v = -v;
  for (double &v : gz) v = -v;
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
  REQUIRE(max_abs_diff(rhs, ref) > 1.0e-8);
  REQUIRE(std::abs(sum_of(rhs) * dx * dy * dz + sum_of(ref) * dx * dy * dz) <
          1.0e-13);

  fill_faces(flux, gx, gy, gz, nx, ny, nz, active, model_walls());
  auto gx_flip = gx, gy_flip = gy, gz_flip = gz;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        std::swap(gx_flip[xface_index(nx, ny, i, j, k)],
                  gx_flip[xface_index(nx, ny, i + 1, j, k)]);
        std::swap(gy_flip[yface_index(nx, ny, i, j, k)],
                  gy_flip[yface_index(nx, ny, i, j + 1, k)]);
        std::swap(gz_flip[zface_index(nx, ny, i, j, k)],
                  gz_flip[zface_index(nx, ny, i, j, k + 1)]);
      }
  auto rflip = zeros(n_cells(nx, ny, nz));
  face_divergence(gx_flip, gy_flip, gz_flip, rflip, nx, ny, nz, active, dx, dy, dz);
  REQUIRE(max_abs_diff(rflip, ref) > 1.0e-8);

  fill_faces(flux, gx, gy, gz, nx, ny, nz, active, closed_walls());
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
  rhs[cell_index(nx, ny, 2, 2, 1)] += 0.4;
  REQUIRE(std::abs(sum_of(rhs) * dx * dy * dz) > 1.0e-8);

  fill_faces(flux, gx, gy, gz, nx, ny, nz, active, closed_walls());
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
  rhs[cell_index(nx, ny, 2, 1, 0)] += gx[xface_index(nx, ny, 3, 1, 0)] / dx;
  REQUIRE(std::abs(sum_of(rhs) * dx * dy * dz) > 1.0e-8);

  Walls w = closed_walls();
  w.xp = {WallPolicy::Prescribed, 0.5};
  fill_faces(flux, gx, gy, gz, nx, ny, nz, active, w);
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
  const double honest = sum_of(rhs) * dx * dy * dz;
  fill_faces(flux, gx, gy, gz, nx, ny, nz, active, closed_walls());
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
  REQUIRE(std::abs(sum_of(rhs) * dx * dy * dz - honest) > 1.0e-8);
}

TEST_CASE("supplied face current rejects bad extents", "[face_current]") {
  std::vector<double> rhs{1.0};
  REQUIRE_THROWS_AS(face_divergence({}, {}, {}, rhs, 0, 1, 1, {false, false, false},
                                    1.0, 1.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      face_divergence({}, {}, {}, rhs, 1, 1, 1, {true, false, false}, 1.0, 1.0, 1.0),
      std::invalid_argument);
}

TEST_CASE("supplied face current MPI x-slab is decomposition-independent",
          "[MPI][face_current]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  REQUIRE((size == 1 || size == 2 || size == 4 || size == 8));
  const int nx_g = 8, ny = 3, nz = 2;
  REQUIRE(nx_g % size == 0);
  const int nx = nx_g / size;
  const int x0 = rank * nx;
  const double dx = 0.25, dy = 0.5, dz = 0.8;
  const std::array<bool, 3> active{true, true, true};
  auto flux = [x0](int axis, int i, int j, int k) {
    const int I = axis == 0 ? i + x0 : i;
    return 0.05 * (axis + 1) + 0.02 * I + 0.03 * j + 0.04 * k;
  };
  auto gx = zeros(n_xfaces(nx, ny, nz));
  auto gy = zeros(n_yfaces(nx, ny, nz));
  auto gz = zeros(n_zfaces(nx, ny, nz));
  auto rhs = zeros(n_cells(nx, ny, nz));
  fill_faces(flux, gx, gy, gz, nx, ny, nz, active, model_walls());
  face_divergence(gx, gy, gz, rhs, nx, ny, nz, active, dx, dy, dz);
  const double local = sum_of(rhs) * dx * dy * dz;
  const double local_outer = outer_flux(gx, gy, gz, nx, ny, nz, active, dx, dy, dz);
  double global = 0.0, global_outer = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_outer, &global_outer, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(std::abs(global - global_outer) < 1.0e-13);
  const double want =
      (0.02 / dx + 0.03 / dy + 0.04 / dz) * nx_g * ny * nz * dx * dy * dz;
  REQUIRE(std::abs(global - want) < 1.0e-12);

  std::vector<double> plus(static_cast<std::size_t>(ny) * nz), minus(plus.size()),
      recv(plus.size());
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j) {
      plus[static_cast<std::size_t>(k) * ny + j] = gx[xface_index(nx, ny, nx, j, k)];
      minus[static_cast<std::size_t>(k) * ny + j] = gx[xface_index(nx, ny, 0, j, k)];
    }
  MPI_Request reqs[2];
  int nreq = 0;
  if (rank + 1 < size)
    MPI_Isend(plus.data(), static_cast<int>(plus.size()), MPI_DOUBLE, rank + 1, 0,
              MPI_COMM_WORLD, &reqs[nreq++]);
  if (rank > 0)
    MPI_Irecv(recv.data(), static_cast<int>(recv.size()), MPI_DOUBLE, rank - 1, 0,
              MPI_COMM_WORLD, &reqs[nreq++]);
  if (nreq) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
  int bad = 0;
  if (rank > 0) {
    for (std::size_t i = 0; i < minus.size(); ++i)
      if (minus[i] != recv[i]) bad = 1;
  }
  int any = 0;
  MPI_Allreduce(&bad, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(any == 0);
}

TEST_CASE("pointwise face divergence matches independent bulk and analytic currents",
          "[face_current]") {
  for (const auto n :
       {std::array{7, 5, 3}, std::array{1, 5, 3}, std::array{3, 1, 2}}) {
    const auto [nx, ny, nz] = n;
    const std::array<bool, 3> active{true, true, true};
    const double dx = 0.4, dy = 0.7, dz = 1.1;
    auto flux = [](int axis, int i, int j, int k) {
      const double x = i + 1, y = j + 1, z = k + 1;
      return axis == 0   ? 2 * x * x + j - k
             : axis == 1 ? -3 * y * y + i
                         : 4 * z * z + j;
    };
    for (int policy = 0; policy < 3; ++policy) {
      auto walls = model_walls();
      if (policy == 1) walls = closed_walls();
      if (policy == 2) {
        walls.xm = {WallPolicy::Prescribed, 2.5};
        walls.zp = {WallPolicy::Prescribed, -1.2};
      }
      auto gx = zeros(n_xfaces(nx, ny, nz)), gy = zeros(n_yfaces(nx, ny, nz)),
           gz = zeros(n_zfaces(nx, ny, nz));
      auto reference = zeros(n_cells(nx, ny, nz));
      fill_faces(flux, gx, gy, gz, nx, ny, nz, active, walls);
      face_divergence(gx, gy, gz, reference, nx, ny, nz, active, dx, dy, dz);
      const auto eval = face_divergence(flux, nx, ny, nz, active, dx, dy, dz, walls);
      for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
          for (int i = 0; i < nx; ++i) {
            REQUIRE(eval(i, j, k) == reference[cell_index(nx, ny, i, j, k)]);
            if (policy == 0)
              CHECK(std::abs(eval(i, j, k) -
                             (2 * (2 * i + 1) / dx - 3 * (2 * j + 1) / dy +
                              4 * (2 * k + 1) / dz)) < 1e-12);
          }
    }
  }
}

TEST_CASE(
    "pointwise periodic variable-coefficient current telescopes across rank cuts",
    "[face_current]") {
  constexpr int nx = 11, ny = 7, nz = 5;
  auto wrap = [](int i, int n) { return (i % n + n) % n; };
  auto coefficient = [&](int i, int j, int k) {
    return 1 + 0.1 * wrap(i, nx) + 0.03 * wrap(j, ny) + 0.02 * wrap(k, nz);
  };
  auto potential = [&](int i, int j, int k) {
    const int x = wrap(i, nx), y = wrap(j, ny), z = wrap(k, nz);
    return double(x * x - 3 * y + z * z);
  };
  auto global_flux = [&](int axis, int i, int j, int k) {
    const int a = i + (axis == 0), b = j + (axis == 1), c = k + (axis == 2);
    return face_flux(average_face(FaceAverage::Arithmetic, coefficient(i, j, k),
                                  coefficient(a, b, c)),
                     potential(a, b, c), potential(i, j, k), 0.5);
  };
  const auto whole = face_divergence(global_flux, nx, ny, nz, {true, true, true},
                                     0.5, 0.5, 0.5, model_walls());
  double total = 0;
  // Includes a rank owning only one cell along an active global axis.
  for (const auto cut : {std::array{0, 1}, std::array{1, 5}, std::array{5, nx}}) {
    auto local_flux = [&](int axis, int i, int j, int k) {
      return global_flux(axis, i + cut[0], j, k);
    };
    const auto local =
        face_divergence(local_flux, cut[1] - cut[0], ny, nz, {true, true, true}, 0.5,
                        0.5, 0.5, model_walls());
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < cut[1] - cut[0]; ++i) {
          REQUIRE(local(i, j, k) == whole(i + cut[0], j, k));
          total += local(i, j, k);
        }
  }
  CHECK(std::abs(total) < 1e-10);
}

TEST_CASE(
    "pointwise face divergence rejects invalid geometry and skips inactive axes",
    "[face_current]") {
  auto flux = [](int axis, int i, int, int) {
    if (axis != 0) throw std::runtime_error("inactive axis evaluated");
    return double(i + 1);
  };
  const auto eval =
      face_divergence(flux, 3, 1, 1, {true, false, false}, 0.5, 0, 0, model_walls());
  CHECK(eval(0, 0, 0) == 2);
  REQUIRE_THROWS_AS(
      face_divergence(flux, 0, 1, 1, {true, false, false}, 1, 1, 1, model_walls()),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      face_divergence(flux, 1, 1, 1, {true, false, false}, 0, 1, 1, model_walls()),
      std::invalid_argument);
}
