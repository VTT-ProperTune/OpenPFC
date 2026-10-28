// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * `float` instantiations of the host padded FD services against `double`.
 *
 * Inputs are float-representable so both paths start from identical values.
 * Integer data on power-of-two spacings is exact in both precisions and must
 * agree bitwise; generic data must agree within the `float` rounding bound.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/face_current.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/fd_padded_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

namespace {

struct Grad {
  double x{}, y{}, z{};
};

constexpr double kEps = std::numeric_limits<float>::epsilon();

template <class T>
std::unique_ptr<pfc::sim::stacks::BasicFDPaddedCPUStack<T>> make_stack(double dx) {
  auto domain =
      pfc::domain::with_spacing({6, 5, 4}, {dx, dx, dx}, {true, true, true});
  return std::make_unique<pfc::sim::stacks::BasicFDPaddedCPUStack<T>>(
      std::move(domain), 1, 0, 1, MPI_COMM_SELF);
}

/// Fill owned cells with `value(gx, gy, gz)`, halos with a sentinel.
template <class T, class F> void fill(pfc::data::Field<T> &u, F value) {
  std::fill(u.vec().begin(), u.vec().end(), T{-7});
  u.for_each_owned([&](int i, int j, int k) {
    const auto g = u.global(i, j, k);
    u(i, j, k) = static_cast<T>(value(g[0], g[1], g[2]));
  });
}

double integer_value(int x, int y, int z) { return 3 * x - 2 * y + 5 * z + 1; }

} // namespace

TEST_CASE("float padded CPU stack exchanges exact periodic halos",
          "[field][float_services]") {
  auto stack = make_stack<float>(0.5);
  static_assert(std::is_same_v<decltype(stack->u()), pfc::data::Field<float> &>);
  auto u = stack->make_field();
  fill(u, [](int x, int y, int z) { return 16777215.0 - x - 8 * y - 64 * z; });
  std::vector<pfc::data::Field<float> *> fields{&u};
  auto exchange = stack->make_exchange(fields, {});
  exchange.exchange();
  const auto n = u.local_size();
  const auto gs = u.global_size();
  // One rank: the x halos are the periodic images of the last/first plane.
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j) {
      REQUIRE(u(-1, j, k) ==
              static_cast<float>(16777215.0 - (gs[0] - 1) - 8 * j - 64 * k));
      REQUIRE(u(n[0], j, k) == static_cast<float>(16777215.0 - 8 * j - 64 * k));
    }
}

TEST_CASE("float FD gradient matches double", "[field][float_services]") {
  for (const bool exact : {true, false}) {
    const double dx = exact ? 0.5 : 0.37;
    auto sf = make_stack<float>(dx);
    auto sd = make_stack<double>(dx);
    auto uf = sf->make_field();
    auto ud = sd->make_field();
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> random(uf.size());
    for (auto &v : random) v = dist(rng);
    auto value = [&](int x, int y, int z) {
      return exact ? integer_value(x, y, z)
                   : static_cast<double>(
                         random[static_cast<std::size_t>(x + 6 * (y + 5 * z))]);
    };
    fill(uf, value);
    fill(ud, value);
    std::vector<pfc::data::Field<float> *> ff{&uf};
    std::vector<pfc::data::Field<double> *> fd{&ud};
    sf->make_exchange(ff, {}).exchange();
    sd->make_exchange(fd, {}).exchange();
    const auto gf = pfc::field::create<Grad>(uf, 2);
    const auto gd = pfc::field::create<Grad>(ud, 2);
    static_assert(std::is_same_v<std::remove_const_t<decltype(gf)>,
                                 pfc::gradient::FDGradient<Grad, float>>);
    const auto n = uf.local_size();
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const Grad a = gf(i, j, k), b = gd(i, j, k);
          // |u| <= 1 (or integer exact): (u+ - u-) / (2 dx) carries at most
          // a few float roundings of the difference and the scale.
          const double tol = exact ? 0.0 : 4 * kEps * (2.0 / (2 * dx));
          REQUIRE(std::fabs(a.x - b.x) <= tol);
          REQUIRE(std::fabs(a.y - b.y) <= tol);
          REQUIRE(std::fabs(a.z - b.z) <= tol);
        }
  }
}

TEST_CASE("float face fill and divergence match double", "[field][float_services]") {
  using namespace pfc::field::fd;
  constexpr int nx = 5, ny = 4, nz = 3;
  for (const bool exact : {true, false}) {
    const double h = exact ? 0.25 : 0.3;
    auto flux = [&](int axis, int i, int j, int k) -> double {
      const double v = integer_value(i + 1, j + 2, k + 3) + 7 * axis;
      return exact ? v : static_cast<double>(static_cast<float>(std::sin(v)));
    };
    std::array<std::vector<double>, 3> fd{std::vector<double>(n_xfaces(nx, ny, nz)),
                                          std::vector<double>(n_yfaces(nx, ny, nz)),
                                          std::vector<double>(n_zfaces(nx, ny, nz))};
    std::array<std::vector<float>, 3> ff{std::vector<float>(fd[0].size()),
                                         std::vector<float>(fd[1].size()),
                                         std::vector<float>(fd[2].size())};
    auto walls = model_walls();
    walls.xm.policy = WallPolicy::Zero;
    walls.zp.policy = WallPolicy::Prescribed;
    walls.zp.value = exact ? 2.0 : 0.1;
    fill_faces(flux, fd[0], fd[1], fd[2], nx, ny, nz, {true, true, true}, walls);
    fill_faces(flux, ff[0], ff[1], ff[2], nx, ny, nz, {true, true, true}, walls);
    for (int d = 0; d < 3; ++d)
      for (std::size_t q = 0; q < fd[d].size(); ++q)
        REQUIRE(ff[d][q] == static_cast<float>(fd[d][q]));
    std::vector<double> rd(n_cells(nx, ny, nz));
    std::vector<float> rf(rd.size());
    face_divergence(fd[0], fd[1], fd[2], rd, nx, ny, nz, {true, true, true}, h, h,
                    h);
    face_divergence(ff[0], ff[1], ff[2], rf, nx, ny, nz, {true, true, true}, h, h,
                    h);
    for (std::size_t q = 0; q < rd.size(); ++q) {
      // Three differences of |G| <= 1, each scaled and summed in float.
      const double tol = exact ? 0.0 : 12 * kEps * (2.0 / h);
      REQUIRE(std::fabs(rf[q] - rd[q]) <= tol);
    }
  }
}

TEST_CASE("owned Euler stores float rates and updates float fields",
          "[simulation][float_services]") {
  auto stack = make_stack<float>(0.5);
  auto u = stack->make_field();
  fill(u, integer_value);
  const auto before = u.vec();
  struct Rhs {
    void operator()(double, std::tuple<std::vector<float> &> &du) const {
      auto &v = std::get<0>(du);
      for (std::size_t q = 0; q < v.size(); ++q) v[q] = static_cast<float>(q % 4);
    }
  };
  pfc::sim::steppers::OwnedEulerStepper<Rhs, 1, float> euler(0.5, {u.size()}, Rhs{});
  REQUIRE(euler.workspace_bytes() == u.size() * sizeof(float));
  euler.step(0.0, u);
  const auto n = u.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        const auto q = u.idx(i, j, k);
        REQUIRE(u(i, j, k) == before[q] + 0.5f * static_cast<float>(q % 4));
      }
  REQUIRE(u.vec()[u.idx(-1, 0, 0)] == -7.0f); // halos are not written
}
