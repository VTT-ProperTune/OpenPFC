// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../../../fixtures/session.hpp"

#include "mechanics_backend.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <openpfc/runtime/gpu/krylov.hpp>
#include <openpfc/runtime/gpu/multigrid.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/structured_operators.hpp>
#include <random>
using Tag = mechanics_test::Tag;
namespace sol = pfc::solvers;
using Ops = sol::gpu::VectorOperations<Tag>;
using Vec = Ops::Vector;
using Matrix = std::vector<std::vector<double>>;
namespace {
sol::gpu::SparseOperator<Tag> upload(const Matrix &a) {
  std::vector<int> offsets{0}, columns;
  std::vector<double> values, diagonal(a.size());
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    for (int j = 0; j < static_cast<int>(a[i].size()); ++j)
      if (a[i][j] != 0) {
        columns.push_back(j);
        values.push_back(a[i][j]);
        if (i == j) diagonal[static_cast<std::size_t>(i)] = a[i][j];
      }
    offsets.push_back(static_cast<int>(values.size()));
  }
  return sol::gpu::SparseOperator<Tag>(static_cast<int>(a.size()),
                                       static_cast<int>(a.size()), offsets, columns,
                                       values, diagonal);
}
double max_abs(const std::vector<double> &a, const std::vector<double> &b) {
  double error = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    error = std::max(error, std::abs(a[i] - b[i]));
  return error;
}
template <class Action, class Pre>
void agree(Action &&action, Pre &&pre, const std::vector<double> &rhs,
           const std::vector<double> &guess, sol::KrylovControls controls,
           bool use_cg, int poll = 1) {
  Ops host_ops, device_ops;
  auto hb = host_ops.make(rhs.size()), hx = host_ops.make(rhs.size());
  auto db = device_ops.make(rhs.size()), dx = device_ops.make(rhs.size());
  hb.copy_from_host(rhs);
  hx.copy_from_host(guess);
  db.copy_from_host(rhs);
  dx.copy_from_host(guess);
  sol::KrylovResult host, device;
  if (use_cg) {
    host = sol::cg(host_ops, action, hb, hx, controls, pre);
    device = sol::gpu::cg(device_ops, action, db, dx, controls, pre, poll);
  } else {
    host = sol::gmres(host_ops, action, hb, hx, controls, pre);
    device = sol::gpu::gmres(device_ops, action, db, dx, controls, pre, poll);
  }
  INFO("host iterations " << host.iterations << " device " << device.iterations);
  REQUIRE(device.status == host.status);
  REQUIRE(device.iterations == host.iterations);
  if (poll == 1) {
    REQUIRE(device.operator_calls == host.operator_calls);
    REQUIRE(device.preconditioner_calls == host.preconditioner_calls);
    REQUIRE(device.history.size() == host.history.size());
  }
  REQUIRE(max_abs(dx.to_host(), hx.to_host()) < 1e-8);
  REQUIRE(std::abs(device.residual - host.residual) < 1e-8);
}
} // namespace
TEST_CASE("device Krylov scalars agree with the host-scalar GPU path",
          "[krylov][device]") {
  if (!mechanics_test::available()) SKIP("GPU unavailable");
  Matrix spd{{4, -1, 0}, {-1, 4, -1}, {0, -1, 3}};
  Matrix nonsymmetric(9, std::vector<double>(9));
  std::vector<double> nonsymmetric_rhs(9);
  for (int i = 0; i < 9; ++i) {
    nonsymmetric[i][i] = 3 + .1 * i;
    nonsymmetric_rhs[static_cast<std::size_t>(i)] = std::sin(1.3 * i) + .2;
    if (i < 8) nonsymmetric[i][i + 1] = 1.4;
    if (i > 0) nonsymmetric[i][i - 1] = -.6;
  }
  auto spd_op = upload(spd);
  auto nonsymmetric_op = upload(nonsymmetric);
  Matrix diagonal(9, std::vector<double>(9));
  for (int i = 0; i < 9; ++i) diagonal[i][i] = 1 / nonsymmetric[i][i];
  auto diagonal_op = upload(diagonal);
  auto spd_action = [&](const Vec &x, Vec &y) { spd_op(x, y); };
  auto nonsymmetric_action = [&](const Vec &x, Vec &y) { nonsymmetric_op(x, y); };
  Ops ferry;
  auto copy = [&](const Vec &r, Vec &z, int) { ferry.copy(r, z); };
  auto jacobi = [&](const Vec &r, Vec &z, int) { diagonal_op(r, z); };
  std::vector<double> spd_rhs{1, 3, 2}, zero3(3), guess9(9, .07);
  sol::KrylovControls cg_controls;
  cg_controls.relative_tolerance = 1e-12;
  agree(spd_action, copy, spd_rhs, zero3, cg_controls, true);
  agree(spd_action, copy, spd_rhs, zero3, cg_controls, true, 4);
  cg_controls.preconditioning = sol::Preconditioning::None;
  auto forbidden = [](const Vec &, Vec &, int) {
    throw std::runtime_error("inverse must be bypassed");
  };
  agree(spd_action, forbidden, spd_rhs, zero3, cg_controls, true);
  Matrix negative{{-1, 0}, {0, -2}};
  auto negative_op = upload(negative);
  auto negative_action = [&](const Vec &x, Vec &y) { negative_op(x, y); };
  std::vector<double> negative_rhs{1, 2}, zero2(2);
  cg_controls.preconditioning = sol::Preconditioning::Right;
  agree(negative_action, copy, negative_rhs, zero2, cg_controls, true);
  sol::KrylovControls gmres_controls;
  gmres_controls.restart = 3;
  gmres_controls.relative_tolerance = 1e-12;
  gmres_controls.flexible = false;
  gmres_controls.preconditioning = sol::Preconditioning::None;
  agree(nonsymmetric_action, forbidden, nonsymmetric_rhs, guess9, gmres_controls,
        false);
  for (auto side : {sol::Preconditioning::Left, sol::Preconditioning::Right}) {
    gmres_controls.preconditioning = side;
    agree(nonsymmetric_action, jacobi, nonsymmetric_rhs, guess9, gmres_controls,
          false);
  }
  gmres_controls.preconditioning = sol::Preconditioning::Right;
  gmres_controls.flexible = true;
  agree(nonsymmetric_action, jacobi, nonsymmetric_rhs, guess9, gmres_controls, false,
        4);
  Matrix small{{4, 2, -1}, {-1, 3, 1}, {0, -2, 5}};
  auto small_op = upload(small);
  auto small_action = [&](const Vec &x, Vec &y) { small_op(x, y); };
  auto variable = [&](const Vec &r, Vec &z, int k) {
    auto host = r.to_host();
    for (int i = 0; i < 3; ++i)
      host[static_cast<std::size_t>(i)] /= 1 + ((k + 2 * i) % 4);
    if (z.size() != host.size()) z.resize(host.size());
    z.copy_from_host(host);
  };
  sol::KrylovControls flexible;
  flexible.restart = 3;
  flexible.relative_tolerance = 1e-12;
  agree(small_action, variable, std::vector<double>{1, -4, 2}, zero3, flexible,
        false);
  Ops ops;
  auto b = ops.make(3), x = ops.make(3);
  b.copy_from_host(spd_rhs);
  REQUIRE_THROWS_AS(sol::gpu::cg(ops, spd_action, b, x, cg_controls, copy, 0),
                    std::invalid_argument);
}
TEST_CASE("device Krylov scalars copy less on the multigrid problem",
          "[krylov][device]") {
  if (!mechanics_test::available()) SKIP("GPU unavailable");
  sol::StructuredGrid grid{{33, 33, 33}, 1, sol::Boundary::Periodic};
  auto factory = [](const sol::StructuredGrid &g) {
    return sol::poisson_auxiliary(g);
  };
  sol::gpu::Multigrid<Tag> multigrid(grid, factory);
  Ops host_ops, device_ops;
  auto hb = host_ops.make(grid.size()), hx = host_ops.make(grid.size());
  auto db = device_ops.make(grid.size()), dx = device_ops.make(grid.size());
  std::vector<double> rhs(static_cast<std::size_t>(grid.size()));
  std::mt19937 rng(391);
  for (auto &value : rhs) value = std::uniform_real_distribution<double>(-1, 1)(rng);
  grid.project(rhs);
  hb.copy_from_host(rhs);
  db.copy_from_host(rhs);
  auto action = [&](const Vec &v, Vec &w) { multigrid.operator_at(0)(v, w); };
  auto inverse = [&](const Vec &v, Vec &w, int iteration) {
    multigrid(v, w, iteration);
  };
  sol::KrylovControls controls;
  controls.relative_tolerance = 1e-8;
  controls.restart = 20;
  const auto host_began = std::chrono::steady_clock::now();
  const auto host = sol::gmres(host_ops, action, hb, hx, controls, inverse);
  const double host_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - host_began)
          .count();
  const auto device_began = std::chrono::steady_clock::now();
  const auto device =
      sol::gpu::gmres(device_ops, action, db, dx, controls, inverse, 1);
  const double device_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - device_began)
          .count();
  std::cout << "device-krylov host_s=" << host_seconds
            << " device_s=" << device_seconds
            << " host_bytes=" << host_ops.scalar_transfer_bytes
            << " device_bytes=" << device_ops.scalar_transfer_bytes
            << " iterations=" << host.iterations << "\n";
  REQUIRE(host.converged());
  REQUIRE(device.converged());
  REQUIRE(device.iterations == host.iterations);
  REQUIRE(device_ops.scalar_transfer_bytes < host_ops.scalar_transfer_bytes);
  REQUIRE(max_abs(dx.to_host(), hx.to_host()) < 1e-6);
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int result = pfc::test::run(argc, argv);
  MPI_Finalize();
  return result;
}
