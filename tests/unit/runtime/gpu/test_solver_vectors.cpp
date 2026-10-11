// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "test_helpers.hpp"
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <openpfc/runtime/gpu/vector_operations.hpp>
#include <openpfc/solvers/newton.hpp>
template <class Tag>
concept HasResidentOperations =
    requires { typename pfc::solvers::gpu::VectorOperations<Tag>; };
static_assert(HasResidentOperations<pfc::backend::CUDATag>);
static_assert(!HasResidentOperations<pfc::backend::CPUTag>);
#if !defined(OpenPFC_ENABLE_HIP)
static_assert(!HasResidentOperations<pfc::backend::HIPTag>);
#endif
using Ops = pfc::solvers::gpu::VectorOperations<pfc::backend::CUDATag>;
using Vec = Ops::Vector;
namespace sol = pfc::solvers;
void solver_toy(int, const double *, double *, int, const double * = nullptr);
TEST_CASE("resident FGMRES solves a nonsymmetric cyclic operator with restarts",
          "[solver][device][krylov]") {
  if (!pfc::gpu::test::is_cuda_available()) SKIP("GPU unavailable");
  Ops ops(MPI_COMM_WORLD);
  constexpr int n = 73;
  auto x = ops.make(n), b = ops.make(n);
  std::vector<double> reference(n), rhs(n);
  for (int i = 0; i < n; ++i) reference[i] = std::sin(.4 * i) + .02 * i;
  for (int i = 0; i < n; ++i)
    rhs[i] = (4 + .01 * i) * reference[i] + .7 * reference[(i + 1) % n] -
             .2 * reference[(i + n - 1) % n];
  b.copy_from_host(rhs);
  for (auto side : {sol::Preconditioning::None, sol::Preconditioning::Left,
                    sol::Preconditioning::Right}) {
    ops.scale(x, 0);
    sol::KrylovControls c;
    c.restart = 4;
    c.relative_tolerance = 1e-12;
    c.preconditioning = side;
    c.flexible = side == sol::Preconditioning::Right;
    auto result = sol::gmres(
        ops, [&](const Vec &v, Vec &w) { solver_toy(n, v.data(), w.data(), 0); }, b,
        x, c,
        [&](const Vec &v, Vec &w, int k) {
          ops.copy(v, w);
          ops.scale(w, c.flexible ? 1. / (4 + .01 * k) : .25);
        });
    REQUIRE(result.converged());
    REQUIRE(result.restarts > 0);
    auto got = x.to_host();
    for (int i = 0; i < n; ++i) REQUIRE(std::abs(got[i] - reference[i]) < 1e-10);
    REQUIRE(result.residual <= c.relative_tolerance * result.rhs_norm);
  }
  ops.scale(x, 0);
  sol::KrylovControls c;
  c.maximum_iterations = 1;
  c.relative_tolerance = 1e-14;
  auto fail = sol::gmres(
      ops, [&](const Vec &v, Vec &w) { solver_toy(n, v.data(), w.data(), 0); }, b, x,
      c, [&](const Vec &v, Vec &w, int) { ops.copy(v, w); });
  REQUIRE_FALSE(fail.converged());
  REQUIRE(ops.scalar_transfer_bytes == 8 * ops.reductions.calls);
  auto z = ops.make(n);
  auto address = z.data();
  ops.copy(b, z);
  REQUIRE(z.data() == address);
  ops.scale(z, 0);
  REQUIRE(ops.norm(z) == 0);
}
struct NonlinearToy {
  Ops &ops;
  Vec base;
  int rejects = 0, accepts = 0;
  void residual(const Vec &x, Vec &r) {
    solver_toy(x.size(), x.data(), r.data(), 1);
  }
  void linearize(const Vec &x) { ops.copy(x, base); }
  void jacobian(const Vec &v, Vec &w) {
    solver_toy(v.size(), v.data(), w.data(), 2, base.data());
  }
  void precondition(const Vec &v, Vec &w, int) { ops.copy(v, w); }
  void begin_trial(const Vec &) {}
  void accept_trial(const Vec &) { ++accepts; }
  void reject_trial() { ++rejects; }
};
TEST_CASE(
    "resident Newton globalizes a nonlinear reference without committing trials",
    "[solver][device][newton]") {
  if (!pfc::gpu::test::is_cuda_available()) SKIP("GPU unavailable");
  Ops ops(MPI_COMM_WORLD);
  auto x = ops.make(19), committed = ops.make(19);
  std::vector<double> initial(19, .1);
  x.copy_from_host(initial);
  ops.copy(x, committed);
  NonlinearToy toy{ops, ops.make(19)};
  sol::NewtonControls c;
  c.maximum_iterations = 30;
  c.relative_tolerance = 1e-10;
  c.initial_forcing = .1;
  auto result = sol::newton(ops, toy, x, c);
  REQUIRE(result.converged());
  REQUIRE(toy.rejects > 0);
  REQUIRE(committed.to_host() == initial);
  auto got = x.to_host();
  for (int i = 0; i < 19; ++i) REQUIRE(std::abs(got[i] - (1 + .002 * i)) < 1e-9);
  ops.copy(x, committed);
  REQUIRE(committed.to_host() == got);
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
