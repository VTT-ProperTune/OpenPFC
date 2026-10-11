// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "test_helpers.hpp"
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <openpfc/runtime/gpu/multigrid.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/structured_operators.hpp>
#include <random>
using Tag = pfc::backend::CUDATag;
namespace sol = pfc::solvers;
using Ops = sol::gpu::VectorOperations<Tag>;
using Vec = Ops::Vector;
TEST_CASE("device multigrid preserves host transfers boundaries and coupled coarse "
          "correction",
          "[multigrid][device]") {
  if (!pfc::gpu::test::is_cuda_available()) SKIP("GPU unavailable");
  for (int components : {1, 3})
    for (auto boundary : {sol::Boundary::Periodic, sol::Boundary::Dirichlet})
      for (auto coarse :
           {sol::CoarseOperator::Rediscretized, sol::CoarseOperator::Galerkin})
        for (int n : {5, 9, 17}) {
          sol::StructuredGrid grid{{n, n, n}, components, boundary};
          auto factory = [&](const sol::StructuredGrid &g) {
            return g.components == 1 ? sol::poisson_auxiliary(g)
                                     : sol::elastic_auxiliary(g, 1, .5);
          };
          sol::MultigridControls controls;
          controls.coarse_operator = coarse;
          sol::Multigrid host(grid, factory, controls);
          sol::gpu::Multigrid<Tag> mg(grid, factory, controls);
          Ops ops;
          auto b = ops.make(grid.size()), z = ops.make(grid.size());
          std::vector<double> rhs(grid.size()), reference;
          std::mt19937 rng(391);
          for (auto &v : rhs) v = std::uniform_real_distribution<double>(-1, 1)(rng);
          grid.project(rhs);
          b.copy_from_host(rhs);
          host(rhs, reference);
          mg(b, z);
          auto got = z.to_host();
          double error = 0, scale = 0;
          for (int i = 0; i < grid.size(); ++i) {
            error += (got[i] - reference[i]) * (got[i] - reference[i]);
            scale += reference[i] * reference[i];
          }
          REQUIRE(std::sqrt(error / std::max(1., scale)) < 1e-12);
          // Coupled random RHS and manufactured displacement test every component.
          auto truth = rhs;
          auto a = factory(grid);
          a(truth, rhs);
          b.copy_from_host(rhs);
          auto x = ops.make(grid.size());
          sol::KrylovControls c;
          c.relative_tolerance = 1e-9;
          c.restart = 30;
          auto result = sol::gmres(
              ops, [&](const Vec &v, Vec &w) { mg.operator_at(0)(v, w); }, b, x, c,
              [&](const Vec &v, Vec &w, int i) { mg(v, w, i); });
          REQUIRE(result.converged());
          REQUIRE(result.iterations < 60);
          got = x.to_host();
          error = scale = 0;
          for (int i = 0; i < grid.size(); ++i) {
            error += (got[i] - truth[i]) * (got[i] - truth[i]);
            scale += truth[i] * truth[i];
          }
          REQUIRE(std::sqrt(error / std::max(1., scale)) < 1e-7);
          if (boundary == sol::Boundary::Periodic) {
            std::vector<double> constant(grid.size(), 1);
            b.copy_from_host(constant);
            mg(b, z);
            REQUIRE(ops.norm(z) < 1e-12);
          }
        }
}
int main(int argc, char **argv) { return Catch::Session().run(argc, argv); }
