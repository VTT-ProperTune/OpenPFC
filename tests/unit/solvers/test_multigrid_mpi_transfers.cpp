// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/solvers/multigrid_mpi_transfers.hpp>

namespace {

using pfc::Box3i;
using pfc::Int3;
using pfc::Real3;
using Plan = pfc::solvers::MultigridPlan;
using Transfers = pfc::solvers::MultigridMPITransfers;

pfc::Domain free_q1(const Int3 &n) {
  const Real3 h{1.0 / (n[0] + 1), 1.3 / (n[1] + 1),
                0.7 / (n[2] + 1)};
  return pfc::domain::create(
      pfc::GridSize(n), pfc::PhysicalOrigin(h),
      pfc::GridSpacing(h), pfc::Bool3{false, false, false});
}

const Plan::Rank *rank_for(const Plan::Level &level, int world) {
  for (const auto &r : level.ranks)
    if (r.world_rank == world) return &r;
  return nullptr;
}

double fine_value(const Int3 &p) {
  return std::sin(0.071 * (p[0] + 1) + 0.043 * p[1]) +
         0.3 * std::cos(0.11 * (p[2] + 1) - 0.037 * p[0]);
}

double coarse_value(const Int3 &p) {
  return 0.7 + std::cos(0.13 * p[0] - 0.049 * p[2]) +
         0.4 * std::sin(0.09 * (p[1] + 1) + 0.021 * p[2]);
}

// Oracle deliberately does not use MultigridPlan routes or runtime kernels.
double basis(int i, int j) {
  const int delta = i - (2 * j + 1);
  if (delta == 0) return 1.0;
  if (delta == -1 || delta == 1) return 0.5;
  return 0.0;
}

double reference_restrict(const Int3 &coarse, const Int3 &fine_size) {
  double sum = 0.0;
  for (int dz = 0; dz <= 2; ++dz)
    for (int dy = 0; dy <= 2; ++dy)
      for (int dx = 0; dx <= 2; ++dx) {
        const Int3 at{2 * coarse[0] + dx, 2 * coarse[1] + dy,
                      2 * coarse[2] + dz};
        if (at[0] >= fine_size[0] || at[1] >= fine_size[1] ||
            at[2] >= fine_size[2]) continue;
        const double wx = dx == 1 ? 1.0 : 0.5;
        const double wy = dy == 1 ? 1.0 : 0.5;
        const double wz = dz == 1 ? 1.0 : 0.5;
        sum += wx * wy * wz * fine_value(at);
      }
  return sum;
}

double reference_prolong(const Int3 &fine, const Int3 &coarse_size) {
  double result = 0.0;
  const Int3 lo{
      std::max(0, fine[0] / 2 - 1),
      std::max(0, fine[1] / 2 - 1),
      std::max(0, fine[2] / 2 - 1)};
  const Int3 hi{
      std::min(coarse_size[0] - 1, fine[0] / 2),
      std::min(coarse_size[1] - 1, fine[1] / 2),
      std::min(coarse_size[2] - 1, fine[2] / 2)};
  for (int z = lo[2]; z <= hi[2]; ++z)
    for (int y = lo[1]; y <= hi[1]; ++y)
      for (int x = lo[0]; x <= hi[0]; ++x) {
        const Int3 j{x, y, z};
        result += basis(fine[0], x) * basis(fine[1], y) *
                  basis(fine[2], z) * coarse_value(j);
      }
  return result;
}

std::vector<Int3> layouts(int size) {
  switch (size) {
    case 1: return {{1, 1, 1}};
    case 2: return {{2, 1, 1}, {1, 1, 2}};
    case 3: return {{3, 1, 1}, {1, 1, 3}};
    case 4: return {{2, 2, 1}, {1, 2, 2}};
    case 6: return {{6, 1, 1}};
    default: return {};
  }
}

void verify_step(const Plan &plan, const Transfers &transfers,
                 std::size_t step, int rank) {
  const auto &fine = plan.levels()[step];
  const auto &coarse = plan.levels()[step + 1];
  const auto *fr = rank_for(fine, rank);
  const auto *cr = rank_for(coarse, rank);

  std::vector<double> f(fr ? fr->owned_nodes : 0);
  std::vector<double> c(cr ? cr->owned_nodes : 0);
  if (fr)
    pfc::for_each_index(fr->owned, [&](const Int3 &at) {
      f[static_cast<std::size_t>(fr->owned.to_linear(at))] = fine_value(at);
    });
  if (cr)
    pfc::for_each_index(cr->owned, [&](const Int3 &at) {
      c[static_cast<std::size_t>(cr->owned.to_linear(at))] = coarse_value(at);
    });

  std::vector<double> pf, ptf;
  transfers.prolong(step, c, pf);
  transfers.restrict_pt(step, f, ptf);

  double local_error = 0.0, local_dot_f = 0.0, local_dot_c = 0.0;
  if (fr) {
    REQUIRE(pf.size() == f.size());
    pfc::for_each_index(fr->owned, [&](const Int3 &at) {
      const auto index = static_cast<std::size_t>(fr->owned.to_linear(at));
      const double expected = reference_prolong(at, coarse.domain.size);
      local_error = std::max(local_error, std::abs(pf[index] - expected));
      local_dot_f += f[index] * pf[index];
    });
  } else REQUIRE(pf.empty());
  if (cr) {
    REQUIRE(ptf.size() == c.size());
    pfc::for_each_index(cr->owned, [&](const Int3 &at) {
      const auto index = static_cast<std::size_t>(cr->owned.to_linear(at));
      const double expected = reference_restrict(at, fine.domain.size);
      local_error = std::max(local_error, std::abs(ptf[index] - expected));
      local_dot_c += c[index] * ptf[index];
    });
  } else REQUIRE(ptf.empty());

  double max_error = 0.0;
  MPI_Allreduce(&local_error, &max_error, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  const double local_pair[2] = {local_dot_f, local_dot_c};
  double global_pair[2] = {};
  MPI_Allreduce(local_pair, global_pair, 2, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  INFO("transition " << step << " on "
       << plan.world_size() << " ranks");
  REQUIRE(max_error < 4e-12);
  REQUIRE(std::abs(global_pair[0] - global_pair[1]) <
          4e-12 * std::max({1.0, std::abs(global_pair[0]),
                             std::abs(global_pair[1])}));

  // One more invocation checks that the MPI route state is reusable and that
  // repeating the same transfer has no hidden additive accumulation.
  std::vector<double> again;
  transfers.restrict_pt(step, f, again);
  if (cr) {
    REQUIRE(again.size() == ptf.size());
    for (std::size_t i = 0; i < again.size(); ++i)
      local_error = std::max(local_error, std::abs(again[i] - ptf[i]));
  } else REQUIRE(again.empty());
  double repeated_error = 0.0;
  MPI_Allreduce(&local_error, &repeated_error, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  REQUIRE(repeated_error < 4e-12);
}

} // namespace

TEST_CASE("planned MPI Q1 transfers match independent P and P-transpose",
          "[multigrid_plan_mpi][MPI]") {
  int rank = -1, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto grids = layouts(size);
  if (grids.empty()) SKIP("supports bounded 1, 2, 3, 4 and 6 rank tests");

  for (const auto &grid : grids) {
    for (const bool tall : {false, true}) {
      if (size == 6 && tall) continue; // keep the six-rank gate bounded
      const int short_nodes = size == 1 ? 31 : size == 6 ? 127 : 63;
      const Int3 n{short_nodes, short_nodes,
                   tall ? 2 * short_nodes + 1 : short_nodes};
      const Plan plan = Plan::build(pfc::decomposition::create(
          free_q1(n), grid));
      Transfers transfers(plan, MPI_COMM_WORLD);

      for (std::size_t l = 0; l < plan.levels().size(); ++l) {
        const auto *r = rank_for(plan.levels()[l], rank);
        REQUIRE((r != nullptr) ==
                (transfers.level_comm(l) != MPI_COMM_NULL));
        if (r) {
          int local = -1, active_count = 0;
          MPI_Comm_rank(transfers.level_comm(l), &local);
          MPI_Comm_size(transfers.level_comm(l), &active_count);
          REQUIRE(local == r->level_rank);
          REQUIRE(active_count == plan.levels()[l].active_count());
        }
      }
      if (size == 6 && grid == Int3{6, 1, 1}) {
        REQUIRE(plan.levels()[0].active_count() == 6);
        REQUIRE(plan.levels()[1].active_count() == 3);
        REQUIRE(plan.levels()[2].active_count() == 1);
        const auto &routes = plan.transitions()[0].routes;
        REQUIRE(std::any_of(routes.begin(), routes.end(),
                            [](const auto &r) { return r.crosses_group; }));
      }
      for (std::size_t l = 0; l < plan.transitions().size(); ++l)
        verify_step(plan, transfers, l, rank);
    }
  }
}
