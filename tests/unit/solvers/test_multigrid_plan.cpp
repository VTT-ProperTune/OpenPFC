// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/solvers/multigrid_plan.hpp>

namespace {

using Plan = pfc::solvers::MultigridPlan;
using Options = pfc::solvers::MultigridPlanOptions;
using pfc::Box3i;
using pfc::Int3;
using pfc::Real3;

pfc::Domain free_q1(const Int3 &n) {
  return pfc::domain::create(
      pfc::GridSize(n), pfc::PhysicalOrigin({
          1.0 / (n[0] + 1), 1.3 / (n[1] + 1), 0.7 / (n[2] + 1)}),
      pfc::GridSpacing({
          1.0 / (n[0] + 1), 1.3 / (n[1] + 1), 0.7 / (n[2] + 1)}),
      pfc::Bool3{false, false, false});
}

std::uint64_t volume(const Int3 &n) {
  return static_cast<std::uint64_t>(n[0]) * n[1] * n[2];
}

int owner_of(const Plan::Level &level, const Int3 &at) {
  int owner = -1;
  for (const auto &r : level.ranks) {
    if (!r.owned.contains(at)) continue;
    REQUIRE(owner == -1);
    owner = r.world_rank;
  }
  REQUIRE(owner >= 0);
  return owner;
}

void check_hierarchy_invariants(const Plan &plan) {
  const auto &levels = plan.levels();
  const auto &steps = plan.transitions();
  REQUIRE(levels.size() >= 2);
  REQUIRE(steps.size() + 1 == levels.size());
  REQUIRE(levels.front().active_count() == plan.world_size());
  REQUIRE(levels.back().active_count() == 1);
  REQUIRE(levels.back().active_world_ranks[0] == 0);

  for (std::size_t i = 0; i < levels.size(); ++i) {
    const auto &L = levels[i];
    REQUIRE(L.index == static_cast<int>(i));
    REQUIRE(L.ranks.size() == L.active_world_ranks.size());
    REQUIRE(static_cast<std::uint64_t>(L.active_count()) ==
            volume(L.process_grid));
    std::set<int> active(L.active_world_ranks.begin(),
                         L.active_world_ranks.end());
    REQUIRE(active.size() == L.ranks.size());
    std::uint64_t owned = 0, estimated = 0;
    for (std::size_t k = 0; k < L.ranks.size(); ++k) {
      const auto &r = L.ranks[k];
      REQUIRE(r.level_rank == static_cast<int>(k));
      REQUIRE(r.world_rank == L.active_world_ranks[k]);
      REQUIRE(r.owned == pfc::decomposition::local_box(
          L.partition, static_cast<int>(k)));
      REQUIRE(r.owned_nodes == volume(r.owned.size));
      REQUIRE(r.padded_nodes >= r.owned_nodes);
      REQUIRE(r.estimated_device_bytes ==
              8u * static_cast<std::uint64_t>(
                  plan.options().estimated_padded_fields) * r.padded_nodes);
      owned += r.owned_nodes;
      estimated += r.estimated_device_bytes;
    }
    REQUIRE(owned == L.global_nodes);
    REQUIRE(owned == volume(L.domain.size));
    REQUIRE(estimated == L.total_estimated_device_bytes);

    if (i + 1 == levels.size()) break;
    const auto &C = levels[i + 1];
    const auto &T = steps[i];
    REQUIRE(T.fine_level == static_cast<int>(i));
    REQUIRE(T.coarse_level == static_cast<int>(i + 1));
    REQUIRE(T.groups.size() == C.ranks.size());
    for (int axis = 0; axis < 3; ++axis) {
      REQUIRE(L.domain.size[axis] == 2 * C.domain.size[axis] + 1);
      REQUIRE(C.domain.spacing[axis] == Catch::Approx(
          2.0 * L.domain.spacing[axis]));
      REQUIRE(C.domain.origin[axis] == Catch::Approx(
          L.domain.origin[axis] + L.domain.spacing[axis]));
      REQUIRE(L.process_grid[axis] ==
              T.rank_contraction[axis] * C.process_grid[axis]);
      REQUIRE((T.rank_contraction[axis] == 1 ||
               T.rank_contraction[axis] == 2 ||
               T.rank_contraction[axis] == 3));
    }
    std::set<int> grouped;
    for (std::size_t c = 0; c < T.groups.size(); ++c) {
      const auto &g = T.groups[c];
      REQUIRE(g.coarse_world_rank == C.active_world_ranks[c]);
      REQUIRE(std::find(g.fine_world_ranks.begin(),
                        g.fine_world_ranks.end(),
                        g.coarse_world_rank) != g.fine_world_ranks.end());
      for (int w : g.fine_world_ranks)
        REQUIRE(grouped.insert(w).second);
    }
    REQUIRE(grouped == active);

    std::uint64_t messages = 0;
    for (const auto &route : T.routes) {
      REQUIRE(active.count(route.fine_world_rank) == 1);
      REQUIRE(std::find(C.active_world_ranks.begin(),
                        C.active_world_ranks.end(),
                        route.coarse_world_rank) != C.active_world_ranks.end());
      REQUIRE(route.coarse_overlap.is_consistent());
      REQUIRE(route.fine_support_overlap.is_consistent());
      REQUIRE(route.values_per_direction ==
              volume(route.coarse_overlap.size));
      REQUIRE(route.values_per_direction <=
              plan.options().maximum_route_elements);
      for (int a = 0; a < 3; ++a) {
        REQUIRE(route.fine_support_overlap.low[a] >= 0);
        REQUIRE(route.fine_support_overlap.high[a] < L.domain.size[a]);
        REQUIRE(route.coarse_overlap.low[a] >= 0);
        REQUIRE(route.coarse_overlap.high[a] < C.domain.size[a]);
      }
      messages += route.values_per_direction;
    }
    REQUIRE(T.routed_values_per_direction == messages);
    REQUIRE(T.routed_bytes_per_direction == 8 * messages);

    // Independent support oracle. Each fine-owned cell contributes to
    // exactly one owner of every coarse support point. This is true for
    // R=P^T additive reductions and reverse P interpolation broadcasts.
    for (const auto &f : L.ranks) {
      const int lo[3] = {
          std::max(0, (f.owned.low[0] + 1) / 2 - 1),
          std::max(0, (f.owned.low[1] + 1) / 2 - 1),
          std::max(0, (f.owned.low[2] + 1) / 2 - 1)};
      const int hi[3] = {
          std::min(C.domain.size[0] - 1, f.owned.high[0] / 2),
          std::min(C.domain.size[1] - 1, f.owned.high[1] / 2),
          std::min(C.domain.size[2] - 1, f.owned.high[2] / 2)};
      for (int z = lo[2]; z <= hi[2]; ++z)
        for (int y = lo[1]; y <= hi[1]; ++y)
          for (int x = lo[0]; x <= hi[0]; ++x) {
            const Int3 at{x, y, z};
            const int expected = owner_of(C, at);
            int found = 0;
            for (const auto &route : T.routes)
              if (route.fine_world_rank == f.world_rank &&
                  route.coarse_overlap.contains(at)) {
                REQUIRE(route.coarse_world_rank == expected);
                ++found;
              }
            REQUIRE(found == 1);
          }
    }
  }
  for (std::uint64_t bytes : plan.estimated_device_bytes_by_world_rank())
    REQUIRE(bytes > 0);
}

} // namespace

TEST_CASE("MultigridPlan contracts 64 ranks over a five-level Q1 hierarchy",
          "[multigrid_plan]") {
  const auto first = pfc::decomposition::create(
      free_q1({127, 127, 127}), Int3{4, 4, 4});
  const auto p = Plan::build(first);
  REQUIRE(p.world_size() == 64);
  REQUIRE(p.levels().size() == 5);
  REQUIRE((p.levels()[0].process_grid == Int3{4, 4, 4}));
  REQUIRE((p.levels()[1].process_grid == Int3{2, 2, 2}));
  REQUIRE((p.levels()[2].process_grid == Int3{1, 1, 1}));
  REQUIRE((p.levels()[3].process_grid == Int3{1, 1, 1}));
  REQUIRE((p.levels()[4].process_grid == Int3{1, 1, 1}));
  REQUIRE(p.levels()[0].active_count() == 64);
  REQUIRE(p.levels()[1].active_count() == 8);
  REQUIRE(p.levels()[2].active_count() == 1);
  REQUIRE(p.transitions()[0].groups[0].fine_world_ranks.size() == 8);
  REQUIRE(p.transitions()[1].groups[0].fine_world_ranks.size() == 8);
  REQUIRE(p.levels()[1].active_world_ranks[1] == 2);
  REQUIRE(p.levels()[1].active_world_ranks[2] == 8);
  REQUIRE(p.levels()[1].active_world_ranks[4] == 32);
  REQUIRE(std::any_of(p.transitions()[0].routes.begin(),
                      p.transitions()[0].routes.end(),
                      [](const auto &e) { return e.crosses_group; }));
  check_hierarchy_invariants(p);
}

TEST_CASE("MultigridPlan admits divisible 6 to 3 to 1 rank contraction",
          "[multigrid_plan]") {
  const auto first = pfc::decomposition::create(
      free_q1({255, 255, 255}), Int3{6, 1, 1});
  const auto p = Plan::build(first);
  REQUIRE(p.levels().size() == 6);
  REQUIRE(p.levels()[0].active_count() == 6);
  REQUIRE(p.levels()[1].active_count() == 6);
  REQUIRE((p.levels()[2].process_grid == Int3{3, 1, 1}));
  REQUIRE((p.levels()[3].process_grid == Int3{1, 1, 1}));
  REQUIRE((p.levels()[2].active_world_ranks == std::vector<int>{0, 2, 4}));
  REQUIRE((p.transitions()[1].rank_contraction == Int3{2, 1, 1}));
  REQUIRE((p.transitions()[2].rank_contraction == Int3{3, 1, 1}));
  check_hierarchy_invariants(p);
}

TEST_CASE("MultigridPlan retains Q1 anisotropy and 64-bit global counts",
          "[multigrid_plan]") {
  const auto p = Plan::build(pfc::decomposition::create(
      free_q1({127, 127, 255}), Int3{2, 4, 8}));
  REQUIRE(p.levels().front().active_count() == 64);
  REQUIRE((p.levels().back().domain.size == Int3{7, 7, 15}));
  check_hierarchy_invariants(p);

  // Planning does not allocate these arrays. The analytical scalar count
  // exceeds the 32-bit limit but remains a checked 64-bit quantity.
  const Int3 massive{4095, 4095, 8191};
  const auto huge = Plan::build(pfc::decomposition::create(
      free_q1(massive), Int3{2, 4, 8}));
  REQUIRE(huge.levels().front().global_nodes >
          std::numeric_limits<std::uint32_t>::max());
  REQUIRE(huge.levels().front().global_nodes == volume(massive));
  REQUIRE((huge.levels().back().domain.size == Int3{7, 7, 15}));
  REQUIRE(huge.estimated_device_bytes_by_world_rank()[0] >
          std::numeric_limits<std::uint32_t>::max());
  REQUIRE(huge.levels().back().active_count() == 1);
}

TEST_CASE("MultigridPlan rejects illegal rank factors, geometry and budgets",
          "[multigrid_plan]") {
  REQUIRE_THROWS_AS(Plan::build(pfc::decomposition::create(
                        free_q1({127, 127, 127}), Int3{5, 1, 1})),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(Plan::build(pfc::decomposition::create(
                        free_q1({64, 64, 64}), Int3{1, 1, 1})),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(Plan::build(pfc::decomposition::create(
                        free_q1({31, 31, 32}), Int3{1, 1, 1})),
                    std::invalid_argument);
  const auto fine = pfc::decomposition::create(
      free_q1({127, 127, 127}), Int3{4, 4, 4});
  Options wrong;
  wrong.halo_width = 1;
  REQUIRE_THROWS_AS(Plan::build(fine, wrong), std::invalid_argument);
  wrong = Options{};
  wrong.minimum_owned_extent = 0;
  REQUIRE_THROWS_AS(Plan::build(fine, wrong), std::invalid_argument);
  wrong = Options{};
  wrong.maximum_route_elements = 1;
  REQUIRE_THROWS_AS(Plan::build(fine, wrong), std::overflow_error);
}
