// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <mpi.h>
#include <openpfc/solvers/coarse_correction.hpp>
#include <openpfc/solvers/owned_component_mean.hpp>
#include <openpfc/solvers/structured_operators.hpp>
#include <vector>
using namespace pfc::solvers;

namespace {
std::vector<int> owned_by_rank(int size, int rank, int ranks) {
  std::vector<int> owned;
  for (int i = 0; i < size; ++i)
    if (i % ranks == rank) owned.push_back(i);
  return owned;
}

std::vector<double> pattern_plus_constant(const std::vector<int> &owned,
                                          int components, int nodes,
                                          const std::vector<double> &constant) {
  std::vector<double> values(owned.size());
  for (std::size_t i = 0; i < owned.size(); ++i) {
    int node = owned[i] / components;
    int component = owned[i] % components;
    double pattern = node == 0 ? nodes - 1. : -1.;
    values[i] = pattern + constant[static_cast<std::size_t>(component)];
  }
  return values;
}
} // namespace

TEST_CASE("owned component mean uses the global node count", "[multigrid][coarse]") {
  std::vector<double> one{8};
  remove_owned_component_mean(one, {0}, 1, 4, MPI_COMM_NULL);
  REQUIRE(one[0] == 6);
  REQUIRE_THROWS_AS(remove_owned_component_mean(one, {0, 1}, 1, 4, MPI_COMM_NULL),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(remove_owned_component_mean(one, {-1}, 1, 4, MPI_COMM_NULL),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(remove_owned_component_mean(one, {0}, 0, 4, MPI_COMM_NULL),
                    std::invalid_argument);
}

TEST_CASE("owned component mean restores one constant under unequal ownership",
          "[multigrid][MPI][coarse]") {
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 2) SKIP("owned component mean needs two ranks");
  for (int components : {1, 3}) {
    StructuredGrid grid{{9, 7, 5}, components};
    const int nodes = grid.nodes();
    auto owned = owned_by_rank(grid.size(), rank, ranks);
    int count = static_cast<int>(owned.size()), counts[2];
    MPI_Allgather(&count, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    REQUIRE(counts[0] != counts[1]);
    std::vector<double> constant(static_cast<std::size_t>(components));
    for (int c = 0; c < components; ++c)
      constant[static_cast<std::size_t>(c)] = 3 - c;
    auto original = pattern_plus_constant(owned, components, nodes, constant);
    std::vector<double> local_mean(static_cast<std::size_t>(components));
    std::vector<int> local_count(static_cast<std::size_t>(components));
    for (std::size_t i = 0; i < owned.size(); ++i) {
      int component = owned[i] % components;
      local_mean[static_cast<std::size_t>(component)] += original[i];
      ++local_count[static_cast<std::size_t>(component)];
    }
    for (int c = 0; c < components; ++c)
      local_mean[static_cast<std::size_t>(c)] /=
          local_count[static_cast<std::size_t>(c)];
    std::vector<double> both(static_cast<std::size_t>(2 * components));
    MPI_Allgather(local_mean.data(), components, MPI_DOUBLE, both.data(), components,
                  MPI_DOUBLE, MPI_COMM_WORLD);
    REQUIRE(std::abs(both[0] - both[static_cast<std::size_t>(components)]) > 1);

    auto values = original;
    remove_owned_component_mean(values, owned, components, nodes, MPI_COMM_WORLD);
    auto restored = pattern_plus_constant(owned, components, nodes,
                                          std::vector<double>(components, 0));
    for (std::size_t i = 0; i < values.size(); ++i) {
      int component = owned[i] % components;
      REQUIRE(values[i] == restored[i]);
      REQUIRE(original[i] - values[i] ==
              constant[static_cast<std::size_t>(component)]);
    }
    auto again = values;
    remove_owned_component_mean(again, owned, components, nodes, MPI_COMM_WORLD);
    REQUIRE(again == values);

    auto factory = [=](const StructuredGrid &g) {
      return components == 1 ? poisson_auxiliary(g) : elastic_auxiliary(g, 1, .5);
    };
    SharedCoarseCorrection correction(grid, owned, factory, MPI_COMM_WORLD);
    auto gauged = original;
    correction.gauge(gauged);
    REQUIRE(gauged == values);

    StructuredGrid dirichlet{{9, 7, 5}, components, Boundary::Dirichlet};
    SharedCoarseCorrection fixed(dirichlet, owned, factory, MPI_COMM_WORLD);
    auto kept = original;
    fixed.gauge(kept);
    REQUIRE(kept == original);
  }
}
