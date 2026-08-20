// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <stdexcept>
#include <type_traits>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/simulation/stacks/fd_padded_cpu_stack.hpp>

using pfc::sim::stacks::FDPaddedCPUStack;

TEST_CASE("FDPaddedCPUStack padded field and extra-field factory",
          "[cpu_stack][padded][stacks][unit]") {
  constexpr int N = 8;
  auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  pfc::comm::HaloExchangeOptions opt;
  opt.directions = pfc::halo::presets::Axes2D();
  FDPaddedCPUStack stack(domain, 1, rank, mpi_size, MPI_COMM_WORLD, opt);
  REQUIRE(stack.halo_width() == 1);
  REQUIRE(stack.u().storage_halo() == 1);
  REQUIRE(stack.u().local_size()[0] == N);
  REQUIRE(stack.rank() == rank);
  REQUIRE(stack.nproc() == mpi_size);

  auto extra = stack.make_field();
  REQUIRE(extra.size() == stack.u().size());
  REQUIRE(extra.storage_halo() == 1);

  auto group = stack.make_exchange({&stack.u(), &extra}, opt);
  stack.exchange_halos();
  group.exchange();
}

TEST_CASE("FDPaddedCPUStack rejects a zero storage halo",
          "[cpu_stack][padded][stacks][unit]") {
  auto domain = pfc::domain::create(pfc::GridSize({8, 8, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  REQUIRE_THROWS_AS(FDPaddedCPUStack(domain, 0, 0, 1), std::invalid_argument);
}

TEST_CASE("FDPaddedCPUStack is non-copyable and non-movable",
          "[cpu_stack][padded][stacks][unit]") {
  REQUIRE_FALSE(std::is_copy_constructible_v<FDPaddedCPUStack>);
  REQUIRE_FALSE(std::is_copy_assignable_v<FDPaddedCPUStack>);
  REQUIRE_FALSE(std::is_move_constructible_v<FDPaddedCPUStack>);
  REQUIRE_FALSE(std::is_move_assignable_v<FDPaddedCPUStack>);
}
