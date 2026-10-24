// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "fixtures/full_halo_fd6.hpp"

TEST_CASE("Host Full FD6 global-coordinate halos and fresh stage values",
          "[MPI][full_halo_fd6]") {
  auto producer = [](auto &, int, int) {};
  for (const auto periodic :
       {pfc::Bool3{true, true, true}, pfc::Bool3{false, false, false},
        pfc::Bool3{false, true, false}, pfc::Bool3{true, true, false}})
    full_halo_fd6::run<pfc::HostSpace>(periodic, false, producer);
  full_halo_fd6::run<pfc::HostSpace>({true, true, false}, true, producer);
}
TEST_CASE("Host Full FD6 rejects thin active axes", "[MPI][full_halo_fd6]") {
  full_halo_fd6::thin_rejection<pfc::HostSpace>();
}
