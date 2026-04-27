// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/frontend/utils/memory_reporter.hpp>

TEST_CASE("MemoryUsage total_bytes sums application and FFT allocations",
          "[memory_reporter]") {
  const pfc::utils::MemoryUsage usage{1024, 2048};

  REQUIRE(usage.total_bytes() == 3072);
}

#if defined(__linux__)
TEST_CASE("get_system_memory_bytes reads Linux MemTotal", "[memory_reporter]") {
  REQUIRE(pfc::utils::get_system_memory_bytes() > 0);
}
#endif
