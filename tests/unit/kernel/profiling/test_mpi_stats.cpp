// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mpi.h>
#include <openpfc/kernel/profiling/format.hpp>
#include <openpfc/kernel/profiling/memory_sample.hpp>
#include <openpfc/kernel/profiling/mpi_stats.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("reduce_scalar_across_ranks single process", "[profiling]") {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) return;

  const double v = 2.25;
  pfc::profiling::RankStats s =
      pfc::profiling::reduce_scalar_across_ranks(MPI_COMM_WORLD, v, 0);
  REQUIRE(s.count == 1);
  REQUIRE_THAT(s.min, WithinAbs(2.25, 1e-12));
  REQUIRE_THAT(s.max, WithinAbs(2.25, 1e-12));
  REQUIRE_THAT(s.mean, WithinAbs(2.25, 1e-12));
  REQUIRE_THAT(s.sum, WithinAbs(2.25, 1e-12));
  REQUIRE(s.stddev_sample == 0.0);
}

TEST_CASE("reduce_max_to_root", "[profiling]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto x = static_cast<double>(rank + 1);
  const double m = pfc::profiling::reduce_max_to_root(MPI_COMM_WORLD, x, 0);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) REQUIRE_THAT(m, WithinAbs(static_cast<double>(size), 1e-12));
}

TEST_CASE("format_bytes non-empty", "[profiling]") {
  REQUIRE_FALSE(pfc::profiling::format_bytes(0).empty());
  REQUIRE_FALSE(pfc::profiling::format_bytes(1536).empty());
}

#if defined(__linux__)
TEST_CASE("try_read_process_rss_bytes on Linux", "[profiling]") {
  const std::size_t rss = pfc::profiling::try_read_process_rss_bytes();
  REQUIRE(rss > 0);
}
#endif
