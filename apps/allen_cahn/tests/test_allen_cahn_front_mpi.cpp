// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_allen_cahn_front_mpi.cpp
 * @brief Sub-cell fronts stay global when the zero crosses an x-rank boundary.
 *
 * A local-x diagnostic misses the interpolant that uses one sample from each
 * side of the split, and a per-rank periodic wrap drops or double-counts the
 * domain seam. This process is only meaningful under `mpiexec -n 2`.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <vector>

#include <mpi.h>

#include <allen_cahn/common.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

using Catch::Matchers::WithinAbs;

namespace {

constexpr int kNx = 8;
constexpr int kNy = 2;

struct FrontOracle {
  double rising;
  double falling;
  double area;
  int positive_samples;
};

FrontOracle measure(const std::array<double, kNx> &column) {
  std::vector<double> field(static_cast<std::size_t>(kNx * kNy));
  int positive = 0;
  for (int iy = 0; iy < kNy; ++iy) {
    for (int ix = 0; ix < kNx; ++ix) {
      const double value = column[static_cast<std::size_t>(ix)];
      field[static_cast<std::size_t>(ix + kNx * iy)] = value;
      positive += value > 0.0 ? 1 : 0;
    }
  }
  const auto fronts = allen_cahn::planar_fronts(field.data(), kNx, kNy);
  return FrontOracle{
      fronts.rising.position,
      fronts.falling.position,
      allen_cahn::periodic_positive_area(field.data(), kNx, kNy),
      positive,
  };
}

} // namespace

TEST_CASE("sub-cell fronts agree across an x-rank boundary",
          "[AllenCahn][kinetics][mpi][subcell]") {
  int nproc = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  REQUIRE(nproc == 2);

  auto domain = pfc::domain::create(pfc::GridSize({kNx, kNy, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, pfc::types::Int3{2, 1, 1});
  const auto grid = pfc::decomposition::get_grid(decomp);
  REQUIRE(grid[0] == 2);
  REQUIRE(grid[1] == 1);
  REQUIRE(grid[2] == 1);

  // Both ranks read both boxes. Rank 0 owns x = 0..3 and rank 1 owns x = 4..7,
  // so the samples on either side of index 3.5 belong to different ranks, and
  // the periodic seam joins rank 1's last cell to rank 0's first cell.
  const auto left = pfc::decomposition::local_box(decomp, 0);
  const auto right = pfc::decomposition::local_box(decomp, 1);
  REQUIRE(left.low[0] == 0);
  REQUIRE(left.high[0] == 3);
  REQUIRE(left.size[0] == 4);
  REQUIRE(right.low[0] == 4);
  REQUIRE(right.high[0] == 7);
  REQUIRE(right.size[0] == 4);
  REQUIRE(left.size[1] == kNy);
  REQUIRE(right.size[1] == kNy);

  const auto mine = pfc::decomposition::local_box(decomp, rank);

  // The partial cells are not complements, so the interpolated area is not
  // the integer count of positive samples. A dropped seam sample or a
  // double-counted boundary cell moves that area.
  // Rank 0 owns the first four samples. The first row crosses the split;
  // the second crosses the periodic seam, so each rank owns each side once.
  const std::array<std::array<double, kNx>, 2> columns = {{
      {-1.0, -1.0, -1.0, -1.0, 0.2, 1.0, 1.0, 1.0},
      {1.0, 1.0, 1.0, 0.2, -1.0, -1.0, -1.0, -1.0},
  }};
  const std::array<double, 2> rising_at{{3.0 + 1.0 / 1.2, 7.5}};
  const std::array<double, 2> falling_at{{7.5, 3.0 + 1.0 / 6.0}};

  for (std::size_t case_index = 0; case_index < columns.size(); ++case_index) {
    const auto &column = columns[case_index];
    const FrontOracle oracle = measure(column);
    REQUIRE(oracle.area != static_cast<double>(oracle.positive_samples));
    REQUIRE_THAT(oracle.rising, WithinAbs(rising_at[case_index], 1e-12));
    REQUIRE_THAT(oracle.falling, WithinAbs(falling_at[case_index], 1e-12));

    std::vector<double> local(static_cast<std::size_t>(mine.size[0] * mine.size[1]));
    for (int iy = 0; iy < mine.size[1]; ++iy) {
      for (int ix = 0; ix < mine.size[0]; ++ix) {
        const int gx = mine.low[0] + ix;
        local[static_cast<std::size_t>(ix + mine.size[0] * iy)] =
            column[static_cast<std::size_t>(gx)];
      }
    }

    const std::vector<double> gathered = allen_cahn::gather_xy_samples(
        MPI_COMM_WORLD, decomp, rank, local.data(), mine.size[0], mine.size[1]);
    REQUIRE(gathered.size() == static_cast<std::size_t>(kNx * kNy));
    for (int iy = 0; iy < kNy; ++iy) {
      for (int ix = 0; ix < kNx; ++ix) {
        REQUIRE(gathered[static_cast<std::size_t>(ix + kNx * iy)] ==
                column[static_cast<std::size_t>(ix)]);
      }
    }

    allen_cahn::SubcellSamples samples;
    samples.planar = true;
    allen_cahn::sample_subcell(MPI_COMM_WORLD, decomp, rank, local.data(),
                               mine.size[0], mine.size[1], &samples,
                               allen_cahn::SampleWhen::Initial);
    REQUIRE(samples.fronts_initial.rising.found);
    REQUIRE(samples.fronts_initial.falling.found);
    REQUIRE_THAT(samples.fronts_initial.rising.position,
                 WithinAbs(oracle.rising, 1e-12));
    REQUIRE_THAT(samples.fronts_initial.falling.position,
                 WithinAbs(oracle.falling, 1e-12));
    REQUIRE_THAT(samples.area_initial, WithinAbs(oracle.area, 1e-12));
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
