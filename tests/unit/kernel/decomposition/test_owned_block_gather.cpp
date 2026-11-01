// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/owned_block_gather.hpp>

namespace {

int world_rank() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int world_size() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

double manufactured(int field, int i, int j, int k) {
  return 1000000.0 * field + 10000.0 * i + 100.0 * j + static_cast<double>(k);
}

std::vector<double> pack_manufactured(const pfc::Box3i &box, int field) {
  std::vector<double> values(static_cast<std::size_t>(box.count()));
  pfc::for_each_index(box, [&](const std::array<int, 3> &idx) {
    values[static_cast<std::size_t>(box.to_linear(idx))] =
        manufactured(field, idx[0], idx[1], idx[2]);
  });
  return values;
}

template <typename Call>
void require_invalid(const Call &call, const char *fragment) {
  REQUIRE_THROWS_AS(call(), std::invalid_argument);
  try {
    call();
    FAIL("expected std::invalid_argument");
  } catch (const std::invalid_argument &error) {
    REQUIRE(std::string(error.what()).find(fragment) != std::string::npos);
  }
}

void transfer_shape(const pfc::Int3 &shape) {
  const int nprocs = world_size();
  const int rank = world_rank();
  const pfc::Domain domain = pfc::domain::create(shape);
  const pfc::Decomposition decomp =
      pfc::decomposition::create(domain, pfc::Int3{nprocs, 1, 1});
  bool uneven = nprocs == 1;
  if (nprocs > 1) {
    const auto first = pfc::decomposition::local_box(decomp, 0).count();
    uneven = false;
    for (int other = 1; other < nprocs; ++other) {
      if (pfc::decomposition::local_box(decomp, other).count() != first) {
        uneven = true;
      }
    }
  }

  const pfc::Box3i box = pfc::decomposition::local_box(decomp, rank);
  constexpr int nfields = 2;
  pfc::Field<double> halo_field(domain, box, 1);
  for (std::size_t cell = 0; cell < halo_field.size(); ++cell) {
    halo_field.data()[cell] = -7.0;
  }
  for (int k = 0; k < box.size[2]; ++k) {
    for (int j = 0; j < box.size[1]; ++j) {
      for (int i = 0; i < box.size[0]; ++i) {
        halo_field(i, j, k) =
            manufactured(0, box.low[0] + i, box.low[1] + j, box.low[2] + k);
      }
    }
  }
  std::vector<std::vector<double>> packed(nfields);
  packed[0] = pfc::decomposition::pack_owned_block(halo_field);
  packed[1] = pack_manufactured(box, 1);
  bool halo_kept = halo_field(-1, 0, 0) == -7.0 && halo_field(0, -1, 0) == -7.0 &&
                   halo_field(0, 0, -1) == -7.0;
  bool packed_owned = packed[0].size() == static_cast<std::size_t>(box.count()) &&
                      packed[0].size() < halo_field.size();
  for (double value : packed[0]) {
    if (value == -7.0) {
      packed_owned = false;
    }
  }
  pfc::Field<double> restored(domain, box, 1);
  for (std::size_t cell = 0; cell < restored.size(); ++cell) {
    restored.data()[cell] = -7.0;
  }
  pfc::decomposition::unpack_owned_block(restored, packed[0]);
  bool unpacked = restored(-1, 0, 0) == -7.0;
  for (int k = 0; k < box.size[2]; ++k) {
    for (int j = 0; j < box.size[1]; ++j) {
      for (int i = 0; i < box.size[0]; ++i) {
        const double expected =
            manufactured(0, box.low[0] + i, box.low[1] + j, box.low[2] + k);
        if (restored(i, j, k) != expected) {
          unpacked = false;
        }
      }
    }
  }

  const std::size_t nglob = static_cast<std::size_t>(shape[0]) * shape[1] * shape[2];
  std::vector<int> roots{0};
  if (nprocs > 1) {
    roots.push_back(nprocs - 1);
  }
  for (int root : roots) {
    std::vector<pfc::decomposition::OwnedBlockView> views;
    views.reserve(nfields);
    for (int field = 0; field < nfields; ++field) {
      views.push_back({box, packed[static_cast<std::size_t>(field)]});
    }
    const auto gathered = pfc::decomposition::gather_owned_blocks(
        decomp, MPI_COMM_WORLD, root,
        std::span<const pfc::decomposition::OwnedBlockView>(views));

    std::vector<std::vector<double>> round_trip(nfields);
    std::vector<pfc::decomposition::OwnedBlockBuffer> round_buffers;
    for (int field = 0; field < nfields; ++field) {
      round_trip[static_cast<std::size_t>(field)].assign(packed[field].size(), -1.0);
      round_buffers.push_back({box, round_trip[static_cast<std::size_t>(field)]});
    }
    pfc::decomposition::scatter_owned_blocks(
        decomp, MPI_COMM_WORLD, root, gathered,
        std::span<pfc::decomposition::OwnedBlockBuffer>(round_buffers));

    std::vector<std::vector<double>> oracle;
    if (rank == root) {
      oracle.resize(nfields);
      for (int field = 0; field < nfields; ++field) {
        oracle[static_cast<std::size_t>(field)].resize(nglob);
        for (int k = 0; k < shape[2]; ++k) {
          for (int j = 0; j < shape[1]; ++j) {
            for (int i = 0; i < shape[0]; ++i) {
              const std::size_t index = static_cast<std::size_t>(i) +
                                        static_cast<std::size_t>(shape[0]) *
                                            (static_cast<std::size_t>(j) +
                                             static_cast<std::size_t>(shape[1]) * k);
              oracle[static_cast<std::size_t>(field)][index] =
                  manufactured(field, i, j, k);
            }
          }
        }
      }
    }
    std::vector<std::vector<double>> scattered(nfields);
    std::vector<pfc::decomposition::OwnedBlockBuffer> scatter_buffers;
    for (int field = 0; field < nfields; ++field) {
      scattered[static_cast<std::size_t>(field)].assign(packed[field].size(), -1.0);
      scatter_buffers.push_back({box, scattered[static_cast<std::size_t>(field)]});
    }
    pfc::decomposition::scatter_owned_blocks(
        decomp, MPI_COMM_WORLD, root, oracle,
        std::span<pfc::decomposition::OwnedBlockBuffer>(scatter_buffers));

    REQUIRE(uneven);
    REQUIRE(halo_kept);
    REQUIRE(packed_owned);
    REQUIRE(unpacked);
    if (rank == root) {
      REQUIRE(gathered.size() == static_cast<std::size_t>(nfields));
      for (int field = 0; field < nfields; ++field) {
        REQUIRE(gathered[static_cast<std::size_t>(field)].size() == nglob);
        for (int k = 0; k < shape[2]; ++k) {
          for (int j = 0; j < shape[1]; ++j) {
            for (int i = 0; i < shape[0]; ++i) {
              const std::size_t index = static_cast<std::size_t>(i) +
                                        static_cast<std::size_t>(shape[0]) *
                                            (static_cast<std::size_t>(j) +
                                             static_cast<std::size_t>(shape[1]) * k);
              REQUIRE(gathered[static_cast<std::size_t>(field)][index] ==
                      manufactured(field, i, j, k));
            }
          }
        }
      }
    } else {
      REQUIRE(gathered.empty());
    }
    for (int field = 0; field < nfields; ++field) {
      REQUIRE(round_trip[static_cast<std::size_t>(field)] ==
              packed[static_cast<std::size_t>(field)]);
      REQUIRE(scattered[static_cast<std::size_t>(field)] ==
              packed[static_cast<std::size_t>(field)]);
    }
  }
}

} // namespace

TEST_CASE("owned blocks gather and scatter bitwise", "[owned_block][MPI]") {
  const int nprocs = world_size();
  transfer_shape({2 * nprocs + 1, 4, 1});
  transfer_shape({2 * nprocs + 1, 3, 3});
}

TEST_CASE("owned block transfer accepts an empty field list", "[owned_block][MPI]") {
  const int nprocs = world_size();
  const pfc::Domain domain = pfc::domain::create(pfc::Int3{2 * nprocs + 1, 2, 1});
  const pfc::Decomposition decomp =
      pfc::decomposition::create(domain, pfc::Int3{nprocs, 1, 1});
  const std::span<const pfc::decomposition::OwnedBlockView> no_views;
  const auto gathered =
      pfc::decomposition::gather_owned_blocks(decomp, MPI_COMM_WORLD, 0, no_views);
  REQUIRE(gathered.empty());
  const std::span<pfc::decomposition::OwnedBlockBuffer> no_buffers;
  pfc::decomposition::scatter_owned_blocks(decomp, MPI_COMM_WORLD, 0, gathered,
                                           no_buffers);
}

TEST_CASE("owned block transfer rejects a null communicator", "[owned_block][MPI]") {
  const pfc::Domain domain = pfc::domain::create(pfc::Int3{2, 2, 1});
  const pfc::Decomposition decomp =
      pfc::decomposition::create(domain, pfc::Int3{1, 1, 1});
  const pfc::Box3i box = pfc::decomposition::local_box(decomp, 0);
  std::vector<double> values(static_cast<std::size_t>(box.count()), 1.0);
  const pfc::decomposition::OwnedBlockView view{box, values};
  const std::span<const pfc::decomposition::OwnedBlockView> blocks(&view, 1);
  require_invalid(
      [&] {
        (void)pfc::decomposition::gather_owned_blocks(decomp, MPI_COMM_NULL, 0,
                                                      blocks);
      },
      "communicator is null");
}

TEST_CASE("owned block transfer rejects a root outside the communicator",
          "[owned_block][MPI]") {
  const int nprocs = world_size();
  const pfc::Domain domain = pfc::domain::create(pfc::Int3{2 * nprocs + 1, 2, 1});
  const pfc::Decomposition decomp =
      pfc::decomposition::create(domain, pfc::Int3{nprocs, 1, 1});
  const pfc::Box3i box = pfc::decomposition::local_box(decomp, world_rank());
  std::vector<double> values(static_cast<std::size_t>(box.count()), 1.0);
  const pfc::decomposition::OwnedBlockView view{box, values};
  const std::span<const pfc::decomposition::OwnedBlockView> blocks(&view, 1);
  require_invalid(
      [&] {
        (void)pfc::decomposition::gather_owned_blocks(decomp, MPI_COMM_WORLD, -1,
                                                      blocks);
      },
      "root is outside the communicator");
}

TEST_CASE("owned block transfer rejects a span shorter than its box",
          "[owned_block][MPI]") {
  const int nprocs = world_size();
  const pfc::Domain domain = pfc::domain::create(pfc::Int3{2 * nprocs + 1, 2, 1});
  const pfc::Decomposition decomp =
      pfc::decomposition::create(domain, pfc::Int3{nprocs, 1, 1});
  const pfc::Box3i box = pfc::decomposition::local_box(decomp, world_rank());
  const pfc::decomposition::OwnedBlockView view{box, {}};
  const std::span<const pfc::decomposition::OwnedBlockView> blocks(&view, 1);
  require_invalid(
      [&] {
        (void)pfc::decomposition::gather_owned_blocks(decomp, MPI_COMM_WORLD, 0,
                                                      blocks);
      },
      "block length");
}

TEST_CASE(
    "owned block transfer rejects a decomposition smaller than the communicator",
    "[owned_block][MPI]") {
  const int nprocs = world_size();
  if (nprocs == 1) {
    return;
  }
  const pfc::Domain domain = pfc::domain::create(pfc::Int3{4, 2, 1});
  const pfc::Decomposition decomp =
      pfc::decomposition::create(domain, pfc::Int3{1, 1, 1});
  const pfc::Box3i box = pfc::decomposition::local_box(decomp, 0);
  std::vector<double> values(static_cast<std::size_t>(box.count()), 1.0);
  const pfc::decomposition::OwnedBlockView view{box, values};
  const std::span<const pfc::decomposition::OwnedBlockView> blocks(&view, 1);
  require_invalid(
      [&] {
        (void)pfc::decomposition::gather_owned_blocks(decomp, MPI_COMM_WORLD, 0,
                                                      blocks);
      },
      "rank count");
}

TEST_CASE("owned block scatter rejects a global buffer on a non-root rank",
          "[owned_block][MPI]") {
  const int nprocs = world_size();
  if (nprocs == 1) {
    return;
  }
  const int rank = world_rank();
  const pfc::Int3 shape{2 * nprocs + 1, 2, 1};
  const pfc::Domain domain = pfc::domain::create(shape);
  const pfc::Decomposition decomp =
      pfc::decomposition::create(domain, pfc::Int3{nprocs, 1, 1});
  const pfc::Box3i box = pfc::decomposition::local_box(decomp, rank);
  std::vector<double> local(static_cast<std::size_t>(box.count()), 0.0);
  pfc::decomposition::OwnedBlockBuffer buffer{box, local};
  const std::span<pfc::decomposition::OwnedBlockBuffer> blocks(&buffer, 1);
  std::vector<std::vector<double>> global(1);
  if (rank == 0) {
    global[0].assign(static_cast<std::size_t>(shape[0]) * shape[1] * shape[2], 0.0);
  }
  const char *fragment = rank == 0 ? "peer rank" : "not empty off the root";
  require_invalid(
      [&] {
        pfc::decomposition::scatter_owned_blocks(decomp, MPI_COMM_WORLD, 0, global,
                                                 blocks);
      },
      fragment);
}
