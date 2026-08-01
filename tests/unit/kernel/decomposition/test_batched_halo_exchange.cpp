// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/batched_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;
using namespace pfc::types;
using namespace pfc::communication;

TEMPLATE_TEST_CASE("BatchedHaloExchange construction", "[halo][batched]", double,
                   float) {
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  SECTION("Construction with Box3i and Domain") {
    const auto domain =
        domain::create(GridSize{{16, 16, 16}}, PhysicalOrigin{{0.0, 0.0, 0.0}},
                       GridSpacing{{1.0, 1.0, 1.0}});

    const auto decomp = decomposition::create(domain, nproc);
    const Box3i local_box = decomposition::local_box(decomp, rank);

    const int halo_width = 2;
    const std::size_t n_fields = 3;

    REQUIRE_NOTHROW(BatchedHaloExchange<TestType>(
        local_box, domain, decomp, rank, halo_width, MPI_COMM_WORLD, n_fields));
  }

  SECTION("Construction rejects zero fields") {
    const auto domain =
        domain::create(GridSize{{16, 16, 16}}, PhysicalOrigin{{0.0, 0.0, 0.0}},
                       GridSpacing{{1.0, 1.0, 1.0}});

    const auto decomp = decomposition::create(domain, nproc);
    const Box3i local_box = decomposition::local_box(decomp, rank);

    REQUIRE_THROWS_AS(BatchedHaloExchange<TestType>(local_box, domain, decomp, rank,
                                                    2, MPI_COMM_WORLD, 0),
                      std::invalid_argument);
  }

  SECTION("Construction supports direction sets") {
    const auto domain =
        domain::create(GridSize{{16, 16, 16}}, PhysicalOrigin{{0.0, 0.0, 0.0}},
                       GridSpacing{{1.0, 1.0, 1.0}});

    const auto decomp = decomposition::create(domain, nproc);
    const Box3i local_box = decomposition::local_box(decomp, rank);

    const std::size_t n_fields = 2;

    REQUIRE_NOTHROW(BatchedHaloExchange<TestType>(local_box, domain, decomp, rank, 2,
                                                  MPI_COMM_WORLD, n_fields, 0,
                                                  halo::presets::Axes2D()));
  }
}

TEST_CASE("BatchedHaloExchange tag layout correctness", "[halo][batched]") {
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const auto domain =
      domain::create(GridSize{{16, 16, 16}}, PhysicalOrigin{{0.0, 0.0, 0.0}},
                     GridSpacing{{1.0, 1.0, 1.0}});

  const auto decomp = decomposition::create(domain, nproc);
  const Box3i local_box = decomposition::local_box(decomp, rank);

  const int halo_width = 1;
  const int base_tag = 100;
  const std::size_t n_fields = 3;

  BatchedHaloExchange<double> halo(local_box, domain, decomp, rank, halo_width,
                                   MPI_COMM_WORLD, n_fields, base_tag);

  SECTION("Correct field count") { REQUIRE(halo.n_fields() == n_fields); }

  SECTION("Direction set accessibility") {
    const auto &dirs = halo.direction_set();
    REQUIRE(dirs.contains(Int3{1, 0, 0}));
    REQUIRE(dirs.contains(Int3{-1, 0, 0}));
    REQUIRE(dirs.contains(Int3{0, 1, 0}));
    REQUIRE(dirs.contains(Int3{0, -1, 0}));
    REQUIRE(dirs.contains(Int3{0, 0, 1}));
    REQUIRE(dirs.contains(Int3{0, 0, -1}));
  }

  SECTION("Active faces count") {
    // Single rank should have no active faces (all are self)
    REQUIRE(halo.num_active_faces() == 0);
  }
}

TEST_CASE("BatchedHaloExchange multi-field exchange", "[halo][batched][mpi]") {
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const auto domain =
      domain::create(GridSize{{32, 32, 32}}, PhysicalOrigin{{0.0, 0.0, 0.0}},
                     GridSpacing{{1.0, 1.0, 1.0}});

  const auto decomp = decomposition::create(domain, nproc);
  const Box3i local_box = decomposition::local_box(decomp, rank);

  const int halo_width = 2;
  const std::size_t n_fields = 3;

  BatchedHaloExchange<double> halo(local_box, domain, decomp, rank, halo_width,
                                   MPI_COMM_WORLD, n_fields);

  SECTION("Field count validation") {
    std::vector<double *> wrong_count(2); // Should be 3
    REQUIRE_THROWS_AS(halo.exchange_halos(wrong_count), std::invalid_argument);
  }

  SECTION("Successful exchange with correct field count") {
    // Create padded fields matching the decomposition
    std::vector<double *> fields;
    for (std::size_t i = 0; i < n_fields; ++i) {
      auto field = data::field_from_subdomain<double>(decomp, rank, halo_width);
      // Initialize with unique values per field for debugging
      const double value = static_cast<double>(i + 1);
      field.apply([value](double, double, double) { return value; });
      fields.push_back(field.data());
    }

    REQUIRE_NOTHROW(halo.exchange_halos(fields));
  }
}

TEST_CASE("BatchedHaloExchange vs PaddedHaloExchanger compatibility",
          "[halo][batched][mpi]") {
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const auto domain =
      domain::create(GridSize{{32, 32, 32}}, PhysicalOrigin{{0.0, 0.0, 0.0}},
                     GridSpacing{{1.0, 1.0, 1.0}});

  const auto decomp = decomposition::create(domain, nproc);
  const int halo_width = 2;

  // Create fields for testing
  auto field1 = data::field_from_subdomain<double>(decomp, rank, halo_width);
  auto field2 = data::field_from_subdomain<double>(decomp, rank, halo_width);

  // Initialize with known patterns
  field1.apply([](double x, double, double) { return x; });
  field2.apply([](double, double y, double) { return y; });

  SECTION("Both interfaces handle single rank correctly") {
    // PaddedHaloExchanger reference
    PaddedHaloExchanger<double> single_halo(field1, decomp, rank, MPI_COMM_WORLD);
    REQUIRE_NOTHROW(single_halo.exchange_halos(field1.data(), field1.size()));

    // BatchedHaloExchange equivalent
    const std::size_t n_fields = 2;
    BatchedHaloExchange<double> batched_halo(field1.bounds(), domain, decomp, rank,
                                             halo_width, MPI_COMM_WORLD, n_fields);

    std::vector<double *> batched_fields = {field1.data(), field2.data()};
    REQUIRE_NOTHROW(batched_halo.exchange_halos(batched_fields));
  }
}

TEST_CASE("BatchedHaloExchange direction set integration", "[halo][batched]") {
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const auto domain =
      domain::create(GridSize{{16, 32, 16}}, PhysicalOrigin{{0.0, 0.0, 0.0}},
                     GridSpacing{{1.0, 1.0, 1.0}});

  const auto decomp = decomposition::create(domain, nproc);
  const Box3i local_box = decomposition::local_box(decomp, rank);

  const int halo_width = 1;
  const std::size_t n_fields = 2;

  SECTION("2D direction set (X-Y plane)") {
    BatchedHaloExchange<double> halo(local_box, domain, decomp, rank, halo_width,
                                     MPI_COMM_WORLD, n_fields, 0,
                                     halo::presets::Axes2D());

    const auto &dirs = halo.direction_set();
    REQUIRE(dirs.contains(Int3{1, 0, 0}));
    REQUIRE(dirs.contains(Int3{-1, 0, 0}));
    REQUIRE(dirs.contains(Int3{0, 1, 0}));
    REQUIRE(dirs.contains(Int3{0, -1, 0}));
    REQUIRE_FALSE(dirs.contains(Int3{0, 0, 1}));
    REQUIRE_FALSE(dirs.contains(Int3{0, 0, -1}));
  }

  SECTION("1D line direction set") {
    BatchedHaloExchange<double> halo(
        local_box, domain, decomp, rank, halo_width, MPI_COMM_WORLD, n_fields, 0,
        halo::HaloDirectionSet{{{1, 0, 0}, {-1, 0, 0}}});

    const auto &dirs = halo.direction_set();
    REQUIRE(dirs.contains(Int3{1, 0, 0}));
    REQUIRE(dirs.contains(Int3{-1, 0, 0}));
    REQUIRE_FALSE(dirs.contains(Int3{0, 1, 0}));
    REQUIRE_FALSE(dirs.contains(Int3{0, -1, 0}));
    REQUIRE_FALSE(dirs.contains(Int3{0, 0, 1}));
    REQUIRE_FALSE(dirs.contains(Int3{0, 0, -1}));
  }
}
