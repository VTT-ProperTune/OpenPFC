// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;
using Catch::Approx;

namespace {

// Helper to compute expected geometry from decomposition context
struct FieldGeometry {
  std::array<int, 3> owned_core;  // local_owned from decomposition
  std::array<int, 3> with_halo;    // owned_core + 2*halo_extents
  int halo_depth;                 // from context
};

FieldGeometry compute_expected(const decomposition::Decomposition& decomp, int rank,
                               const std::array<int,3>& halo) {
  auto local_box = decomposition::local_box(decomp, rank);
  return {
    {local_box.size[0], local_box.size[1], local_box.size[2]},
    {local_box.size[0] + 2*halo[0], local_box.size[1] + 2*halo[1], local_box.size[2] + 2*halo[2]},
    halo[0]
  };
}

// Test geometry invariants for field_from_subdomain
template <typename T>
void verify_field_from_subdomain_geometry(const decomposition::Decomposition& decomp,
                                          int rank, int halo) {
  auto field = pfc::data::field_from_subdomain<T>(decomp, rank, halo);

  // Compute expected geometry
  auto geom = compute_expected(decomp, rank, {halo, halo, halo});

  // Verify total field size includes halo padding
  REQUIRE(field.size() ==
          static_cast<std::size_t>(geom.with_halo[0]) *
          static_cast<std::size_t>(geom.with_halo[1]) *
          static_cast<std::size_t>(geom.with_halo[2]));

  // Verify halo extents
  REQUIRE(field.storage_halo() == halo);
  REQUIRE(field.halo_width() == halo);

  // Verify owned core dimensions match decomposition
  REQUIRE(field.local_size()[0] == geom.owned_core[0]);
  REQUIRE(field.local_size()[1] == geom.owned_core[1]);
  REQUIRE(field.local_size()[2] == geom.owned_core[2]);

  // Verify padded extents
  REQUIRE(field.padded_extent(0) == geom.with_halo[0]);
  REQUIRE(field.padded_extent(1) == geom.with_halo[1]);
  REQUIRE(field.padded_extent(2) == geom.with_halo[2]);

  // Verify indexing consistency across the full padded range
  for (int k = -halo; k < geom.owned_core[2] + halo; ++k) {
    for (int j = -halo; j < geom.owned_core[1] + halo; ++j) {
      for (int i = -halo; i < geom.owned_core[0] + halo; ++i) {
        // idx should be monotonic and unique for each cell
        std::size_t idx = field.idx(i, j, k);
        REQUIRE(idx < field.size());
      }
    }
  }

  // Verify coordinate queries for owned cells
  for (int k = 0; k < geom.owned_core[2]; ++k) {
    for (int j = 0; j < geom.owned_core[1]; ++j) {
      for (int i = 0; i < geom.owned_core[0]; ++i) {
        auto global = field.global(i, j, k);
        auto coords = field.coords(i, j, k);

        // Global index should be consistent with decomposition
        REQUIRE(global[0] >= 0);
        REQUIRE(global[1] >= 0);
        REQUIRE(global[2] >= 0);

        // Physical coordinates should be reasonable
        REQUIRE(coords[0] == Approx(
          field.origin()[0] + static_cast<double>(global[0]) * field.spacing()[0]));
        REQUIRE(coords[1] == Approx(
          field.origin()[1] + static_cast<double>(global[1]) * field.spacing()[1]));
        REQUIRE(coords[2] == Approx(
          field.origin()[2] + static_cast<double>(global[2]) * field.spacing()[2]));
      }
    }
  }

  // Verify data access works identically across full padded range
  for (int k = -halo; k < geom.owned_core[2] + halo; ++k) {
    for (int j = -halo; j < geom.owned_core[1] + halo; ++j) {
      for (int i = -halo; i < geom.owned_core[0] + halo; ++i) {
        const T test_value = static_cast<T>(
          i + geom.owned_core[0] * (j + geom.owned_core[1] * k));
        field(i, j, k) = test_value;
        REQUIRE(field(i, j, k) == test_value);
      }
    }
  }
}

// Test geometry invariants for field_from_subdomain_unpadded
template <typename T>
void verify_field_from_subdomain_unpadded_geometry(const decomposition::Decomposition& decomp,
                                                   int rank, int iteration_halo) {
  auto field = pfc::data::field_from_subdomain_unpadded<T>(decomp, rank, iteration_halo);

  // Compute expected geometry (no storage halo)
  auto geom = compute_expected(decomp, rank, {0, 0, 0});

  // Verify total field size is unpadded
  REQUIRE(field.size() ==
          static_cast<std::size_t>(geom.owned_core[0]) *
          static_cast<std::size_t>(geom.owned_core[1]) *
          static_cast<std::size_t>(geom.owned_core[2]));

  // Verify halo properties
  REQUIRE(field.storage_halo() == 0);
  REQUIRE(field.halo_width() == iteration_halo);

  // Verify owned core dimensions
  REQUIRE(field.local_size()[0] == geom.owned_core[0]);
  REQUIRE(field.local_size()[1] == geom.owned_core[1]);
  REQUIRE(field.local_size()[2] == geom.owned_core[2]);

  // Verify padded extents match owned (no padding)
  REQUIRE(field.padded_extent(0) == geom.owned_core[0]);
  REQUIRE(field.padded_extent(1) == geom.owned_core[1]);
  REQUIRE(field.padded_extent(2) == geom.owned_core[2]);

  // Verify indexing consistency for owned cells only
  for (int k = 0; k < geom.owned_core[2]; ++k) {
    for (int j = 0; j < geom.owned_core[1]; ++j) {
      for (int i = 0; i < geom.owned_core[0]; ++i) {
        std::size_t idx = field.idx(i, j, k);
        REQUIRE(idx < field.size());
      }
    }
  }

  // Verify coordinate queries match padded version
  auto field_padded = pfc::data::field_from_subdomain<T>(decomp, rank, 0);
  for (int k = 0; k < geom.owned_core[2]; ++k) {
    for (int j = 0; j < geom.owned_core[1]; ++j) {
      for (int i = 0; i < geom.owned_core[0]; ++i) {
        REQUIRE(field.global(i, j, k) == field_padded.global(i, j, k));
        REQUIRE(field.coords(i, j, k)[0] == Approx(field_padded.coords(i, j, k)[0]));
        REQUIRE(field.coords(i, j, k)[1] == Approx(field_padded.coords(i, j, k)[1]));
        REQUIRE(field.coords(i, j, k)[2] == Approx(field_padded.coords(i, j, k)[2]));
      }
    }
  }

  // Verify data access works only within owned range
  for (int k = 0; k < geom.owned_core[2]; ++k) {
    for (int j = 0; j < geom.owned_core[1]; ++j) {
      for (int i = 0; i < geom.owned_core[0]; ++i) {
        const T test_value = static_cast<T>(i + j + k);
        field(i, j, k) = test_value;
        REQUIRE(field(i, j, k) == test_value);
      }
    }
  }
}

// Test geometry invariants for field_from_inbox
template <typename T>
void verify_field_from_inbox_geometry(const pfc::Domain& domain,
                                      const pfc::Box3i& inbox) {
  auto field = pfc::data::field_from_inbox<T>(domain, inbox);

  // Compute expected geometry
  std::array<int, 3> owned_core = {inbox.size[0], inbox.size[1], inbox.size[2]};
  int halo = 0;

  // Verify total field size is unpadded
  REQUIRE(field.size() ==
          static_cast<std::size_t>(owned_core[0]) *
          static_cast<std::size_t>(owned_core[1]) *
          static_cast<std::size_t>(owned_core[2]));

  // Verify no halo padding
  REQUIRE(field.storage_halo() == halo);
  REQUIRE(field.halo_width() == halo);

  // Verify owned core dimensions match inbox
  REQUIRE(field.local_size()[0] == owned_core[0]);
  REQUIRE(field.local_size()[1] == owned_core[1]);
  REQUIRE(field.local_size()[2] == owned_core[2]);

  // Verify box matches inbox
  REQUIRE(field.box().low == inbox.low);
  REQUIRE(field.box().high == inbox.high);
  REQUIRE(field.box().size == inbox.size);

  // Verify domain is preserved
  REQUIRE(pfc::domain::get_origin(field.domain()) == pfc::domain::get_origin(domain));
  REQUIRE(pfc::domain::get_spacing(field.domain()) == pfc::domain::get_spacing(domain));
  REQUIRE(pfc::domain::get_size(field.domain()) == pfc::domain::get_size(domain));

  // Verify padded extents match owned (no padding)
  REQUIRE(field.padded_extent(0) == owned_core[0]);
  REQUIRE(field.padded_extent(1) == owned_core[1]);
  REQUIRE(field.padded_extent(2) == owned_core[2]);

  // Verify indexing consistency
  for (int k = 0; k < owned_core[2]; ++k) {
    for (int j = 0; j < owned_core[1]; ++j) {
      for (int i = 0; i < owned_core[0]; ++i) {
        std::size_t idx = field.idx(i, j, k);
        REQUIRE(idx < field.size());

        // Verify global index is consistent with inbox
        auto global = field.global(i, j, k);
        REQUIRE(global[0] == inbox.low[0] + i);
        REQUIRE(global[1] == inbox.low[1] + j);
        REQUIRE(global[2] == inbox.low[2] + k);
      }
    }
  }

  // Verify data access works
  for (int k = 0; k < owned_core[2]; ++k) {
    for (int j = 0; j < owned_core[1]; ++j) {
      for (int i = 0; i < owned_core[0]; ++i) {
        const T test_value = static_cast<T>(i + j + k);
        field(i, j, k) = test_value;
        REQUIRE(field(i, j, k) == test_value);
      }
    }
  }
}

} // namespace

TEST_CASE("field_from_subdomain: basic geometry verification", "[field_factory][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  auto domain = domain::create({nx, ny, nz});
  auto decomp = decomposition::create(domain, 1);

  for (int halo : {0, 1, 2}) {
    verify_field_from_subdomain_geometry<double>(decomp, 0, halo);
    verify_field_from_subdomain_geometry<float>(decomp, 0, halo);
  }
}

TEST_CASE("field_from_subdomain: multiple ranks", "[field_factory][unit]") {
  const int nx = 12, ny = 8, nz = 6;
  auto domain = domain::create({nx, ny, nz});
  auto decomp = decomposition::create(domain, 8);

  for (int rank = 0; rank < 8; ++rank) {
    verify_field_from_subdomain_geometry<double>(decomp, rank, 1);
    verify_field_from_subdomain_geometry<double>(decomp, rank, 2);
  }
}

TEST_CASE("field_from_subdomain: rejects negative halo", "[field_factory][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  auto domain = domain::create({nx, ny, nz});
  auto decomp = decomposition::create(domain, 1);

  REQUIRE_THROWS_AS(pfc::data::field_from_subdomain<double>(decomp, 0, -1),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::data::field_from_subdomain<double>(decomp, 0, -10),
                    std::invalid_argument);
}

TEST_CASE("field_from_subdomain: geometry matches decomposition", "[field_factory][unit]") {
  const int nx = 16, ny = 12, nz = 8;
  auto domain = domain::create({nx, ny, nz});
  auto decomp = decomposition::create(domain, 4);

  for (int rank = 0; rank < 4; ++rank) {
    auto field = pfc::data::field_from_subdomain<double>(decomp, rank, 2);

    // Verify domain geometry is preserved
    const auto& field_domain = field.domain();
    const auto decomp_domain = decomposition::domain(decomp);
    REQUIRE(pfc::domain::get_origin(field_domain) == pfc::domain::get_origin(decomp_domain));
    REQUIRE(pfc::domain::get_spacing(field_domain) == pfc::domain::get_spacing(decomp_domain));
    REQUIRE(pfc::domain::get_size(field_domain) == pfc::domain::get_size(decomp_domain));

    // Verify local box matches decomposition
    const auto& field_box = field.box();
    const auto local_box = decomposition::local_box(decomp, rank);
    REQUIRE(field_box.low == local_box.low);
    REQUIRE(field_box.high == local_box.high);
    REQUIRE(field_box.size == local_box.size);

    // Verify halo width matches
    REQUIRE(field.halo_width() == 2);
  }
}

TEST_CASE("field_from_subdomain: creates valid field with varying sizes",
          "[field_factory][unit]") {
  auto domain = domain::create({4, 4, 4});
  auto decomp = decomposition::create(domain, 1);

  for (int halo = 0; halo <= 2; ++halo) {
    auto field = pfc::data::field_from_subdomain<double>(decomp, 0, halo);

    // Verify field is properly constructed
    REQUIRE(field.size() > 0);
    REQUIRE(field.local_size() == Int3{4, 4, 4});
    REQUIRE(field.halo_width() == halo);

    // Verify we can write and read values
    field.apply([](double x, double y, double z) { return x + y + z; });
    REQUIRE(field(0, 0, 0) == Approx(0.0)); // origin
  }
}

TEST_CASE("field_from_subdomain_unpadded: basic geometry verification", "[field_factory][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  auto domain = domain::create({nx, ny, nz});
  auto decomp = decomposition::create(domain, 1);

  for (int iteration_halo : {0, 1, 2}) {
    verify_field_from_subdomain_unpadded_geometry<double>(decomp, 0, iteration_halo);
    verify_field_from_subdomain_unpadded_geometry<float>(decomp, 0, iteration_halo);
  }
}

TEST_CASE("field_from_subdomain_unpadded: multiple ranks", "[field_factory][unit]") {
  const int nx = 12, ny = 8, nz = 6;
  auto domain = domain::create({nx, ny, nz});
  auto decomp = decomposition::create(domain, 8);

  for (int rank = 0; rank < 8; ++rank) {
    verify_field_from_subdomain_unpadded_geometry<double>(decomp, rank, 1);
    verify_field_from_subdomain_unpadded_geometry<double>(decomp, rank, 2);
  }
}

TEST_CASE("field_from_subdomain_unpadded: rejects negative iteration halo",
          "[field_factory][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  auto domain = domain::create({nx, ny, nz});
  auto decomp = decomposition::create(domain, 1);

  REQUIRE_THROWS_AS(pfc::data::field_from_subdomain_unpadded<double>(decomp, 0, -1),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::data::field_from_subdomain_unpadded<double>(decomp, 0, -10),
                    std::invalid_argument);
}

TEST_CASE("field_from_subdomain_unpadded: geometry matches decomposition",
          "[field_factory][unit]") {
  const int nx = 16, ny = 12, nz = 8;
  auto domain = domain::create({nx, ny, nz});
  auto decomp = decomposition::create(domain, 4);

  for (int rank = 0; rank < 4; ++rank) {
    auto field = pfc::data::field_from_subdomain_unpadded<double>(decomp, rank, 2);

    // Verify domain geometry is preserved
    const auto& field_domain = field.domain();
    const auto decomp_domain = decomposition::domain(decomp);
    REQUIRE(pfc::domain::get_origin(field_domain) == pfc::domain::get_origin(decomp_domain));
    REQUIRE(pfc::domain::get_spacing(field_domain) == pfc::domain::get_spacing(decomp_domain));
    REQUIRE(pfc::domain::get_size(field_domain) == pfc::domain::get_size(decomp_domain));

    // Verify local box matches decomposition
    const auto& field_box = field.box();
    const auto local_box = decomposition::local_box(decomp, rank);
    REQUIRE(field_box.low == local_box.low);
    REQUIRE(field_box.high == local_box.high);
    REQUIRE(field_box.size == local_box.size);

    // Verify storage and iteration halo are decoupled
    REQUIRE(field.storage_halo() == 0);
    REQUIRE(field.halo_width() == 2);
  }
}

TEST_CASE("field_from_inbox: basic geometry verification", "[field_factory][unit]") {
  auto domain = domain::create({8, 6, 4});

  // Test various inbox configurations
  std::vector<pfc::Box3i> inboxes = {
    Box3i::from_bounds({0, 0, 0}, {3, 2, 1}),  // size {4, 3, 2}
    Box3i::from_bounds({4, 3, 2}, {7, 5, 3}),  // size {4, 3, 2}
    Box3i::from_bounds({0, 0, 0}, {7, 5, 3}),  // size {8, 6, 4} full domain
  };

  for (const auto& inbox : inboxes) {
    verify_field_from_inbox_geometry<double>(domain, inbox);
    verify_field_from_inbox_geometry<float>(domain, inbox);
  }
}

TEST_CASE("field_from_inbox: with spacing and origin", "[field_factory][unit]") {
  const Real3 spacing{1.5, 2.0, 2.5};
  const Real3 origin{1.0, 1.0, 1.0};
  auto domain = pfc::domain::create(GridSize({8, 6, 4}), PhysicalOrigin(origin), GridSpacing(spacing));

  Box3i inbox = Box3i::from_bounds({2, 1, 1}, {5, 4, 2}); // size will be {4, 4, 2}
  auto field = pfc::data::field_from_inbox<double>(domain, inbox);

  // Verify geometry preservation
  REQUIRE(field.spacing() == spacing);
  REQUIRE(field.origin() == origin);

  // Verify coordinates are correctly computed
  int i = 0, j = 0, k = 0;
  auto coords = field.coords(i, j, k);
  REQUIRE(coords[0] == Approx(origin[0] + static_cast<double>(inbox.low[0] + i) * spacing[0]));
  REQUIRE(coords[1] == Approx(origin[1] + static_cast<double>(inbox.low[1] + j) * spacing[1]));
  REQUIRE(coords[2] == Approx(origin[2] + static_cast<double>(inbox.low[2] + k) * spacing[2]));
}

TEST_CASE("field_from_subdomain: dual-construct consistency", "[field_factory][unit]") {
  const int nx = 12, ny = 8, nz = 6;
  auto domain = domain::create({nx, ny, nz});
  auto decomp = decomposition::create(domain, 4);

  for (int halo : {0, 1, 2}) {
    for (int rank = 0; rank < 4; ++rank) {
      // Dual-construct: two fields from same context
      auto field1 = pfc::data::field_from_subdomain<double>(decomp, rank, halo);
      auto field2 = pfc::data::field_from_subdomain<double>(decomp, rank, halo);

      // Verify geometry matches exactly
      REQUIRE(field1.local_size() == field2.local_size());
      REQUIRE(field1.halo_width() == field2.halo_width());
      REQUIRE(field1.storage_halo() == field2.storage_halo());
      REQUIRE(field1.padded_extent(0) == field2.padded_extent(0));
      REQUIRE(field1.padded_extent(1) == field2.padded_extent(1));
      REQUIRE(field1.padded_extent(2) == field2.padded_extent(2));

      // Verify indexing matches
      for (int k = -halo; k < field1.local_size()[2] + halo; ++k) {
        for (int j = -halo; j < field1.local_size()[1] + halo; ++j) {
          for (int i = -halo; i < field1.local_size()[0] + halo; ++i) {
            REQUIRE(field1.idx(i, j, k) == field2.idx(i, j, k));
          }
        }
      }

      // Verify coordinate queries match
      for (int k = 0; k < field1.local_size()[2]; ++k) {
        for (int j = 0; j < field1.local_size()[1]; ++j) {
          for (int i = 0; i < field1.local_size()[0]; ++i) {
            REQUIRE(field1.global(i, j, k) == field2.global(i, j, k));
            REQUIRE(field1.coords(i, j, k)[0] == Approx(field2.coords(i, j, k)[0]));
            REQUIRE(field1.coords(i, j, k)[1] == Approx(field2.coords(i, j, k)[1]));
            REQUIRE(field1.coords(i, j, k)[2] == Approx(field2.coords(i, j, k)[2]));
          }
        }
      }
    }
  }
}

TEST_CASE("field_from_subdomain: vs unpadded consistency (halo=0)", "[field_factory][unit]") {
  const int nx = 12, ny = 8, nz = 6;
  auto domain = domain::create({nx, ny, nz});
  auto decomp = decomposition::create(domain, 4);

  for (int rank = 0; rank < 4; ++rank) {
    // When halo=0, padded and unpadded should have identical geometry
    auto padded = pfc::data::field_from_subdomain<double>(decomp, rank, 0);
    auto unpadded = pfc::data::field_from_subdomain_unpadded<double>(decomp, rank, 0);

    // Verify storage is identical
    REQUIRE(padded.size() == unpadded.size());
    REQUIRE(padded.local_size() == unpadded.local_size());
    REQUIRE(padded.halo_width() == unpadded.halo_width());
    REQUIRE(padded.storage_halo() == unpadded.storage_halo());

    // Verify indexing matches
    for (int k = 0; k < padded.local_size()[2]; ++k) {
      for (int j = 0; j < padded.local_size()[1]; ++j) {
        for (int i = 0; i < padded.local_size()[0]; ++i) {
          REQUIRE(padded.idx(i, j, k) == unpadded.idx(i, j, k));
        }
      }
    }

    // Verify coordinates match
    for (int k = 0; k < padded.local_size()[2]; ++k) {
      for (int j = 0; j < padded.local_size()[1]; ++j) {
        for (int i = 0; i < padded.local_size()[0]; ++i) {
          REQUIRE(padded.global(i, j, k) == unpadded.global(i, j, k));
          REQUIRE(padded.coords(i, j, k)[0] == Approx(unpadded.coords(i, j, k)[0]));
          REQUIRE(padded.coords(i, j, k)[1] == Approx(unpadded.coords(i, j, k)[1]));
          REQUIRE(padded.coords(i, j, k)[2] == Approx(unpadded.coords(i, j, k)[2]));
        }
      }
    }
  }
}
