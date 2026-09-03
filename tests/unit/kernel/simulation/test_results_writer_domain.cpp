// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>
#include <openpfc/kernel/simulation/results_writer_domain.hpp>

namespace {

class RecordingWriter : public pfc::ResultsWriter {
public:
  std::array<int, 3> global{};
  std::array<int, 3> local{};
  std::array<int, 3> offset{};
  bool set = false;

  void set_domain(const std::array<int, 3> &arr_global,
                  const std::array<int, 3> &arr_local,
                  const std::array<int, 3> &arr_offset) override {
    global = arr_global;
    local = arr_local;
    offset = arr_offset;
    set = true;
  }
  MPI_Status write(int, pfc::field::FieldView<double>) override { return MPI_Status{}; }
  MPI_Status write(int, pfc::field::FieldView<std::complex<double>>) override { return MPI_Status{}; }
};

} // namespace

TEST_CASE("apply_writer_domain uses Domain and owned Box3i",
          "[simulation][io][unit]") {
  auto domain = pfc::domain::create(pfc::GridSize({16, 8, 4}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  const pfc::Box3i owned = pfc::Box3i::from_bounds({4, 0, 0}, {11, 7, 3});
  RecordingWriter writer;
  pfc::apply_writer_domain(writer, domain, owned);
  REQUIRE(writer.set);
  REQUIRE(writer.global == std::array<int, 3>{16, 8, 4});
  REQUIRE(writer.local == std::array<int, 3>{8, 8, 4});
  REQUIRE(writer.offset == std::array<int, 3>{4, 0, 0});
}

TEST_CASE("apply_writer_domain uses Field geometry", "[simulation][io][unit]") {
  auto domain = pfc::domain::create(pfc::GridSize({8, 8, 8}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  const pfc::Box3i owned = pfc::Box3i::from_bounds({0, 0, 0}, {7, 7, 7});
  pfc::data::Field<double> field(domain, owned, 0);
  RecordingWriter writer;
  pfc::apply_writer_domain(writer, field);
  REQUIRE(writer.set);
  REQUIRE(writer.global == std::array<int, 3>{8, 8, 8});
  REQUIRE(writer.local == std::array<int, 3>{8, 8, 8});
  REQUIRE(writer.offset == std::array<int, 3>{0, 0, 0});
}
