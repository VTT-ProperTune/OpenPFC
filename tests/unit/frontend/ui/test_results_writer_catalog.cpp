// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/frontend/ui/results_writer_catalog.hpp>

TEST_CASE("builtin results writer catalog supports binary", "[ui][results_writer]") {
  pfc::ui::ResultsWriterCatalog cat = pfc::ui::make_builtin_results_writer_catalog();
  REQUIRE(cat.has_type("binary"));
  REQUIRE(cat.has_type("vtk"));
#ifdef OPENPFC_HAS_HDF5
  REQUIRE(cat.has_type("hdf5"));
#endif
  auto types = cat.registered_writer_types();
  REQUIRE_FALSE(types.empty());
  auto w = cat.create_writer("binary", "test_output.bin", MPI_COMM_SELF);
  REQUIRE(w != nullptr);
  auto vtk = cat.create_writer("vtk", "test_output_%04d.vti", MPI_COMM_SELF);
  REQUIRE(vtk != nullptr);
}

TEST_CASE("unknown writer type is a hard error", "[ui][results_writer]") {
  pfc::ui::ResultsWriterCatalog cat = pfc::ui::make_builtin_results_writer_catalog();
  REQUIRE_THROWS_AS(cat.create_writer("unknown_format", "out.bin", MPI_COMM_SELF),
                    std::invalid_argument);
  try {
    (void)cat.create_writer("unknown_format", "out.bin", MPI_COMM_SELF, "psi");
    FAIL("expected invalid_argument");
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    REQUIRE(msg.find("unknown_format") != std::string::npos);
    REQUIRE(msg.find("psi") != std::string::npos);
    REQUIRE(msg.find("binary") != std::string::npos);
  }
}

TEST_CASE("custom writer type can be registered", "[ui][results_writer]") {
  pfc::ui::ResultsWriterCatalog cat = pfc::ui::make_builtin_results_writer_catalog();
  cat.register_writer("raw", [](std::string path, MPI_Comm comm) {
    return std::make_unique<pfc::BinaryWriter>(std::move(path), comm);
  });
  REQUIRE(cat.has_type("raw"));
  auto w = cat.create_writer("raw", "test_raw.bin", MPI_COMM_SELF);
  REQUIRE(w != nullptr);
}
