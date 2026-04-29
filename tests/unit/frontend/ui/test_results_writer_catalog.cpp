// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/frontend/ui/results_writer_catalog.hpp>

TEST_CASE("builtin results writer catalog supports binary", "[ui][results_writer]") {
  pfc::ui::ResultsWriterCatalog cat = pfc::ui::make_builtin_results_writer_catalog();
  REQUIRE(cat.has_type("binary"));
  auto w = cat.try_create("binary", "test_output.bin", MPI_COMM_SELF);
  REQUIRE(w.has_value());
  REQUIRE(*w != nullptr);
}

TEST_CASE("unknown writer type yields nullopt", "[ui][results_writer]") {
  pfc::ui::ResultsWriterCatalog cat = pfc::ui::make_builtin_results_writer_catalog();
  auto w = cat.try_create("unknown_format", "out.bin", MPI_COMM_SELF);
  REQUIRE_FALSE(w.has_value());
}

TEST_CASE("custom writer type can be registered", "[ui][results_writer]") {
  pfc::ui::ResultsWriterCatalog cat = pfc::ui::make_builtin_results_writer_catalog();
  cat.register_writer_type("raw", [](std::string path, MPI_Comm comm) {
    return std::make_unique<pfc::BinaryWriter>(std::move(path), comm);
  });
  REQUIRE(cat.has_type("raw"));
  auto w = cat.try_create("raw", "test_raw.bin", MPI_COMM_SELF);
  REQUIRE(w.has_value());
  REQUIRE(*w != nullptr);
}
