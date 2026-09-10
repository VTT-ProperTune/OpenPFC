// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_field_snapshots.cpp
 * @brief `openpfc_apps/field_snapshots.hpp`: the standalone drivers' `fields[]`.
 *
 * The two science drivers that own their own `main()`
 * (`surface_diffusion_anisotropic`, `ehd_film_nonlinear`) get their field
 * output from this helper rather than from `SpectralETDSession`. The point of
 * these tests is the failure modes, not the happy path: a preset that
 * *believes* it is writing snapshots and silently is not would be invisible
 * until someone tried to render a figure from the missing files, which is
 * exactly how the report chapters lost their figures in the first place.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <string>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc_apps/field_snapshots.hpp>

using json = nlohmann::json;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  Catch::Session session;
  session.configData().rngSeed = 1u;
  session.configData().runOrder = Catch::TestRunOrder::Declared;
  const int cli = session.applyCommandLine(argc, argv);
  if (cli != 0) {
    MPI_Finalize();
    return cli;
  }
  const int result = session.run();
  MPI_Finalize();
  return result;
}

namespace {

/// A tiny host field, laid out the way the standalone drivers lay theirs out.
pfc::data::Field<double> make_field(const pfc::Domain &domain) {
  const auto size = pfc::domain::get_size(domain);
  const auto owned =
      pfc::Box3i::from_bounds({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1});
  return pfc::data::Field<double>(domain, owned, 0);
}

pfc::Domain small_domain() {
  return pfc::domain::create(pfc::GridSize({4, 4, 1}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
}

} // namespace

TEST_CASE("field_snapshots: no fields[] means no writer and no files",
          "[apps][field_snapshots]") {
  const auto domain = small_domain();
  auto field = make_field(domain);

  // The pre-existing behaviour of both drivers, and what their test presets
  // still rely on: diagnostics only, no file-system traffic at all.
  REQUIRE(pfc::apps::make_field_snapshot_writer(json::object(), "h", field) ==
          nullptr);
  REQUIRE(pfc::apps::make_field_snapshot_writer(json{{"fields", json::array()}}, "h",
                                                field) == nullptr);

  // A null writer is a no-op rather than a crash, so a driver can call the
  // write unconditionally at every save.
  REQUIRE_NOTHROW(pfc::apps::write_field_snapshot(nullptr, 0, field));
}

TEST_CASE("field_snapshots: a malformed fields[] is rejected, not ignored",
          "[apps][field_snapshots]") {
  const auto domain = small_domain();
  auto field = make_field(domain);

  SECTION("wrong field name") {
    const json cfg = {{"fields", {{{"name", "psi"}, {"data", "out_%04d.vti"}}}}};
    REQUIRE_THROWS_AS(pfc::apps::make_field_snapshot_writer(cfg, "h", field),
                      std::invalid_argument);
  }
  SECTION("more entries than the driver has fields") {
    const json cfg = {{"fields",
                       {{{"name", "h"}, {"data", "a_%04d.vti"}},
                        {{"name", "h"}, {"data", "b_%04d.vti"}}}}};
    REQUIRE_THROWS_AS(pfc::apps::make_field_snapshot_writer(cfg, "h", field),
                      std::invalid_argument);
  }
  SECTION("an output format this helper cannot write") {
    const json cfg = {{"fields", {{{"name", "h"}, {"data", "out_%04d.bin"}}}}};
    REQUIRE_THROWS_AS(pfc::apps::make_field_snapshot_writer(cfg, "h", field),
                      std::invalid_argument);
  }
  SECTION("fields[] is not an array") {
    const json cfg = {{"fields", {{"name", "h"}}}};
    REQUIRE_THROWS_AS(pfc::apps::make_field_snapshot_writer(cfg, "h", field),
                      std::invalid_argument);
  }
}

TEST_CASE("field_snapshots: snapshots are indexed by save and land on disk",
          "[apps][field_snapshots]") {
  if (pfc::mpi::get_size() != 1) return; // single-rank helper check

  const auto dir = std::filesystem::temp_directory_path() /
                   ("openpfc_field_snapshots_" + std::to_string(::getpid()));
  std::filesystem::remove_all(dir);

  const auto domain = small_domain();
  auto field = make_field(domain);
  field.apply([](const pfc::Real3 &x) { return x[0] + 10.0 * x[1]; });

  // A `results/`-style path whose directory does not exist yet: the helper is
  // expected to create it, as the drivers' diagnostics CSV already does.
  const std::string pattern = (dir / "results" / "h_%04d.vti").string();
  const json cfg = {{"fields", {{{"name", "h"}, {"data", pattern}}}}};
  auto writer = pfc::apps::make_field_snapshot_writer(cfg, "h", field);
  REQUIRE(writer != nullptr);

  for (int i = 0; i < 3; ++i)
    pfc::apps::write_field_snapshot(writer.get(), i, field);

  // Save index, not step index: the first save is _0000.
  REQUIRE(std::filesystem::exists(dir / "results" / "h_0000.vti"));
  REQUIRE(std::filesystem::exists(dir / "results" / "h_0001.vti"));
  REQUIRE(std::filesystem::exists(dir / "results" / "h_0002.vti"));
  REQUIRE(std::filesystem::file_size(dir / "results" / "h_0000.vti") > 0);

  std::filesystem::remove_all(dir);
}
