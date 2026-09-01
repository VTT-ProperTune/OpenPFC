// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cmath>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include "22_external_coupling.hpp"

#include <openpfc/kernel/checkpoint/checkpoint_metadata.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/checkpoint_service.hpp>
#include <openpfc/kernel/simulation/coupling.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using pfc::sim::CheckpointService;

namespace {

struct TempCkpt {
  std::filesystem::path root;
  explicit TempCkpt(const char *tag) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    root = std::filesystem::temp_directory_path() / tag;
    if (rank == 0) {
      std::error_code ec;
      std::filesystem::remove_all(root, ec);
      std::filesystem::create_directories(root);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  ~TempCkpt() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::error_code ec;
      std::filesystem::remove_all(root, ec);
    }
  }
};

pfc::SimulationState make_state(int n, int nproc, int rank) {
  auto domain = pfc::domain::create(pfc::GridSize({n, n, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, nproc);
  auto field = pfc::data::field_from_subdomain<double>(decomp, rank, 0);
  field.apply([](double x, double y, double) { return x + 0.1 * y; });
  pfc::SimulationState state;
  state.add_field("u", std::move(field));
  return state;
}

void require_fields_equal(const pfc::data::Field<double> &a,
                          const pfc::data::Field<double> &b) {
  const auto sz = a.local_size();
  REQUIRE(sz == b.local_size());
  for (int k = 0; k < sz[2]; ++k) {
    for (int j = 0; j < sz[1]; ++j) {
      for (int i = 0; i < sz[0]; ++i) {
        REQUIRE(a(i, j, k) == b(i, j, k));
      }
    }
  }
}

} // namespace

TEST_CASE("from_json CheckpointMetadata round-trip", "[checkpoint][metadata]") {
  pfc::checkpoint::CheckpointMetadata meta{
      .format_version = 1,
      .accepted_time = 0.4,
      .accepted_increment = 4,
      .result_counter = 2,
      .domain =
          {
              .global_dimensions = {8, 8, 1},
              .physical_origin = {0.0, 0.0, 0.0},
              .grid_spacing = {1.0, 1.0, 1.0},
          },
      .method_identity = "euler",
      .fields = {"u"},
  };
  const auto j = pfc::checkpoint::to_json(meta);
  const auto back = pfc::checkpoint::from_json(j);
  REQUIRE(back.accepted_increment == 4);
  REQUIRE(back.method_identity == "euler");
  REQUIRE(back.fields == std::vector<std::string>{"u"});
  REQUIRE(back.result_counter == 2);
}

TEST_CASE("from_json rejects schema version mismatch", "[checkpoint][metadata]") {
  auto j = nlohmann::json{
      {"format_version", 99},
      {"accepted_time", 0.0},
      {"accepted_increment", 0},
      {"domain",
       {{"global_dimensions", {2, 2, 1}},
        {"physical_origin", {0, 0, 0}},
        {"grid_spacing", {1, 1, 1}}}},
      {"method_identity", "euler"},
  };
  REQUIRE_THROWS_WITH(pfc::checkpoint::from_json(j),
                      ContainsSubstring("schema version"));
}

TEST_CASE("checkpoint_config_from_json reads every/directory/restart_from",
          "[checkpoint][service]") {
  const auto cfg = pfc::sim::checkpoint_config_from_json(nlohmann::json{
      {"restart_from", "/tmp/ckpt/step_3"},
      {"checkpoint", {{"every", 10}, {"directory", "/tmp/ckpt"}}},
  });
  REQUIRE(cfg.every == 10);
  REQUIRE(cfg.directory == std::filesystem::path("/tmp/ckpt"));
  REQUIRE(cfg.restart_from == std::filesystem::path("/tmp/ckpt/step_3"));
}

TEST_CASE("CheckpointService save n=8 without evolve", "[checkpoint][service]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }
  TempCkpt tmp("openpfc_ckpt_n8");
  CheckpointService svc({.every = 0, .directory = tmp.root}, MPI_COMM_WORLD);
  auto state = make_state(8, 1, 0);
  pfc::Time time({0.0, 1.0, 0.1}, 0.0);
  time.next();
  svc.save(state, time);
  REQUIRE(std::filesystem::exists(svc.step_dir(1) / "metadata.json"));
}

TEST_CASE("CheckpointService save/load restart equivalence 1-rank",
          "[checkpoint][service]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }

  TempCkpt tmp("openpfc_ckpt_service_1");
  pfc::sim::CheckpointConfig cfg{.every = 0, .directory = tmp.root};
  CheckpointService svc(cfg, MPI_COMM_WORLD);

  auto state = make_state(8, 1, 0);
  pfc::Time time({0.0, 1.0, 0.1}, 0.0);
  auto &u = state.get_field<double>("u");
  for (int s = 0; s < 3; ++s) {
    time.next();
    u.apply([&](double x, double y, double) {
      return x + 0.1 * y + static_cast<double>(time.get_increment());
    });
  }
  svc.set_result_counter(7);
  svc.save(state, time);

  auto restored = make_state(8, 1, 0);
  pfc::Time time2({0.0, 1.0, 0.1}, 0.0);
  svc.load(restored, time2, svc.step_dir(3));
  REQUIRE(time2.get_increment() == 3);
  REQUIRE_THAT(time2.get_current(), WithinAbs(time.get_current(), 1e-12));
  REQUIRE(svc.result_counter() == 7);

  const auto &a = state.get_field<double>("u");
  const auto &b = restored.get_field<double>("u");
  require_fields_equal(a, b);

  for (int s = 0; s < 2; ++s) {
    time.next();
    time2.next();
    auto fill = [&](double x, double y, double) {
      return x + 0.1 * y + static_cast<double>(time.get_increment());
    };
    state.get_field<double>("u").apply(fill);
    restored.get_field<double>("u").apply(fill);
  }
  REQUIRE(state.get_field<double>("u")(2, 2, 0) ==
          restored.get_field<double>("u")(2, 2, 0));
}

TEST_CASE("CheckpointService maybe_save and restore_from_config",
          "[checkpoint][service]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }

  TempCkpt tmp("openpfc_ckpt_maybe");
  auto state = make_state(4, 1, 0);
  pfc::Time time({0.0, 1.0, 0.1}, 0.0);
  pfc::sim::CheckpointConfig cfg{.every = 2, .directory = tmp.root};
  CheckpointService svc(cfg, MPI_COMM_WORLD);
  REQUIRE_FALSE(svc.maybe_save(state, time));
  time.next();
  REQUIRE_FALSE(svc.maybe_save(state, time));
  time.next();
  REQUIRE(svc.maybe_save(state, time));
  REQUIRE(std::filesystem::exists(svc.step_dir(2) / "metadata.json"));

  auto restored = make_state(4, 1, 0);
  pfc::Time time2({0.0, 1.0, 0.1}, 0.0);
  pfc::sim::CheckpointConfig restart_cfg{.every = 0,
                                         .restart_from = svc.step_dir(2)};
  CheckpointService loader(restart_cfg, MPI_COMM_WORLD);
  loader.restore_from_config(restored, time2);
  REQUIRE(time2.get_increment() == 2);
  require_fields_equal(state.get_field<double>("u"),
                       restored.get_field<double>("u"));
}

TEST_CASE("CheckpointService identity mismatch method", "[checkpoint][service]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }
  TempCkpt tmp("openpfc_ckpt_method");
  CheckpointService svc({.every = 0, .directory = tmp.root}, MPI_COMM_WORLD);
  auto state = make_state(4, 1, 0);
  pfc::Time time({0.0, 1.0, 0.1}, 0.0);
  time.next();
  svc.save(state, time);

  auto restored = make_state(4, 1, 0);
  pfc::Time other({0.0, 1.0, 0.1}, 0.0);
  other.set_method(pfc::sim::steppers::RKIntegratorMethod::RK4_Classical);
  REQUIRE_THROWS_WITH(svc.load(restored, other, svc.step_dir(1)),
                      ContainsSubstring("method"));
}

TEST_CASE("CheckpointService identity mismatch grid", "[checkpoint][service]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }
  TempCkpt tmp("openpfc_ckpt_grid");
  CheckpointService svc({.every = 0, .directory = tmp.root}, MPI_COMM_WORLD);
  auto state = make_state(4, 1, 0);
  pfc::Time time({0.0, 1.0, 0.1}, 0.0);
  time.next();
  svc.save(state, time);

  auto other = make_state(8, 1, 0);
  pfc::Time time2({0.0, 1.0, 0.1}, 0.0);
  REQUIRE_THROWS_WITH(svc.load(other, time2, svc.step_dir(1)),
                      ContainsSubstring("grid"));
}

TEST_CASE("CheckpointService save/load restart equivalence 2-rank",
          "[checkpoint][service][MPI]") {
  int nproc = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (nproc != 2) {
    return;
  }

  TempCkpt tmp("openpfc_ckpt_service_2");
  CheckpointService svc({.every = 0, .directory = tmp.root}, MPI_COMM_WORLD);
  auto state = make_state(8, nproc, rank);
  pfc::Time time({0.0, 1.0, 0.1}, 0.0);
  time.next();
  time.next();
  state.get_field<double>("u").apply([&](double x, double y, double) {
    return x + 0.1 * y + static_cast<double>(time.get_increment());
  });
  svc.save(state, time);

  auto restored = make_state(8, nproc, rank);
  pfc::Time time2({0.0, 1.0, 0.1}, 0.0);
  svc.load(restored, time2, svc.step_dir(2));
  REQUIRE(time2.get_increment() == 2);
  require_fields_equal(state.get_field<double>("u"),
                       restored.get_field<double>("u"));
}

TEST_CASE("interrupted publish leaves no loadable bundle",
          "[checkpoint][publish][crash]") {
  namespace fs = std::filesystem;
  const auto root = fs::temp_directory_path() / "openpfc_ckpt_crash";
  fs::create_directories(root);
  const auto final_dir = root / "ckpt";
  std::vector<double> psi{1.0, 2.0, 3.0, 4.0};
  pfc::checkpoint::CheckpointMetadata meta{
      .format_version = 1,
      .accepted_time = 0.0,
      .accepted_increment = 1,
      .domain = {.global_dimensions = {4, 1, 1},
                 .physical_origin = {0, 0, 0},
                 .grid_spacing = {1, 1, 1}},
      .method_identity = "euler",
  };
  const auto brick = pfc::checkpoint::PublishedFieldBrick{
      .id = "psi",
      .dtype = "float64",
      .extents = {4, 1, 1},
      .bytes = std::as_bytes(std::span<const double>{psi}),
  };
  std::array<pfc::checkpoint::PublishedFieldBrick, 1> fields{brick};
  const auto outcome = pfc::checkpoint::publish_checkpoint_directory(
      final_dir, meta, fields,
      [](std::size_t, const auto &) { throw std::runtime_error("inject"); });
  REQUIRE_FALSE(outcome.ok);
  REQUIRE_FALSE(fs::exists(final_dir));
  REQUIRE_FALSE(fs::exists(fs::path(final_dir.string() + ".publishing")));
  fs::remove_all(root);
}

TEST_CASE("coupling FieldHandle exports host field geometry", "[coupling][unit]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }
  auto state = make_state(4, 1, 0);
  auto h = pfc::coupling::export_host_field(state, "u");
  REQUIRE(h.name == "u");
  REQUIRE(h.view.size() > 0);
  REQUIRE(h.owned_box.size[0] == 4);
  REQUIRE(std::string(h.memory_space) == "host");
}

TEST_CASE("coupling source via FieldHandle matches Field write",
          "[coupling][unit]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }
  auto a = make_state(4, 1, 0);
  auto b = make_state(4, 1, 0);
  auto h = pfc::coupling::export_host_field(a, "u");
  REQUIRE(h.view.data() == a.get_field<double>("u").data());
  openpfc_examples::HostSourceModifier src;
  src.apply(a, 0.3);
  b.get_field<double>("u").apply([](double x, double y, double z) {
    return openpfc_examples::coupling_source(x, y, z, 0.3);
  });
  require_fields_equal(a.get_field<double>("u"), b.get_field<double>("u"));
}

TEST_CASE("coupling imposed source matches FieldModifier-shaped write 2-rank",
          "[coupling][MPI]") {
  int nproc = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (nproc != 2) {
    return;
  }
  auto a = make_state(8, nproc, rank);
  auto b = make_state(8, nproc, rank);
  auto h = pfc::coupling::export_host_field(a, "u");
  REQUIRE(h.owned_box.size[0] > 0);
  openpfc_examples::HostSourceModifier src;
  src.apply(a, 1.2);
  b.get_field<double>("u").apply([](double x, double y, double z) {
    return openpfc_examples::coupling_source(x, y, z, 1.2);
  });
  require_fields_equal(a.get_field<double>("u"), b.get_field<double>("u"));
}
