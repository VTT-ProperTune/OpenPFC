// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include "openpfc/kernel/checkpoint/field_bundle.hpp"
#include "openpfc/kernel/data/domain.hpp"
#include "openpfc/kernel/data/grid_field.hpp"

namespace {

namespace fs = std::filesystem;
using pfc::Box3i;
using pfc::checkpoint::FieldBundleSidecar;
using pfc::checkpoint::FieldBundleState;
using pfc::checkpoint::load_field_bundle;
using pfc::checkpoint::NamedField;
using pfc::checkpoint::publish_field_bundle;
using Field = pfc::data::Field<double>;

constexpr int kNx = 7;
constexpr int kNy = 5;
constexpr int kNz = 3;

int comm_rank(MPI_Comm c) {
  int r = 0;
  MPI_Comm_rank(c, &r);
  return r;
}

int comm_size(MPI_Comm c) {
  int n = 1;
  MPI_Comm_size(c, &n);
  return n;
}

/// Uneven split of x over the ranks of @p c: rank r owns ceil-or-floor cells,
/// the first ranks one more.
Box3i x_slab(MPI_Comm c) {
  const int n = comm_size(c);
  const int r = comm_rank(c);
  const int base = kNx / n;
  const int extra = kNx % n;
  const int lo = r * base + std::min(r, extra);
  const int len = base + (r < extra ? 1 : 0);
  return Box3i::from_bounds({lo, 0, 0}, {lo + len - 1, kNy - 1, kNz - 1});
}

pfc::Domain grid() {
  return pfc::domain::create(pfc::GridSize({kNx, kNy, kNz}),
                             pfc::PhysicalOrigin({0.25, -1.0, 2.0}),
                             pfc::GridSpacing({0.5, 0.75, 1.25}));
}

double manufactured(int gi, int gj, int gk, int field) {
  return 1000.0 * field + 100.0 * gi + 10.0 * gj + gk + 0.125;
}

void fill(Field &f, int field) {
  const auto lo = f.lower_global();
  const auto n = f.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i)
        f(i, j, k) = manufactured(lo[0] + i, lo[1] + j, lo[2] + k, field);
}

bool matches(const Field &f, int field) {
  const auto lo = f.lower_global();
  const auto n = f.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i)
        if (f(i, j, k) != manufactured(lo[0] + i, lo[1] + j, lo[2] + k, field))
          return false;
  return true;
}

bool all_ranks(bool local, MPI_Comm c) {
  int in = local ? 1 : 0;
  int out = 0;
  MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, c);
  return out == 1;
}

/// Shared scratch directory chosen by rank 0, removed on destruction.
struct BundleDir {
  fs::path root;
  explicit BundleDir(const char *tag) {
    std::string path;
    if (comm_rank(MPI_COMM_WORLD) == 0) {
      path = (fs::temp_directory_path() / "openpfc_field_bundle" /
              (std::string(tag) + "_" +
               std::to_string(reinterpret_cast<std::uintptr_t>(this))))
                 .string();
      fs::remove_all(path);
      fs::create_directories(path);
    }
    int n = static_cast<int>(path.size());
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    path.resize(static_cast<std::size_t>(n));
    MPI_Bcast(path.data(), n, MPI_CHAR, 0, MPI_COMM_WORLD);
    root = path;
  }
  ~BundleDir() {
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm_rank(MPI_COMM_WORLD) == 0) {
      std::error_code ec;
      fs::remove_all(root, ec);
    }
  }
  BundleDir(const BundleDir &) = delete;
  BundleDir &operator=(const BundleDir &) = delete;
};

std::string read_file(const fs::path &p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

} // namespace

TEST_CASE("field bundle round-trips padded and unpadded fields",
          "[checkpoint][MPI]") {
  BundleDir tmp("roundtrip");
  const auto box = x_slab(MPI_COMM_WORLD);
  Field u(grid(), box, 0);
  Field v(grid(), box, 2);
  fill(u, 1);
  fill(v, 2);

  FieldBundleState state;
  state.accepted_time = 1.5;
  state.accepted_increment = 12;
  state.method_identity = "example/relaxation";
  state.boundary_conditions = {
      {"x", "periodic"}, {"y", "periodic"}, {"z", "periodic"}};
  const nlohmann::json side = {{"model", "example"}, {"value", 0.1}};
  const auto out = publish_field_bundle(
      tmp.root / "b", std::vector<NamedField<>>{{"u", &u}, {"v", &v}}, state,
      MPI_COMM_WORLD, {"example.json", side});
  REQUIRE(out.ok);

  if (comm_rank(MPI_COMM_WORLD) == 0) {
    REQUIRE(read_file(tmp.root / "b" / "example.json") == side.dump(2) + "\n");
    const auto meta =
        nlohmann::json::parse(read_file(tmp.root / "b" / "metadata.json"));
    REQUIRE(meta["fields"] == nlohmann::json({"u", "v"}));
    REQUIRE(meta["decomposition"]["mpi_size"] == comm_size(MPI_COMM_WORLD));
    REQUIRE(meta["boundary_conditions"]["x"] == "periodic");
    REQUIRE(fs::file_size(tmp.root / "b" / "fields" / "u.bin") ==
            sizeof(double) * kNx * kNy * kNz);
  }

  Field u2(grid(), box, 1);
  Field v2(grid(), box, 0);
  const auto meta = load_field_bundle(
      tmp.root / "b", std::vector<NamedField<>>{{"u", &u2}, {"v", &v2}},
      MPI_COMM_WORLD);
  REQUIRE(meta.accepted_time == 1.5);
  REQUIRE(meta.accepted_increment == 12);
  REQUIRE(meta.method_identity == "example/relaxation");
  REQUIRE(all_ranks(matches(u2, 1) && matches(v2, 2), MPI_COMM_WORLD));
}

TEST_CASE("field bundle restarts on another decomposition", "[checkpoint][MPI]") {
  BundleDir tmp("redecompose");
  // Written by rank 0 alone on the whole grid ...
  if (comm_rank(MPI_COMM_WORLD) == 0) {
    Field whole(grid(), Box3i::from_bounds({0, 0, 0}, {kNx - 1, kNy - 1, kNz - 1}),
                1);
    fill(whole, 3);
    const auto out = publish_field_bundle(tmp.root / "b",
                                          std::vector<NamedField<>>{{"psi", &whole}},
                                          FieldBundleState{}, MPI_COMM_SELF);
    REQUIRE(out.ok);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // ... and read back on every rank's slab.
  Field part(grid(), x_slab(MPI_COMM_WORLD), 0);
  (void)load_field_bundle(tmp.root / "b", std::vector<NamedField<>>{{"psi", &part}},
                          MPI_COMM_WORLD);
  REQUIRE(all_ranks(matches(part, 3), MPI_COMM_WORLD));
}

TEST_CASE("field bundle refuses mismatched restarts and inputs",
          "[checkpoint][MPI]") {
  BundleDir tmp("refuse");
  const auto box = x_slab(MPI_COMM_WORLD);
  Field u(grid(), box, 0);
  fill(u, 1);
  REQUIRE(publish_field_bundle(tmp.root / "b", std::vector<NamedField<>>{{"u", &u}},
                               FieldBundleState{}, MPI_COMM_WORLD)
              .ok);

  Field w(grid(), box, 0);
  REQUIRE_THROWS_AS(load_field_bundle(tmp.root / "b",
                                      std::vector<NamedField<>>{{"w", &w}},
                                      MPI_COMM_WORLD),
                    std::invalid_argument);
  Field other(pfc::domain::with_spacing({kNx, kNy, kNz}, {0.5, 0.75, 1.25}), box, 0);
  REQUIRE_THROWS_AS(load_field_bundle(tmp.root / "b",
                                      std::vector<NamedField<>>{{"u", &other}},
                                      MPI_COMM_WORLD),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(load_field_bundle(tmp.root / "missing",
                                      std::vector<NamedField<>>{{"u", &w}},
                                      MPI_COMM_WORLD),
                    std::invalid_argument);

  // A second publish to the same path fails and leaves the first intact.
  const auto again =
      publish_field_bundle(tmp.root / "b", std::vector<NamedField<>>{{"u", &w}},
                           FieldBundleState{}, MPI_COMM_WORLD);
  REQUIRE_FALSE(again.ok);
  (void)load_field_bundle(tmp.root / "b", std::vector<NamedField<>>{{"u", &w}},
                          MPI_COMM_WORLD);
  REQUIRE(all_ranks(matches(w, 1), MPI_COMM_WORLD));

  REQUIRE_THROWS_AS(publish_field_bundle(tmp.root / "c", std::vector<NamedField<>>{},
                                         FieldBundleState{}, MPI_COMM_WORLD),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      publish_field_bundle(tmp.root / "c",
                           std::vector<NamedField<>>{{"u", &u}, {"o", &other}},
                           FieldBundleState{}, MPI_COMM_WORLD),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      publish_field_bundle(tmp.root / "c", std::vector<NamedField<>>{{"u", &u}},
                           FieldBundleState{}, MPI_COMM_WORLD,
                           {"metadata.json", nlohmann::json{{"a", 1}}}),
      std::invalid_argument);
  if (comm_rank(MPI_COMM_WORLD) == 0) {
    REQUIRE_FALSE(fs::exists(tmp.root / "c"));
  }
}
