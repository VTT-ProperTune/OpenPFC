// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
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

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#endif

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

namespace {

template <class Space> void packing_contract() {
  using F = pfc::data::Field<double, Space>;
  for (const auto extent : {std::array{4, 3, 2}, std::array{1, 3, 2}}) {
    for (int halo : {0, 1, 2}) {
      const auto [nx, ny, nz] = extent;
      F field(grid(), Box3i::from_bounds({1, 1, 0}, {nx, ny, nz - 1}), halo);
      std::vector<double> expected;
      double ghost = -777.;
      field.with_host_view([&](double *data, std::size_t size) {
        std::fill(data, data + size, ghost);
        for (int k = 0; k < nz; ++k)
          for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
              const auto value = manufactured(i + 1, j + 1, k, 4);
              data[field.idx(i, j, k)] = value;
              expected.push_back(value);
            }
      });
      field.sync_to_device();
      REQUIRE(pfc::checkpoint::pack_owned_real(field) == expected);
      CHECK_FALSE(field.residency().device_needs_refresh());
      auto snapshot = [&] {
        if constexpr (F::is_host_space)
          return field.vec();
        else
          return field.buffer().to_host();
      };
      if constexpr (!F::is_host_space) {
        // Raw-buffer canary detects a checkpoint-induced stale upload.
        // Restore the normal Field residency protocol immediately afterward.
        auto changed = snapshot();
        for (double &v : changed) v += 8.;
        field.buffer().copy_from_host(changed);
        field.sync_to_device();
        REQUIRE(snapshot() == changed);
        field.note_device_write();
        ghost += 8.;
      }
      const auto before = snapshot();
      REQUIRE_THROWS_AS(pfc::checkpoint::unpack_owned_real(
                            field, std::vector<double>(expected.size() - 1)),
                        std::invalid_argument);
      REQUIRE_THROWS_AS(pfc::checkpoint::unpack_owned_real(
                            field, std::vector<double>(expected.size() + 1)),
                        std::invalid_argument);
      REQUIRE(snapshot() == before);
      std::reverse(expected.begin(), expected.end());
      pfc::checkpoint::unpack_owned_real(field, expected);
      CHECK_FALSE(field.residency().device_needs_refresh());
      REQUIRE(pfc::checkpoint::pack_owned_real(field) == expected);
      const auto actual = snapshot();
      std::size_t owned = 0;
      // Independent padded index formula and ghost sentinels check storage.
      for (int k = -halo; k < nz + halo; ++k)
        for (int j = -halo; j < ny + halo; ++j)
          for (int i = -halo; i < nx + halo; ++i) {
            const auto idx = (i + halo) + (j + halo) * (nx + 2 * halo) +
                             (k + halo) * (nx + 2 * halo) * (ny + 2 * halo);
            if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
              CHECK(actual[idx] == expected[owned++]);
            else
              CHECK(actual[idx] == ghost);
          }
    }
  }
}

template <class Space> void device_redecomposition() {
  using F = pfc::data::Field<double, Space>;
  using Named = NamedField<Space>;
  BundleDir tmp("device-redecompose");
  if (comm_rank(MPI_COMM_WORLD) == 0) {
    F whole(grid(), Box3i::from_bounds({0, 0, 0}, {kNx - 1, kNy - 1, kNz - 1}), 2);
    whole.with_host_view([&](double *data, std::size_t size) {
      std::fill(data, data + size, -777.);
      for (int k = 0; k < kNz; ++k)
        for (int j = 0; j < kNy; ++j)
          for (int i = 0; i < kNx; ++i)
            data[whole.idx(i, j, k)] = manufactured(i, j, k, 3);
    });
    whole.sync_to_device();
    REQUIRE(publish_field_bundle(
                tmp.root / "b", std::vector<Named>{{"psi", &whole}},
                FieldBundleState{.accepted_time = 1.5,
                                 .accepted_increment = 12,
                                 .method_identity = "example/device-relaxation"},
                MPI_COMM_SELF)
                .ok);
    CHECK_FALSE(whole.residency().device_needs_refresh());
  }
  MPI_Barrier(MPI_COMM_WORLD);
  F part(grid(), x_slab(MPI_COMM_WORLD), 1);
  part.with_host_view(
      [](double *data, std::size_t size) { std::fill(data, data + size, -888.); });
  part.sync_to_device();
  const auto state = load_field_bundle(
      tmp.root / "b", std::vector<Named>{{"psi", &part}}, MPI_COMM_WORLD);
  REQUIRE(state.accepted_time == 1.5);
  REQUIRE(state.accepted_increment == 12);
  REQUIRE(state.method_identity == "example/device-relaxation");
  CHECK_FALSE(part.residency().device_needs_refresh());
  const auto actual = part.buffer().to_host();
  const auto n = part.local_size(), lo = part.lower_global();
  bool match = true;
  for (int k = -1; k < n[2] + 1; ++k)
    for (int j = -1; j < n[1] + 1; ++j)
      for (int i = -1; i < n[0] + 1; ++i) {
        const bool interior =
            i >= 0 && i < n[0] && j >= 0 && j < n[1] && k >= 0 && k < n[2];
        const double expected =
            interior ? manufactured(lo[0] + i, lo[1] + j, lo[2] + k, 3) : -888.;
        match = match && actual[part.idx(i, j, k)] == expected;
      }
  REQUIRE(all_ranks(match, MPI_COMM_WORLD));
  if (comm_rank(MPI_COMM_WORLD) == 0)
    REQUIRE(fs::file_size(tmp.root / "b/fields/psi.bin") ==
            sizeof(double) * kNx * kNy * kNz);
}

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
int node_rank() {
  MPI_Comm node = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int rank = 0;
  MPI_Comm_rank(node, &rank);
  MPI_Comm_free(&node);
  return rank;
}
#endif
} // namespace

TEST_CASE("owned checkpoint packing preserves host padding and rejects wrong sizes",
          "[checkpoint_pack]") {
  packing_contract<pfc::HostSpace>();
}

#if defined(OpenPFC_ENABLE_CUDA)
TEST_CASE("owned CUDA checkpoint packing preserves device residency and padding",
          "[checkpoint_pack][cuda]") {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    SKIP("No allocated CUDA device");
  REQUIRE(cudaSetDevice(node_rank() % count) == cudaSuccess);
  packing_contract<pfc::CUDASpace>();
}
TEST_CASE("padded CUDA field bundle restarts on another decomposition",
          "[checkpoint_device][checkpoint][MPI][cuda]") {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    SKIP("No allocated CUDA device");
  REQUIRE(cudaSetDevice(node_rank() % count) == cudaSuccess);
  device_redecomposition<pfc::CUDASpace>();
}
#endif
#if defined(OpenPFC_ENABLE_HIP)
TEST_CASE("owned HIP checkpoint packing preserves device residency and padding",
          "[checkpoint_pack][hip]") {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0)
    SKIP("No allocated HIP device");
  REQUIRE(hipSetDevice(node_rank() % count) == hipSuccess);
  packing_contract<pfc::HIPSpace>();
}
TEST_CASE("padded HIP field bundle restarts on another decomposition",
          "[checkpoint_device][checkpoint][MPI][hip]") {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0)
    SKIP("No allocated HIP device");
  REQUIRE(hipSetDevice(node_rank() % count) == hipSuccess);
  device_redecomposition<pfc::HIPSpace>();
}
#endif

TEST_CASE("field bundle rejects periodicity mismatch before destination writes",
          "[checkpoint][field_bundle][MPI]") {
  auto source_grid = grid();
  source_grid.periodic = {true, true, false};
  const auto local_box = x_slab(MPI_COMM_WORLD);
  Field first_source(source_grid, local_box);
  Field second_source(source_grid, local_box);
  first_source.with_host_view([](double *data, std::size_t size) {
    std::fill(data, data + size, 3.0);
  });
  second_source.with_host_view([](double *data, std::size_t size) {
    std::fill(data, data + size, 5.0);
  });
  auto destination_grid = source_grid;
  destination_grid.periodic = {true, true, true};
  // The first destination is compatible; the second must reject the complete
  // bundle before even the first field has been overwritten, on every rank.
  Field first_destination(source_grid, local_box);
  Field second_destination(destination_grid, local_box);
  auto poison = [](Field &field) {
    field.with_host_view([](double *data, std::size_t size) {
      std::fill(data, data + size, -7.0);
    });
  };
  auto unchanged = [](Field &field) {
    bool intact = false;
    field.with_host_view([&](double *data, std::size_t size) {
      intact = std::all_of(data, data + size, [](double value) { return value == -7.0; });
    });
    return intact;
  };
  poison(first_destination);
  poison(second_destination);
  BundleDir tmp("periodicity");
  const auto dir = tmp.root / "bundle";
  FieldBundleState state;
  const auto saved = publish_field_bundle(
      dir, std::vector<NamedField<>>{{"u", &first_source}, {"v", &second_source}},
      state, MPI_COMM_WORLD);
  REQUIRE(all_ranks(saved.ok, MPI_COMM_WORLD));
  REQUIRE_THROWS_AS(load_field_bundle(
      dir, std::vector<NamedField<>>{{"u", &first_destination}, {"v", &second_destination}},
      MPI_COMM_WORLD), std::invalid_argument);
  REQUIRE(all_ranks(unchanged(first_destination) && unchanged(second_destination), MPI_COMM_WORLD));
  Field compatible_destination(source_grid, local_box);
  const auto restored = load_field_bundle(
      dir, std::vector<NamedField<>>{{"u", &first_destination}, {"v", &compatible_destination}},
      MPI_COMM_WORLD);
  REQUIRE(restored.domain.periodic == source_grid.periodic);
  bool first_ok = false, second_ok = false;
  first_destination.with_host_view([&](double *data, std::size_t size) {
    first_ok = std::all_of(data, data + size, [](double value) { return value == 3.0; });
  });
  compatible_destination.with_host_view([&](double *data, std::size_t size) {
    second_ok = std::all_of(data, data + size, [](double value) { return value == 5.0; });
  });
  REQUIRE(all_ranks(first_ok && second_ok, MPI_COMM_WORLD));
}
