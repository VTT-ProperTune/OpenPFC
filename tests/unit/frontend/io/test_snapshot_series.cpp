// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

#include <openpfc/frontend/io/hdf5_writer.hpp>
#include <openpfc/frontend/io/scalar_field_file.hpp>
#include <openpfc/frontend/io/snapshot_series.hpp>
#include <openpfc/frontend/ui/json_snapshot_fields.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/state_access.hpp>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using nlohmann::json;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

int world_rank() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

std::filesystem::path shared_temp(const char *tag) {
  std::string name;
  if (world_rank() == 0) {
    name = std::string("openpfc-snap-") + tag + "-" + std::to_string(::getpid());
  }
  int len = static_cast<int>(name.size());
  MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
  name.resize(static_cast<std::size_t>(len));
  MPI_Bcast(name.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
  return std::filesystem::temp_directory_path() / name;
}

std::vector<double> read_doubles(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  in.seekg(0, std::ios::end);
  const auto bytes = static_cast<std::size_t>(in.tellg());
  REQUIRE(bytes % sizeof(double) == 0);
  in.seekg(0);
  std::vector<double> out(bytes / sizeof(double));
  in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(bytes));
  REQUIRE(in.good());
  return out;
}

pfc::io::SnapshotSeriesOptions opts(const std::filesystem::path &dir,
                                    std::string prefix, MPI_Comm comm) {
  return pfc::io::SnapshotSeriesOptions{dir, std::move(prefix), comm};
}

} // namespace

TEST_CASE("A writer that omits complex fields rejects them before use",
          "[snapshot][writer]") {
  class RealOnly : public pfc::ResultsWriter {
  public:
    void set_domain(const std::array<int, 3> &, const std::array<int, 3> &,
                    const std::array<int, 3> &) override {}
    [[nodiscard]] bool writes_real() const override { return true; }
    MPI_Status write(int, pfc::field::FieldView<double>) override {
      return MPI_Status{};
    }
  };
  RealOnly writer;
  REQUIRE(writer.writes_real());
  REQUIRE_FALSE(writer.writes_complex());
  pfc::ResultsWriter &sink = writer;
  std::vector<std::complex<double>> z(2);
  REQUIRE_THROWS_AS(sink.write(0, z), std::invalid_argument);

  pfc::BinaryWriter binary("unused_%04d.bin", MPI_COMM_SELF);
  REQUIRE(binary.writes_real());
  REQUIRE(binary.writes_complex());
  pfc::VTKWriter vtk("unused_%04d.vti", MPI_COMM_SELF);
  REQUIRE(vtk.writes_real());
  REQUIRE(vtk.writes_complex());
#ifdef OPENPFC_HAS_HDF5
  pfc::HDF5Writer hdf5("unused.h5", MPI_COMM_SELF);
  REQUIRE(hdf5.writes_real());
  REQUIRE_FALSE(hdf5.writes_complex());
  pfc::ResultsWriter &hdf5_sink = hdf5;
  REQUIRE_THROWS_AS(hdf5_sink.write(0, z), std::invalid_argument);
#endif
}

TEST_CASE("One scalar field is written with step and time metadata", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("one");
  const auto domain = pfc::domain::create({4, 2, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> phi(domain, box, 0);
  phi.apply([](const pfc::Real3 &x) { return 10.0 * x[0] + x[1]; });
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("phi", phi);
    REQUIRE(std::filesystem::is_directory(dir));
    series.write(3, 1.5);
    series.write(6, 3.0);
    series.close();
    REQUIRE(series.steps() == std::vector<int>{3, 6});
    REQUIRE_THAT(series.times()[0], WithinAbs(1.5, 1e-15));
    REQUIRE_THAT(series.times()[1], WithinAbs(3.0, 1e-15));
  }
  const auto values = read_doubles(dir / "run_phi_0000.bin");
  REQUIRE(values.size() == 8);
  REQUIRE_THAT(values[0], WithinAbs(0.0, 1e-12));
  const auto manifest = json::parse(std::ifstream(dir / "run_manifest.json"));
  REQUIRE(manifest.at("fields") == json::array({"phi"}));
  REQUIRE(manifest.at("steps") == json::array({3, 6}));
  REQUIRE(manifest.at("nx") == 4);
  REQUIRE(std::filesystem::exists(dir / "run.xdmf"));
  std::filesystem::remove_all(dir);
}

TEST_CASE("Several scalar fields share one series", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("multi");
  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> phi(domain, box, 0);
  pfc::data::Field<double> temperature(domain, box, 0);
  phi.apply([](const pfc::Real3 &) { return 1.0; });
  temperature.apply([](const pfc::Real3 &) { return 2.0; });
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("phi", phi);
    series.add_field("temperature", temperature);
    series.write(0, 0.0);
    series.close();
  }
  const auto a = read_doubles(dir / "run_phi_0000.bin");
  const auto b = read_doubles(dir / "run_temperature_0000.bin");
  REQUIRE(a.size() == 4);
  REQUIRE(b.size() == 4);
  for (double v : a) REQUIRE_THAT(v, WithinAbs(1.0, 1e-15));
  for (double v : b) REQUIRE_THAT(v, WithinAbs(2.0, 1e-15));
  const auto manifest = json::parse(std::ifstream(dir / "run_manifest.json"));
  REQUIRE(manifest.at("fields").size() == 2);
  std::filesystem::remove_all(dir);
}

TEST_CASE("A padded field writes owned cells only", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("pad");
  const auto domain = pfc::domain::create({3, 2, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> field(domain, box, 1);
  field.with_host_view([&](double *data, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) data[i] = -12345.0;
  });
  field.apply([](double, double, double) { return 7.0; });
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("u", field);
    series.write(0, 0.0);
    series.close();
  }
  const auto values = read_doubles(dir / "run_u_0000.bin");
  REQUIRE(values.size() == 6);
  for (double v : values) REQUIRE_THAT(v, WithinAbs(7.0, 1e-15));
  std::filesystem::remove_all(dir);
}

TEST_CASE("An unpadded face-halo field writes its owned brick", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("face");
  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> field(domain, box, 0, 1);
  REQUIRE(field.storage_halo() == 0);
  REQUIRE(field.halo_width() == 1);
  field.apply([](double, double, double) { return 4.0; });
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("u", field);
    series.write(1, 0.25);
    series.close();
  }
  const auto values = read_doubles(dir / "run_u_0000.bin");
  REQUIRE(values.size() == 4);
  for (double v : values) REQUIRE_THAT(v, WithinAbs(4.0, 1e-15));
  std::filesystem::remove_all(dir);
}

TEST_CASE("JSON fields keep their names and VTK payload", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("vtk");
  const auto pattern = (dir / "h_%04d.vti").string();
  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> h(domain, box, 0);
  h.apply([](double, double, double) { return 1.25; });
  const json cfg{{"fields", json::array({{{"name", "h"}, {"data", pattern}}})}};
  {
    pfc::io::SnapshotSeries series(
        domain, box, pfc::io::SnapshotSeriesOptions{.comm = MPI_COMM_SELF});
    pfc::ui::bind_snapshot_field(series, cfg, "h", h);
    pfc::ui::finish_snapshot_fields(series, cfg);
    series.write(0, 0.0);
    series.close();
  }
  const auto path = dir / "h_0000.vti";
  REQUIRE(std::filesystem::exists(path));
  std::ifstream in(path, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  REQUIRE(text.find("Name=\"h\"") != std::string::npos);
  const auto marker = text.find("\n_");
  REQUIRE(marker != std::string::npos);
  const auto *raw =
      reinterpret_cast<const unsigned char *>(text.data() + marker + 2);
  std::uint64_t bytes = 0;
  for (int i = 0; i < 8; ++i) bytes |= static_cast<std::uint64_t>(raw[i]) << (8 * i);
  REQUIRE(bytes == 2 * sizeof(double));
  double value = 0.0;
  std::memcpy(&value, raw + 8, sizeof(double));
  REQUIRE_THAT(value, WithinAbs(1.25, 1e-15));
  const json missing_name{
      {"fields", json::array({{{"name", "c"}, {"data", pattern}}})}};
  pfc::io::SnapshotSeries missing(
      domain, box, pfc::io::SnapshotSeriesOptions{.comm = MPI_COMM_SELF});
  pfc::ui::bind_snapshot_field(missing, missing_name, "h", h);
  REQUIRE_THROWS_AS(pfc::ui::finish_snapshot_fields(missing, missing_name),
                    std::invalid_argument);
  std::filesystem::remove_all(dir);
}

TEST_CASE("Output directory creation fails closed", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("fail");
  std::filesystem::create_directories(dir);
  const auto blocker = dir / "not-a-directory";
  {
    std::ofstream touch(blocker);
    REQUIRE(touch.good());
  }
  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> u(domain, box, 0);
  pfc::io::SnapshotSeries series(
      domain, box, pfc::io::SnapshotSeriesOptions{.comm = MPI_COMM_SELF});
  REQUIRE_THROWS_AS(series.add_field("u", u, (blocker / "u_%04d.bin").string(),
                                     pfc::io::SnapshotFormat::Binary),
                    std::runtime_error);
  std::filesystem::remove_all(dir);
}

TEST_CASE("XDMF records a non-zero domain origin", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("origin");
  const auto domain = pfc::domain::create(pfc::GridSize({2, 2, 2}),
                                          pfc::PhysicalOrigin({1.5, -2.0, 0.25}),
                                          pfc::GridSpacing({0.5, 0.25, 0.125}));
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> phi(domain, box, 0);
  phi.apply([](double, double, double) { return 1.0; });
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("phi", phi);
    series.write(0, 0.0);
    series.close();
  }
  const auto manifest = json::parse(std::ifstream(dir / "run_manifest.json"));
  REQUIRE_THAT(manifest.at("origin").at(0).get<double>(), WithinAbs(1.5, 1e-15));
  REQUIRE_THAT(manifest.at("origin").at(1).get<double>(), WithinAbs(-2.0, 1e-15));
  REQUIRE_THAT(manifest.at("origin").at(2).get<double>(), WithinAbs(0.25, 1e-15));
  std::ifstream in(dir / "run.xdmf");
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  const auto geo = text.find("ORIGIN_DXDYDZ");
  REQUIRE(geo != std::string::npos);
  REQUIRE(text.find("0.25 -2 1.5", geo) != std::string::npos);
  REQUIRE(text.find("0.125 0.25 0.5", geo) != std::string::npos);
  REQUIRE(text.find(">0 0 0<", geo) == std::string::npos);
  std::filesystem::remove_all(dir);
}

TEST_CASE("XDMF paths are relative to the XDMF file", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto root = shared_temp("xdmfrel");
  const auto domain = pfc::domain::create({2, 2, 2});
  const auto box = pfc::domain::index_box(domain);
  {
    pfc::io::SnapshotSeries series(domain, box, opts(root, "series", MPI_COMM_SELF));
    series.add_field("one", (root / "data1" / "one_%04d.bin").string(),
                     pfc::io::SnapshotFormat::Binary,
                     [](std::vector<double> &out) { out.assign(8, 1.0); });
    series.add_field("two", (root / "data2" / "two_%04d.bin").string(),
                     pfc::io::SnapshotFormat::Binary,
                     [](std::vector<double> &out) { out.assign(8, 2.0); });
    series.write(0, 0.0);
    series.close();
  }
  const auto xdmf = root / "series.xdmf";
  std::ifstream in(xdmf);
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  for (const char *rel : {"data1/one_0000.bin", "data2/two_0000.bin"}) {
    REQUIRE(text.find(rel) != std::string::npos);
    REQUIRE(std::filesystem::exists(xdmf.parent_path() / rel));
  }
  std::filesystem::remove_all(root);
}

TEST_CASE("Manifest JSON escapes names and paths", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("escape");
  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = pfc::domain::index_box(domain);
  const auto pattern = (dir / "odd\"name_%04d.bin").string();
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("phi\"a", pattern, pfc::io::SnapshotFormat::Binary,
                     [](std::vector<double> &out) { out.assign(2, 1.0); });
    series.write(0, 0.0);
    series.close();
  }
  const auto manifest = json::parse(std::ifstream(dir / "run_manifest.json"));
  REQUIRE(manifest.at("fields").at(0) == "phi\"a");
  REQUIRE(manifest.at("patterns").at(0) == pattern);
  std::filesystem::remove_all(dir);
}

TEST_CASE("close reports a manifest that cannot be written", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("manifest-fail");
  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
  series.add_field("u", (dir / "u_%04d.bin").string(),
                   pfc::io::SnapshotFormat::Binary,
                   [](std::vector<double> &out) { out.assign(2, 1.0); });
  series.write(0, 0.0);
  std::filesystem::create_directory(dir / "run_manifest.json");
  REQUIRE_THROWS_WITH(series.close(), ContainsSubstring("cannot open"));
  std::filesystem::remove_all(dir);
}

TEST_CASE("Two ranks write one owned brick", "[snapshot][MPI]") {
  if (world_size() != 2) SKIP("two-rank snapshot");
  const auto dir = shared_temp("mpi");
  const int rank = world_rank();
  const auto domain = pfc::domain::create({4, 2, 1});
  const auto local = pfc::Box3i::from_bounds({rank * 2, 0, 0}, {rank * 2 + 1, 1, 0});
  pfc::data::Field<double> phi(domain, local, 1);
  phi.with_host_view([&](double *data, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) data[i] = -99.0;
  });
  for (int k = 0; k < local.size[2]; ++k) {
    for (int j = 0; j < local.size[1]; ++j) {
      for (int i = 0; i < local.size[0]; ++i) {
        phi(i, j, k) = static_cast<double>(local.low[0] + i + 10 * j);
      }
    }
  }
  {
    pfc::io::SnapshotSeries series(domain, local, opts(dir, "run", MPI_COMM_WORLD));
    series.add_field("phi", phi);
    series.write(4, 0.5);
    series.close();
  }
  if (rank == 0) {
    const auto values = read_doubles(dir / "run_phi_0000.bin");
    REQUIRE(values.size() == 8);
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 4; ++i) {
        REQUIRE_THAT(values[static_cast<std::size_t>(i + 4 * j)],
                     WithinAbs(static_cast<double>(i + 10 * j), 1e-12));
      }
    }
    const auto manifest = json::parse(std::ifstream(dir / "run_manifest.json"));
    REQUIRE(manifest.at("steps") == json::array({4}));
    REQUIRE(std::filesystem::exists(dir / "run.xdmf"));
    std::filesystem::remove_all(dir);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("A pack failure on one rank does not enter the writer",
          "[snapshot][MPI]") {
  if (world_size() != 2) SKIP("two-rank snapshot");
  const auto dir = shared_temp("packfail");
  const int rank = world_rank();
  const auto domain = pfc::domain::create({4, 2, 1});
  const auto local = pfc::Box3i::from_bounds({rank * 2, 0, 0}, {rank * 2 + 1, 1, 0});
  {
    pfc::io::SnapshotSeries series(domain, local, opts(dir, "run", MPI_COMM_WORLD));
    series.add_field("phi", (dir / "run_phi_%04d.bin").string(),
                     pfc::io::SnapshotFormat::Binary, [&](std::vector<double> &out) {
                       if (rank == 1) {
                         throw std::runtime_error("rank 1 pack refused");
                       }
                       out.assign(4, 1.0);
                     });
    if (rank == 1) {
      REQUIRE_THROWS_WITH(series.write(0, 0.0),
                          ContainsSubstring("rank 1 pack refused"));
    } else {
      REQUIRE_THROWS_WITH(series.write(0, 0.0),
                          ContainsSubstring("a peer failed while packing"));
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    REQUIRE_FALSE(std::filesystem::exists(dir / "run_phi_0000.bin"));
    std::filesystem::remove_all(dir);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("close reports an XDMF failure on every rank", "[snapshot][MPI]") {
  if (world_size() != 2) SKIP("two-rank snapshot");
  const auto dir = shared_temp("xdmffail");
  const int rank = world_rank();
  const auto domain = pfc::domain::create({4, 2, 1});
  const auto local = pfc::Box3i::from_bounds({rank * 2, 0, 0}, {rank * 2 + 1, 1, 0});
  pfc::data::Field<double> phi(domain, local, 0);
  phi.apply([](double, double, double) { return 1.0; });
  {
    pfc::io::SnapshotSeries series(domain, local, opts(dir, "run", MPI_COMM_WORLD));
    series.add_field("phi", phi);
    series.write(0, 0.0);
    if (rank == 0) std::filesystem::create_directory(dir / "run.xdmf");
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      REQUIRE_THROWS_WITH(series.close(), ContainsSubstring("cannot open"));
    } else {
      REQUIRE_THROWS_WITH(series.close(), ContainsSubstring("another rank"));
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::filesystem::remove_all(dir);
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("Cadence counts samples and can be skipped", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  REQUIRE_THROWS_AS(pfc::io::SnapshotCadence(0), std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::io::SnapshotCadence(-1), std::invalid_argument);
  const auto dir = shared_temp("cadence");
  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> u(domain, box, 0);
  u.apply([](double, double, double) { return 1.0; });
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.set_cadence(pfc::io::SnapshotCadence(2));
    REQUIRE(series.due(0));
    REQUIRE_FALSE(series.due(1));
    REQUIRE(series.due(2));
    series.add_field("u", u);
    REQUIRE(series.write_if_due(0, 4, 0.4));
    REQUIRE_FALSE(series.write_if_due(1, 5, 0.5));
    REQUIRE(series.write_if_due(2, 6, 0.6));
    series.close();
    REQUIRE(series.steps() == std::vector<int>{4, 6});
    REQUIRE(series.frames() == 2);
  }
  REQUIRE(std::filesystem::exists(dir / "run_u_0000.bin"));
  REQUIRE(std::filesystem::exists(dir / "run_u_0001.bin"));
  REQUIRE_FALSE(std::filesystem::exists(dir / "run_u_0002.bin"));
  std::filesystem::remove_all(dir);
}

TEST_CASE("Restored progress continues at the next frame", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("progress");
  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> u(domain, box, 0);
  u.apply([](double, double, double) { return 3.0; });
  pfc::io::SnapshotProgress saved;
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("u", u);
    series.write(10, 0.1);
    series.write(20, 0.2);
    saved = series.progress();
    series.close();
  }
  REQUIRE(saved.next_frame == 2);
  u.apply([](double, double, double) { return 9.0; });
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("u", u);
    series.restore(saved);
    series.write(30, 0.3);
    series.close();
    REQUIRE(series.steps() == std::vector<int>{10, 20, 30});
    REQUIRE(series.frames() == 3);
  }
  const auto first = read_doubles(dir / "run_u_0000.bin");
  const auto third = read_doubles(dir / "run_u_0002.bin");
  REQUIRE(first.size() == 2);
  REQUIRE(third.size() == 2);
  for (double v : first) REQUIRE_THAT(v, WithinAbs(3.0, 1e-15));
  for (double v : third) REQUIRE_THAT(v, WithinAbs(9.0, 1e-15));
  const auto manifest = json::parse(std::ifstream(dir / "run_manifest.json"));
  REQUIRE(manifest.at("steps") == json::array({10, 20, 30}));
  REQUIRE(manifest.at("times").size() == 3);
  std::filesystem::remove_all(dir);
}

TEST_CASE("A raw host buffer is written without a field object", "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("raw");
  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = pfc::domain::index_box(domain);
  const double raw[4] = {1.0, 2.0, 3.0, 4.0};
  {
    pfc::io::SnapshotSeries series(domain, box, opts(dir, "run", MPI_COMM_SELF));
    series.add_field("u", (dir / "run_u_%04d.bin").string(),
                     pfc::io::SnapshotFormat::Binary,
                     [&](std::vector<double> &out) { out.assign(raw, raw + 4); });
    series.write(1, 0.5);
    series.close();
  }
  const auto values = read_doubles(dir / "run_u_0000.bin");
  REQUIRE(values.size() == 4);
  REQUIRE_THAT(values[0], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(values[3], WithinAbs(4.0, 1e-15));
  std::filesystem::remove_all(dir);
}

TEST_CASE("A named brick and its XDMF sidecar keep the domain origin",
          "[snapshot]") {
  if (world_size() != 1) SKIP("single-rank snapshot");
  const auto dir = shared_temp("once");
  const auto domain = pfc::domain::create(pfc::GridSize({2, 1, 1}),
                                          pfc::PhysicalOrigin({1.5, -2.0, 0.25}),
                                          pfc::GridSpacing({0.5, 0.25, 0.125}));
  const auto box = pfc::domain::index_box(domain);
  pfc::data::Field<double> u(domain, box, 1);
  u.with_host_view([&](double *data, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) data[i] = -7.0;
  });
  u.apply([](double, double, double) { return 4.0; });
  pfc::io::write_scalar_brick(u, MPI_COMM_SELF, dir / "h_final.bin");
  const auto origin = pfc::domain::get_origin(domain);
  const auto spacing = pfc::domain::get_spacing(domain);
  const auto n = pfc::domain::get_size(domain);
  pfc::io::write_scalar_xdmf(MPI_COMM_SELF, dir / "h_final.xdmf",
                             pfc::io::BinarySeriesGeometry{.nx = n[0],
                                                           .ny = n[1],
                                                           .nz = n[2],
                                                           .x0 = origin[0],
                                                           .y0 = origin[1],
                                                           .z0 = origin[2],
                                                           .dx = spacing[0],
                                                           .dy = spacing[1],
                                                           .dz = 1.0},
                             "h", "h_final.bin");
  const auto values = read_doubles(dir / "h_final.bin");
  REQUIRE(values.size() == 2);
  for (double v : values) REQUIRE_THAT(v, WithinAbs(4.0, 1e-15));
  std::ifstream in(dir / "h_final.xdmf");
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  REQUIRE(text.find("h_final.bin") != std::string::npos);
  REQUIRE(text.find("0.25 -2 1.5") != std::string::npos);
  std::filesystem::remove_all(dir);
}

TEST_CASE("Two ranks write VTK pieces of the owned brick", "[snapshot][MPI]") {
  if (world_size() != 2) SKIP("two-rank snapshot");
  const auto dir = shared_temp("vtkmpi");
  const int rank = world_rank();
  const auto domain = pfc::domain::create({4, 2, 1});
  const auto local = pfc::Box3i::from_bounds({rank * 2, 0, 0}, {rank * 2 + 1, 1, 0});
  pfc::data::Field<double> phi(domain, local, 1);
  phi.with_host_view([&](double *data, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) data[i] = -99.0;
  });
  for (int j = 0; j < local.size[1]; ++j) {
    for (int i = 0; i < local.size[0]; ++i) {
      phi(i, j, 0) = static_cast<double>(local.low[0] + i + 10 * j);
    }
  }
  {
    pfc::io::SnapshotSeries series(domain, local, opts(dir, "run", MPI_COMM_WORLD));
    series.add_field("phi", phi, (dir / "phi_%04d.vti").string(),
                     pfc::io::SnapshotFormat::Vtk);
    series.write(2, 0.25);
    series.close();
  }
  const auto piece = dir / ("phi_0000_" + std::to_string(rank) + ".vti");
  REQUIRE(std::filesystem::exists(piece));
  std::ifstream in(piece, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  REQUIRE(text.find("-99") == std::string::npos);
  const auto marker = text.find("\n_");
  REQUIRE(marker != std::string::npos);
  const auto *raw =
      reinterpret_cast<const unsigned char *>(text.data() + marker + 2);
  std::uint64_t bytes = 0;
  for (int i = 0; i < 8; ++i) bytes |= static_cast<std::uint64_t>(raw[i]) << (8 * i);
  // The lower piece includes the next rank's first point plane so the
  // reader can form cells across the cut without gaps or duplicate cells.
  const int width = rank == 0 ? 3 : 2;
  REQUIRE(bytes == static_cast<std::uint64_t>(2 * width) * sizeof(double));
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < width; ++i) {
      double value = 0;
      std::memcpy(&value, raw + 8 + (i + width * j) * sizeof(double),
                  sizeof(double));
      REQUIRE(value == static_cast<double>(rank * 2 + i + 10 * j));
    }
  double first = 0.0;
  std::memcpy(&first, raw + 8, sizeof(double));
  REQUIRE_THAT(first, WithinAbs(static_cast<double>(rank * 2), 1e-12));
  if (rank == 0) REQUIRE(std::filesystem::exists(dir / "phi_0000.pvti"));
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::filesystem::remove_all(dir);
  MPI_Barrier(MPI_COMM_WORLD);
}
