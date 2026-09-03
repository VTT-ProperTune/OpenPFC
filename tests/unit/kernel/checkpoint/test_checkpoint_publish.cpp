// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include "openpfc/kernel/checkpoint/brick_io.hpp"
#include "openpfc/kernel/checkpoint/checkpoint_metadata.hpp"
#include "openpfc/kernel/checkpoint/publish.hpp"

namespace {

namespace fs = std::filesystem;
using pfc::checkpoint::CheckpointMetadata;
using pfc::checkpoint::DomainParams;
using pfc::checkpoint::publish_checkpoint_directory;
using pfc::checkpoint::write_real_brick_mpi;

struct TempResultsDir {
  fs::path root;
  fs::path final_dir;

  TempResultsDir() {
    root = fs::temp_directory_path() / "openpfc_results_checkpoint_publish" /
           ("case_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    fs::create_directories(root);
    final_dir = root / "ckpt";
  }

  ~TempResultsDir() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }

  TempResultsDir(const TempResultsDir &) = delete;
  TempResultsDir &operator=(const TempResultsDir &) = delete;
};

bool single_rank() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n == 1;
}

/// One-rank brick: the whole grid is the local box at offset 0.
void write_brick(const fs::path &fields_dir, const std::string &id,
                 std::array<int, 3> extents, const std::vector<double> &owned) {
  write_real_brick_mpi((fields_dir / (id + ".bin")).string(), MPI_COMM_WORLD,
                       extents, extents, {0, 0, 0}, owned);
}

CheckpointMetadata meta_for(std::array<int, 3> dims, double t, int inc) {
  return CheckpointMetadata{
      .format_version = 1,
      .accepted_time = t,
      .accepted_increment = inc,
      .domain =
          DomainParams{
              .global_dimensions = dims,
              .physical_origin = {0.0, 0.0, 0.0},
              .grid_spacing = {1.0, 1.0, 1.0},
          },
      .decomposition = std::nullopt,
      .method_identity = "euler",
  };
}

} // namespace

TEST_CASE("scalar field checkpoint publish success", "[checkpoint][publish]") {
  if (!single_rank()) {
    return;
  }
  TempResultsDir tmp;
  std::vector<double> psi{1.0, 2.0, 3.0, 4.0};
  const auto meta = meta_for({4, 1, 1}, 1.25, 5);

  const auto outcome = publish_checkpoint_directory(
      tmp.final_dir, meta, MPI_COMM_WORLD,
      [&](const fs::path &dir) { write_brick(dir, "psi", {4, 1, 1}, psi); });
  REQUIRE(outcome.ok);
  REQUIRE(fs::exists(tmp.final_dir / "metadata.json"));
  REQUIRE(fs::exists(tmp.final_dir / "fields" / "psi.bin"));
  REQUIRE(fs::file_size(tmp.final_dir / "fields" / "psi.bin") == 32);
  REQUIRE_FALSE(fs::exists(fs::path(tmp.final_dir.string() + ".publishing")));

  std::ifstream in(tmp.final_dir / "metadata.json");
  nlohmann::json j;
  in >> j;
  REQUIRE(j.at("accepted_time").get<double>() == 1.25);
  REQUIRE(j.at("accepted_increment").get<int>() == 5);
  REQUIRE(j.at("format_version").get<int>() == 1);
  REQUIRE(j.at("method_identity").get<std::string>() == "euler");
}

TEST_CASE("multi-field checkpoint publish success", "[checkpoint][publish]") {
  if (!single_rank()) {
    return;
  }
  TempResultsDir tmp;
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  std::vector<double> v{5.0, 6.0, 7.0, 8.0};
  const auto meta = meta_for({2, 2, 1}, 2.0, 10);

  const auto outcome = publish_checkpoint_directory(
      tmp.final_dir, meta, MPI_COMM_WORLD, [&](const fs::path &dir) {
        write_brick(dir, "u", {2, 2, 1}, u);
        write_brick(dir, "v", {2, 2, 1}, v);
      });
  REQUIRE(outcome.ok);
  REQUIRE(fs::exists(tmp.final_dir / "metadata.json"));
  REQUIRE(fs::file_size(tmp.final_dir / "fields" / "u.bin") == 32);
  REQUIRE(fs::file_size(tmp.final_dir / "fields" / "v.bin") == 32);
}

TEST_CASE("mid-publish failure leaves no final artifact", "[checkpoint][publish]") {
  if (!single_rank()) {
    return;
  }
  TempResultsDir tmp;
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  const auto meta = meta_for({2, 2, 1}, 0.5, 1);
  const fs::path staging = fs::path(tmp.final_dir.string() + ".publishing");

  const auto outcome = publish_checkpoint_directory(
      tmp.final_dir, meta, MPI_COMM_WORLD, [&](const fs::path &dir) {
        write_brick(dir, "u", {2, 2, 1}, u);
        throw std::runtime_error("forced");
      });
  REQUIRE_FALSE(outcome.ok);
  REQUIRE(outcome.message.find("forced") != std::string::npos);
  REQUIRE_FALSE(fs::exists(tmp.final_dir));
  REQUIRE_FALSE(fs::exists(staging));
  REQUIRE(fs::exists(tmp.root));
}

TEST_CASE("mismatched brick size fails publish without final dir",
          "[checkpoint][publish]") {
  if (!single_rank()) {
    return;
  }
  TempResultsDir tmp;
  std::vector<double> bad{1.0, 2.0}; // too short for extents {4,1,1}
  const auto meta = meta_for({4, 1, 1}, 0.0, 0);

  const auto outcome = publish_checkpoint_directory(
      tmp.final_dir, meta, MPI_COMM_WORLD,
      [&](const fs::path &dir) { write_brick(dir, "psi", {4, 1, 1}, bad); });
  REQUIRE_FALSE(outcome.ok);
  REQUIRE_FALSE(fs::exists(tmp.final_dir));
  REQUIRE_FALSE(fs::exists(fs::path(tmp.final_dir.string() + ".publishing")));
}

TEST_CASE("publish refuses an existing final directory", "[checkpoint][publish]") {
  if (!single_rank()) {
    return;
  }
  TempResultsDir tmp;
  fs::create_directories(tmp.final_dir);
  const auto meta = meta_for({4, 1, 1}, 0.0, 0);
  const auto outcome = publish_checkpoint_directory(
      tmp.final_dir, meta, MPI_COMM_WORLD, [](const fs::path &) {});
  REQUIRE_FALSE(outcome.ok);
  REQUIRE(outcome.message.find("already exists") != std::string::npos);
}
