// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "openpfc/kernel/checkpoint/checkpoint_metadata.hpp"
#include "openpfc/kernel/checkpoint/publish.hpp"

namespace {

namespace fs = std::filesystem;
using pfc::checkpoint::CheckpointMetadata;
using pfc::checkpoint::DomainParams;
using pfc::checkpoint::PublishedFieldBrick;
using pfc::checkpoint::PublishWriteHook;
using pfc::checkpoint::publish_checkpoint_directory;

struct TempResultsDir {
  fs::path root;
  fs::path final_dir;

  TempResultsDir() {
    root = fs::temp_directory_path() / "openpfc_results_checkpoint_publish" /
           ("case_" + std::to_string(
                          reinterpret_cast<std::uintptr_t>(this)));
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

[[nodiscard]] PublishedFieldBrick
make_float64_brick(std::string id, std::array<int, 3> extents,
                   const std::vector<double> &owned) {
  return PublishedFieldBrick{
      .id = std::move(id),
      .dtype = "float64",
      .extents = extents,
      .bytes = std::as_bytes(std::span<const double>{owned}),
      .coordinate_order = "fortran",
  };
}

} // namespace

TEST_CASE("scalar field checkpoint publish success",
          "[checkpoint][publish]") {
  TempResultsDir tmp;
  std::vector<double> psi{1.0, 2.0, 3.0, 4.0};

  CheckpointMetadata meta{
      .format_version = 1,
      .accepted_time = 1.25,
      .accepted_increment = 5,
      .domain =
          DomainParams{
              .global_dimensions = {4, 1, 1},
              .physical_origin = {0.0, 0.0, 0.0},
              .grid_spacing = {1.0, 1.0, 1.0},
          },
      .decomposition = std::nullopt,
      .method_identity = "euler",
  };

  const auto brick = make_float64_brick("psi", {4, 1, 1}, psi);
  const std::array<PublishedFieldBrick, 1> fields{brick};

  const auto outcome =
      publish_checkpoint_directory(tmp.final_dir, meta, fields);
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

TEST_CASE("multi-field checkpoint publish success",
          "[checkpoint][publish]") {
  TempResultsDir tmp;
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  std::vector<double> v{5.0, 6.0, 7.0, 8.0};

  CheckpointMetadata meta{
      .format_version = 1,
      .accepted_time = 2.0,
      .accepted_increment = 10,
      .domain =
          DomainParams{
              .global_dimensions = {2, 2, 1},
              .physical_origin = {0.0, 0.0, 0.0},
              .grid_spacing = {1.0, 1.0, 1.0},
          },
      .method_identity = "euler",
  };

  const auto brick_u = make_float64_brick("u", {2, 2, 1}, u);
  const auto brick_v = make_float64_brick("v", {2, 2, 1}, v);
  const std::array<PublishedFieldBrick, 2> fields{brick_u, brick_v};

  const auto outcome =
      publish_checkpoint_directory(tmp.final_dir, meta, fields);
  REQUIRE(outcome.ok);
  REQUIRE(fs::exists(tmp.final_dir / "metadata.json"));
  REQUIRE(fs::exists(tmp.final_dir / "fields" / "u.bin"));
  REQUIRE(fs::exists(tmp.final_dir / "fields" / "v.bin"));
  REQUIRE(fs::file_size(tmp.final_dir / "fields" / "u.bin") == 32);
  REQUIRE(fs::file_size(tmp.final_dir / "fields" / "v.bin") == 32);
}

TEST_CASE("mid-publish failure leaves no final artifact",
          "[checkpoint][publish]") {
  TempResultsDir tmp;
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  std::vector<double> v{5.0, 6.0, 7.0, 8.0};

  CheckpointMetadata meta{
      .format_version = 1,
      .accepted_time = 0.5,
      .accepted_increment = 1,
      .domain =
          DomainParams{
              .global_dimensions = {2, 2, 1},
              .physical_origin = {0.0, 0.0, 0.0},
              .grid_spacing = {1.0, 1.0, 1.0},
          },
      .method_identity = "euler",
  };

  const auto brick_u = make_float64_brick("u", {2, 2, 1}, u);
  const auto brick_v = make_float64_brick("v", {2, 2, 1}, v);
  const std::array<PublishedFieldBrick, 2> fields{brick_u, brick_v};

  const fs::path staging =
      fs::path(tmp.final_dir.string() + ".publishing");

  PublishWriteHook hook = [](std::size_t field_index,
                             const PublishedFieldBrick &) {
    if (field_index == 1) {
      throw std::runtime_error("forced");
    }
  };

  const auto outcome =
      publish_checkpoint_directory(tmp.final_dir, meta, fields, hook);
  REQUIRE_FALSE(outcome.ok);
  REQUIRE_FALSE(fs::exists(tmp.final_dir));
  REQUIRE_FALSE(fs::exists(staging));
  REQUIRE(fs::exists(tmp.root));
}

TEST_CASE("mismatched brick bytes fail publish without final dir",
          "[checkpoint][publish]") {
  TempResultsDir tmp;
  std::vector<double> bad{1.0, 2.0}; // too short for extents {4,1,1}

  CheckpointMetadata meta{
      .format_version = 1,
      .accepted_time = 0.0,
      .accepted_increment = 0,
      .domain =
          DomainParams{
              .global_dimensions = {4, 1, 1},
              .physical_origin = {0.0, 0.0, 0.0},
              .grid_spacing = {1.0, 1.0, 1.0},
          },
      .method_identity = "euler",
  };

  const auto brick = make_float64_brick("psi", {4, 1, 1}, bad);
  const std::array<PublishedFieldBrick, 1> fields{brick};

  const auto outcome =
      publish_checkpoint_directory(tmp.final_dir, meta, fields);
  REQUIRE_FALSE(outcome.ok);
  REQUIRE_FALSE(fs::exists(tmp.final_dir));
  REQUIRE_FALSE(fs::exists(fs::path(tmp.final_dir.string() + ".publishing")));
}
