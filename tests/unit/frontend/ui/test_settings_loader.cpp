// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <openpfc/frontend/ui/settings_loader.hpp>
#include <stdexcept>

using pfc::ui::load_settings_file;

namespace {

std::filesystem::path settings_loader_test_dir() {
  auto dir = std::filesystem::path{".temp/tests/unit/frontend/ui/settings_loader"};
  std::filesystem::create_directories(dir);
  return dir;
}

std::filesystem::path write_test_file(const std::string &filename,
                                      const std::string &contents) {
  auto path = settings_loader_test_dir() / filename;
  std::ofstream out(path);
  out << contents;
  return path;
}

} // namespace

TEST_CASE("load_settings_file reads JSON configuration", "[ui][settings]") {
  const auto path = write_test_file("valid.json", R"({"model": "test", "n": 2})");

  const auto settings = load_settings_file(path);

  REQUIRE(settings["model"] == "test");
  REQUIRE(settings["n"] == 2);
}

TEST_CASE("load_settings_file reads TOML configuration", "[ui][settings]") {
  const auto path = write_test_file("valid.toml", "model = \"test\"\nn = 2\n");

  const auto settings = load_settings_file(path);

  REQUIRE(settings["model"] == "test");
  REQUIRE(settings["n"] == 2);
}

TEST_CASE("load_settings_file rejects missing files", "[ui][settings]") {
  const auto path = settings_loader_test_dir() / "missing.json";
  std::filesystem::remove(path);

  REQUIRE_THROWS_AS(load_settings_file(path), std::invalid_argument);
}

TEST_CASE("load_settings_file rejects unsupported extensions", "[ui][settings]") {
  const auto path = write_test_file("settings.yaml", "model: test\n");

  REQUIRE_THROWS_AS(load_settings_file(path), std::invalid_argument);
}

TEST_CASE("load_settings_file reports JSON parse errors", "[ui][settings]") {
  const auto path = write_test_file("invalid.json", R"({"model":)");

  REQUIRE_THROWS_AS(load_settings_file(path), std::runtime_error);
}

TEST_CASE("load_settings_file reports TOML parse errors", "[ui][settings]") {
  const auto path = write_test_file("invalid.toml", "model = \n");

  REQUIRE_THROWS_AS(load_settings_file(path), std::runtime_error);
}
