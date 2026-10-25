// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <openpfc/frontend/utils/cli_options.hpp>

namespace {

std::vector<char *> argv_from(std::vector<std::string> &args) {
  std::vector<char *> ptrs;
  ptrs.reserve(args.size());
  for (auto &s : args) {
    ptrs.push_back(s.data());
  }
  return ptrs;
}

} // namespace

TEST_CASE("CliOptions parses typed --key=value and bare flags",
          "[frontend][cli_options]") {
  std::vector<std::string> args{"prog",      "--nx=64",       "--dx=0.5",
                                "--elastic", "--csv=out.csv", "--help"};
  auto argv = argv_from(args);
  pfc::utils::CliOptions opt(static_cast<int>(argv.size()), argv.data());

  REQUIRE(opt.help());
  REQUIRE(opt.has("nx"));
  REQUIRE_FALSE(opt.has("ny"));
  REQUIRE(opt.integer("nx", 128) == 64);
  REQUIRE(opt.integer("ny", 128) == 128);
  REQUIRE(opt.real("dx", 1.0) == 0.5);
  REQUIRE(opt.flag("elastic", false));
  REQUIRE(opt.flag("verbose", true));
  REQUIRE(opt.text("csv", "") == "out.csv");
  REQUIRE_NOTHROW(opt.require_all_consumed());
}

TEST_CASE("CliOptions flag treats 0 and false as off", "[frontend][cli_options]") {
  std::vector<std::string> args{"prog", "--elastic=0", "--warm=false"};
  auto argv = argv_from(args);
  pfc::utils::CliOptions opt(static_cast<int>(argv.size()), argv.data());
  REQUIRE_FALSE(opt.flag("elastic", true));
  REQUIRE_FALSE(opt.flag("warm", true));
  REQUIRE_NOTHROW(opt.require_all_consumed());
}

TEST_CASE("CliOptions rejects positional tokens", "[frontend][cli_options]") {
  std::vector<std::string> args{"prog", "128"};
  auto argv = argv_from(args);
  REQUIRE_THROWS_AS(
      pfc::utils::CliOptions(static_cast<int>(argv.size()), argv.data()),
      std::invalid_argument);
}

TEST_CASE("CliOptions require_all_consumed rejects unknown keys",
          "[frontend][cli_options]") {
  std::vector<std::string> args{"prog", "--nx=8", "--lamda=2"};
  auto argv = argv_from(args);
  pfc::utils::CliOptions opt(static_cast<int>(argv.size()), argv.data());
  (void)opt.integer("nx", 1);
  try {
    opt.require_all_consumed();
    FAIL("expected unknown-option throw");
  } catch (const std::invalid_argument &e) {
    REQUIRE(std::string(e.what()).find("lamda") != std::string::npos);
  }
}

TEST_CASE("CliOptions has() does not consume a key", "[frontend][cli_options]") {
  std::vector<std::string> args{"prog", "--lambda-el=1"};
  auto argv = argv_from(args);
  pfc::utils::CliOptions opt(static_cast<int>(argv.size()), argv.data());
  REQUIRE(opt.has("lambda-el"));
  REQUIRE_THROWS_AS(opt.require_all_consumed(), std::invalid_argument);
  REQUIRE(opt.real("lambda-el", 0.0) == 1.0);
  REQUIRE_NOTHROW(opt.require_all_consumed());
}
