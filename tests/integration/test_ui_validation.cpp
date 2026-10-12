// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/ui.hpp>
#include <openpfc/kernel/data/domain.hpp>

using json = nlohmann::json;
using namespace pfc;

namespace {
json domain_block(json size, json lower, json upper) {
  return {{"domain", {{"size", size}, {"lower", lower}, {"upper", upper}}}};
}
} // namespace

TEST_CASE("Domain validation with a non-integer size",
          "[ui][integration][validation]") {
  json j = domain_block({256.5, 256, 256}, {0, 0, 0}, {1, 1, 1});

  try {
    [[maybe_unused]] auto world = ui::from_json<Domain>(j);
    FAIL("Should have thrown exception");
  } catch (const std::invalid_argument &e) {
    std::string msg(e.what());
    REQUIRE((msg.find("domain.size") != std::string::npos));
    REQUIRE((msg.find("integers") != std::string::npos));
    REQUIRE((msg.find("256.5") != std::string::npos));
    REQUIRE((msg.find("Description") != std::string::npos));
    REQUIRE((msg.find("Expected") != std::string::npos));
    REQUIRE((msg.find("Got") != std::string::npos));
    REQUIRE((msg.find("Example") != std::string::npos));
  }
}

TEST_CASE("Domain validation with a missing upper limit",
          "[ui][integration][validation]") {
  json j = domain_block({256, 256, 256}, {0, 0, 0}, {1, 1, 1});
  j["domain"].erase("upper");

  try {
    [[maybe_unused]] auto world = ui::from_json<Domain>(j);
    FAIL("Should have thrown exception");
  } catch (const std::invalid_argument &e) {
    std::string msg(e.what());
    REQUIRE((msg.find("domain.upper") != std::string::npos));
    REQUIRE((msg.find("missing") != std::string::npos));
    REQUIRE((msg.find("upper limits of the box") != std::string::npos));
  }
}

TEST_CASE("Domain validation with a string lower limit",
          "[ui][integration][validation]") {
  json j = domain_block({256, 256, 256}, {"0.0", 0, 0}, {1, 1, 1});

  try {
    [[maybe_unused]] auto world = ui::from_json<Domain>(j);
    FAIL("Should have thrown exception");
  } catch (const std::invalid_argument &e) {
    std::string msg(e.what());
    REQUIRE((msg.find("domain.lower") != std::string::npos));
    REQUIRE((msg.find("finite numbers") != std::string::npos));
  }
}

TEST_CASE("Unknown field modifier type error", "[ui][integration][validation]") {
  json params = {
      {"type", "random_seed"} // Typo: should be "random_seeds"
  };

  try {
    auto modifier = ui::create_field_modifier("random_seed", params);
    FAIL("Should have thrown exception");
  } catch (const std::invalid_argument &e) {
    std::string msg(e.what());
    REQUIRE((msg.find("type") != std::string::npos));
    REQUIRE((msg.find("random_seed") != std::string::npos));
    REQUIRE((msg.find("constant") != std::string::npos));
    REQUIRE((msg.find("single_seed") != std::string::npos));
    REQUIRE((msg.find("random_seeds") != std::string::npos));
    REQUIRE((msg.find("from_file") != std::string::npos));
  }
}

TEST_CASE("Error messages are concise", "[ui][integration][validation]") {
  json j = domain_block({256.5, 256, 256}, {0, 0, 0}, {1, 1, 1});

  try {
    [[maybe_unused]] auto world = ui::from_json<Domain>(j);
    FAIL("Should have thrown exception");
  } catch (const std::invalid_argument &e) {
    std::string msg(e.what());
    // Count newlines - should be < 10 lines
    const auto lines =
        1 + static_cast<int>(std::count(msg.begin(), msg.end(), '\n'));
    REQUIRE((lines <= 10));
  }
}

TEST_CASE("Domain validation accepts integer limits",
          "[ui][integration][validation]") {
  json j = domain_block({256, 128, 1}, {0, -64, 0}, {256, 64, 1});
  REQUIRE_NOTHROW(ui::from_json<Domain>(j));
  Domain domain = ui::from_json<Domain>(j);
  REQUIRE(domain::get_spacing(domain) == Real3{1.0, 1.0, 1.0});
  REQUIRE(domain::get_origin(domain) == Real3{0.0, -64.0, 0.0});
}
