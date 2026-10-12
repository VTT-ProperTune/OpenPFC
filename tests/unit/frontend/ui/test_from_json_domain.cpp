// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_from_json_domain.cpp
 * @brief The `domain` block: grid size and the lower and upper box limits.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_world_time.hpp>
#include <openpfc/kernel/data/domain.hpp>

using json = nlohmann::json;
using Catch::Matchers::ContainsSubstring;
using pfc::Domain;
using pfc::Real3;
using pfc::ui::from_json;

namespace {

constexpr double kDx = 1.1107207345395915;

json box(json size, json lower, json upper) {
  return {{"domain", {{"size", size}, {"lower", lower}, {"upper", upper}}}};
}

std::string error_of(const json &j) {
  try {
    (void)from_json<Domain>(j);
  } catch (const std::invalid_argument &e) {
    return e.what();
  }
  return "";
}

} // namespace

TEST_CASE("Domain block gives size, limits and the derived spacing",
          "[domain][json]") {
  const Domain d =
      from_json<Domain>(box({4, 8, 1}, {-2.0, 0.0, 0.0}, {2.0, 4.0, 1.0}));
  REQUIRE(pfc::domain::get_size(d) == pfc::Int3{4, 8, 1});
  REQUIRE(pfc::domain::lower(d) == Real3{-2.0, 0.0, 0.0});
  REQUIRE(pfc::domain::upper(d) == Real3{2.0, 4.0, 1.0});
  REQUIRE(pfc::domain::get_spacing(d) == Real3{1.0, 0.5, 1.0});
  REQUIRE(pfc::domain::get_periodic(d) == pfc::Bool3{true, true, true});
}

TEST_CASE("Domain block may be passed directly", "[domain][json]") {
  const json inner = box({2, 2, 2}, {0, 0, 0}, {1, 1, 1})["domain"];
  REQUIRE(from_json<Domain>(inner) == from_json<Domain>({{"domain", inner}}));
}

TEST_CASE("A symmetric box reproduces the old center origin exactly",
          "[domain][json]") {
  // What scripts/migrate_domain_block.py writes for Lx = 256, dx = kDx,
  // "origin": "center": the old parser put index 0 at -0.5 * dx * Lx.
  const double lo = -0.5 * kDx * 256;
  const Domain d =
      from_json<Domain>(box({256, 256, 256}, {lo, lo, lo}, {-lo, -lo, -lo}));
  REQUIRE(pfc::domain::get_origin(d) == Real3{lo, lo, lo});
  REQUIRE(pfc::domain::get_spacing(d) == Real3{kDx, kDx, kDx});
}

TEST_CASE("Domain size must be three positive integers", "[domain][json]") {
  const json lo = {0, 0, 0}, up = {1, 1, 1};
  for (const json &size : {json{0, 4, 4}, json{4, -1, 4}, json{4, 4},
                           json{4, 4, 4.5}, json("4"), json{4, 4, 4, 4}}) {
    INFO("size: " << size.dump());
    REQUIRE_THAT(error_of(box(size, lo, up)), ContainsSubstring("domain.size"));
  }
  json missing = box({4, 4, 4}, lo, up);
  missing["domain"].erase("size");
  REQUIRE_THAT(error_of(missing), ContainsSubstring("domain.size"));
}

TEST_CASE("Domain limits must be finite and ordered", "[domain][json]") {
  const json size = {4, 4, 4};
  const double inf = std::numeric_limits<double>::infinity();
  REQUIRE_THAT(error_of(box(size, {0, 0, inf}, {1, 1, 1})),
               ContainsSubstring("domain.lower"));
  REQUIRE_THAT(error_of(box(size, {0, 0, 0}, {1, "1", 1})),
               ContainsSubstring("domain.upper"));
  REQUIRE_THAT(error_of(box(size, {0, 0, 0}, {1, 0, 1})),
               ContainsSubstring("each above its lower limit"));
  REQUIRE_THAT(error_of(box(size, {0, 0, 0}, {1, -1, 1})),
               ContainsSubstring("each above its lower limit"));
}

TEST_CASE("The removed Lx/dx/origin keys fail with the equivalent block",
          "[domain][json]") {
  const json legacy = {{"domain",
                        {{"Lx", 256},
                         {"Ly", 256},
                         {"Lz", 256},
                         {"dx", kDx},
                         {"dy", kDx},
                         {"dz", kDx},
                         {"origin", "center"}}}};
  const std::string msg = error_of(legacy);
  REQUIRE_THAT(msg, ContainsSubstring("'Lx'"));
  REQUIRE_THAT(msg, ContainsSubstring("'origin'"));
  REQUIRE_THAT(msg, ContainsSubstring("migrate_domain_block.py"));

  // The suggested block must be the same domain, bit for bit.
  const auto at = msg.find("equivalent to \"domain\": ");
  REQUIRE(at != std::string::npos);
  const auto start = msg.find('{', at);
  const auto end = msg.find("}.", start);
  const json suggested = json::parse(msg.substr(start, end - start + 1));
  const Domain d = from_json<Domain>({{"domain", suggested}});
  const double lo = -0.5 * kDx * 256;
  REQUIRE(pfc::domain::get_origin(d) == Real3{lo, lo, lo});
  REQUIRE(pfc::domain::get_spacing(d) == Real3{kDx, kDx, kDx});
}

TEST_CASE("Removed keys are rejected at the root and next to new keys",
          "[domain][json]") {
  REQUIRE_THAT(error_of({{"Lx", 8}, {"dx", 1.0}, {"origo", "corner"}}),
               ContainsSubstring("'origo'"));
  json mixed = box({4, 4, 4}, {0, 0, 0}, {1, 1, 1});
  mixed["domain"]["origin"] = "corner";
  REQUIRE_THAT(error_of(mixed), ContainsSubstring("'origin'"));
}
