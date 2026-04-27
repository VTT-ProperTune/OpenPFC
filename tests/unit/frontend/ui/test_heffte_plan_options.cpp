// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <heffte.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/from_json.hpp>
#include <stdexcept>
#include <type_traits>

using json = nlohmann::json;
using pfc::ui::from_json;

TEST_CASE("from_json parses HeFFTe reshape algorithm", "[ui][heffte]") {
  const json config = {{"reshape_algorithm", "p2p"}};

  const auto options = from_json<heffte::plan_options>(config);

  using AlgorithmType = std::underlying_type_t<heffte::reshape_algorithm>;
  REQUIRE(static_cast<AlgorithmType>(options.algorithm) ==
          static_cast<AlgorithmType>(heffte::reshape_algorithm::p2p));
}

TEST_CASE("from_json rejects unknown HeFFTe reshape algorithm", "[ui][heffte]") {
  const json config = {{"reshape_algorithm", "typo"}};

  REQUIRE_THROWS_AS(from_json<heffte::plan_options>(config), std::invalid_argument);
}
