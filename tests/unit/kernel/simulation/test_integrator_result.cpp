// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/simulation/integrator_result.hpp>
#include <string>

using pfc::sim::steppers::IntegratorResult;

TEST_CASE("test_accepted_factory_returns_correct_new_time") {
  const auto result = IntegratorResult::make_accepted(1.5);

  REQUIRE(result.accepted == true);
  REQUIRE(result.new_time == 1.5);
  REQUIRE(result.error_estimate == 0.0);
  REQUIRE(result.rejection_reason.empty());
}

TEST_CASE("test_rejected_factory_sets_accepted_false_and_reason") {
  const auto result = IntegratorResult::make_rejected(1.0, "error too large");

  REQUIRE(result.accepted == false);
  REQUIRE(result.new_time == 1.0);
  REQUIRE(result.error_estimate == 0.0);
  REQUIRE(result.rejection_reason == "error too large");
}

TEST_CASE("test_default_construction_initializes_error_estimate_to_zero") {
  IntegratorResult result;

  REQUIRE(result.error_estimate == 0.0);
}

TEST_CASE("test_accepted_and_rejected_factories_set_last_step_complete_false") {
  const auto ok = IntegratorResult::make_accepted(2.0);
  const auto bad = IntegratorResult::make_rejected(1.0, "too large");

  REQUIRE(ok.last_step_complete == false);
  REQUIRE(bad.last_step_complete == false);
}
