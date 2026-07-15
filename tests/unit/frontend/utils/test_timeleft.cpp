// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <openpfc/frontend/utils/timeleft.hpp>

using namespace pfc::utils;

TEST_CASE("test_timeleft_zero_seconds") {
    TimeLeft tl(0.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "0s");
}

TEST_CASE("test_timeleft_single_second") {
    TimeLeft tl(1.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "1s");
}

TEST_CASE("test_timeleft_exactly_one_minute") {
    TimeLeft tl(60.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "1m 0s");
}

TEST_CASE("test_timeleft_thirty_seconds") {
    TimeLeft tl(30.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "30s");
}

TEST_CASE("test_timeleft_exactly_one_hour") {
    TimeLeft tl(3600.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "1h 0m 0s");
}

TEST_CASE("test_timeleft_ninety_minutes") {
    TimeLeft tl(5400.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "1h 30m 0s");
}

TEST_CASE("test_timeleft_exactly_one_day") {
    TimeLeft tl(86400.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "1d 0h 0m 0s");
}

TEST_CASE("test_timeleft_thirty_hours") {
    TimeLeft tl(108000.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "1d 6h 0m 0s");
}

TEST_CASE("test_timeleft_complex_case") {
    TimeLeft tl(3665.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "1h 1m 5s");
}

TEST_CASE("test_timeleft_one_day_two_hours") {
    TimeLeft tl(93600.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "1d 2h 0m 0s");
}

TEST_CASE("test_timeleft_partial_units") {
    TimeLeft tl(90061.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "1d 1h 1m 1s");
}

TEST_CASE("test_timeleft_hours_with_seconds") {
    TimeLeft tl(3661.0);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "1h 1m 1s");
}

TEST_CASE("test_timeleft_floor_behavior") {
    TimeLeft tl(59.9);
    std::ostringstream oss;
    oss << tl;
    REQUIRE(oss.str() == "59s");
}
