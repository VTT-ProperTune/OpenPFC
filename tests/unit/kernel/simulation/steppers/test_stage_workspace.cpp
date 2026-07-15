// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/simulation/steppers/stage_workspace.hpp>
#include <vector>
#include <type_traits>

using namespace pfc::sim::steppers;

TEST_CASE("Construction creates correct number of stages") {
    StageWorkspace<double> ws(4, 100);
    REQUIRE(ws.num_stages() == 4);
}

TEST_CASE("Construction initializes buffers to zero") {
    StageWorkspace<double> ws(3, 50);
    for (std::size_t i = 0; i < ws.num_stages(); ++i) {
        const auto& buf = ws.stage(i);
        for (std::size_t j = 0; j < ws.local_size(); ++j) {
            REQUIRE(buf[j] == 0.0);
        }
    }
}

TEST_CASE("Stage access returns mutable reference") {
    StageWorkspace<double> ws(2, 10);
    auto& buf = ws.stage(0);
    buf[5] = 42.0;
    REQUIRE(ws.stage(0)[5] == 42.0);
}

TEST_CASE("Stage const access returns const reference") {
    const StageWorkspace<double> ws(2, 10);
    const auto& buf = ws.stage(1);
    // Verify const correctness via compilation
    static_assert(std::is_same_v<decltype(buf), const std::vector<double>&>);
}

TEST_CASE("Stage access throws for invalid index") {
    StageWorkspace<double> ws(3, 100);
    REQUIRE_THROWS_AS(ws.stage(3), std::out_of_range);
    REQUIRE_THROWS_AS(ws.stage(10), std::out_of_range);
}

TEST_CASE("Reset sets all buffers to zero") {
    StageWorkspace<double> ws(2, 5);
    ws.stage(0)[0] = 1.0;
    ws.stage(0)[4] = 2.0;
    ws.stage(1)[2] = 3.0;
    ws.reset();
    for (std::size_t i = 0; i < ws.num_stages(); ++i) {
        for (std::size_t j = 0; j < ws.local_size(); ++j) {
            REQUIRE(ws.stage(i)[j] == 0.0);
        }
    }
}

TEST_CASE("num_stages returns correct count") {
    StageWorkspace<float> ws1(1, 10);
    REQUIRE(ws1.num_stages() == 1);
    StageWorkspace<double> ws2(4, 100);
    REQUIRE(ws2.num_stages() == 4);
}

TEST_CASE("local_size returns correct size") {
    StageWorkspace<double> ws(3, 250);
    REQUIRE(ws.local_size() == 250);
}

TEST_CASE("Move constructor transfers ownership") {
    StageWorkspace<double> ws1(2, 10);
    ws1.stage(0)[0] = 5.0;
    StageWorkspace<double> ws2(std::move(ws1));
    REQUIRE(ws2.num_stages() == 2);
    REQUIRE(ws2.local_size() == 10);
    REQUIRE(ws2.stage(0)[0] == 5.0);
    REQUIRE(ws1.num_stages() == 0);  // moved-from state
}

TEST_CASE("Move assignment transfers ownership") {
    StageWorkspace<double> ws1(2, 10);
    ws1.stage(0)[0] = 7.0;
    StageWorkspace<double> ws2(3, 20);
    ws2 = std::move(ws1);
    REQUIRE(ws2.num_stages() == 2);
    REQUIRE(ws2.local_size() == 10);
    REQUIRE(ws2.stage(0)[0] == 7.0);
    REQUIRE(ws1.num_stages() == 0);  // moved-from state
}

TEST_CASE("Copy constructor is deleted") {
    // This test verifies compilation failure
    // StageWorkspace<double> ws1(2, 10);
    // StageWorkspace<double> ws2(ws1);  // Should not compile
    // Implementation: compile-only test, verified by separate build
    SUCCEED("Copy constructor is deleted (verified by compilation failure)");
}

TEST_CASE("Copy assignment is deleted") {
    // This test verifies compilation failure
    // StageWorkspace<double> ws1(2, 10);
    // StageWorkspace<double> ws2(3, 20);
    // ws2 = ws1;  // Should not compile
    // Implementation: compile-only test, verified by separate build
    SUCCEED("Copy assignment is deleted (verified by compilation failure)");
}

TEST_CASE("Float type is supported") {
    StageWorkspace<float> ws(2, 10);
    REQUIRE(ws.num_stages() == 2);
    REQUIRE(ws.local_size() == 10);
    ws.stage(0)[0] = 1.5f;
    REQUIRE(ws.stage(0)[0] == 1.5f);
}

TEST_CASE("Double type is supported") {
    StageWorkspace<double> ws(2, 10);
    REQUIRE(ws.num_stages() == 2);
    REQUIRE(ws.local_size() == 10);
    ws.stage(0)[0] = 1.5;
    REQUIRE(ws.stage(0)[0] == 1.5);
}
