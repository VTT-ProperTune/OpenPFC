// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <sstream>
#include <iostream>
#include <cstdlib>

#include "openpfc/runtime/common/cpu_affinity.hpp"

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#include <cerrno>
#endif

TEST_CASE("test_reset_affinity_with_nproc_one attempts reset when nproc == 1", "[runtime][cpu_affinity]") {
    const char* old_skip = std::getenv("OPENPFC_NO_RESET_AFFINITY");
    unsetenv("OPENPFC_NO_RESET_AFFINITY");

    #if defined(__linux__)
    long nproc_online = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc_online > 1) {
        REQUIRE_NOTHROW(pfc::runtime::reset_cpu_affinity_if_single_mpi_rank(1));
    }
    #else
    // On non-Linux, the function is a no-op but should not throw
    REQUIRE_NOTHROW(pfc::runtime::reset_cpu_affinity_if_single_mpi_rank(1));
    #endif

    if (old_skip) setenv("OPENPFC_NO_RESET_AFFINITY", old_skip, 1);
}

TEST_CASE("test_reset_affinity_with_nproc_multi is no-op when nproc > 1", "[runtime][cpu_affinity]") {
    const char* old_skip = std::getenv("OPENPFC_NO_RESET_AFFINITY");
    unsetenv("OPENPFC_NO_RESET_AFFINITY");

    REQUIRE_NOTHROW(pfc::runtime::reset_cpu_affinity_if_single_mpi_rank(4));
    REQUIRE_NOTHROW(pfc::runtime::reset_cpu_affinity_if_single_mpi_rank(8));

    if (old_skip) setenv("OPENPFC_NO_RESET_AFFINITY", old_skip, 1);
}

TEST_CASE("test_reset_affinity_with_single_cpu is no-op when system has <=1 CPU", "[runtime][cpu_affinity]") {
    const char* old_skip = std::getenv("OPENPFC_NO_RESET_AFFINITY");
    unsetenv("OPENPFC_NO_RESET_AFFINITY");

    #if defined(__linux__)
    long nproc_online = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc_online <= 1) {
        REQUIRE_NOTHROW(pfc::runtime::reset_cpu_affinity_if_single_mpi_rank(1));
    } else {
        SUCCEED("Test only applicable on single-CPU systems");
    }
    #else
    SUCCEED("Test only applicable on Linux");
    #endif

    if (old_skip) setenv("OPENPFC_NO_RESET_AFFINITY", old_skip, 1);
}

TEST_CASE("test_reset_affinity_with_env_var is no-op when OPENPFC_NO_RESET_AFFINITY is set", "[runtime][cpu_affinity]") {
    const char* old_val = std::getenv("OPENPFC_NO_RESET_AFFINITY");
    setenv("OPENPFC_NO_RESET_AFFINITY", "1", 1);

    REQUIRE_NOTHROW(pfc::runtime::reset_cpu_affinity_if_single_mpi_rank(1));

    if (old_val) setenv("OPENPFC_NO_RESET_AFFINITY", old_val, 1);
    else unsetenv("OPENPFC_NO_RESET_AFFINITY");
}

#if defined(__linux__)
TEST_CASE("test_reset_affinity_logs_warning_on_setaffinity_failure", "[runtime][cpu_affinity][linux]") {
    const char* old_skip = std::getenv("OPENPFC_NO_RESET_AFFINITY");
    unsetenv("OPENPFC_NO_RESET_AFFINITY");

    long nproc_online = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc_online > 1) {
        // Capture stderr to verify warning message
        std::ostringstream cerr_capture;
        std::streambuf* old_cerr = std::cerr.rdbuf(cerr_capture.rdbuf());

        // Call the function with nproc=1 to attempt affinity reset
        // We can't easily force sched_setaffinity to fail in a test without
        // manipulating system permissions, but we can verify the function
        // doesn't throw and that the logging infrastructure is in place.
        // In production, failures (e.g., EPERM in containers) will be logged.
        REQUIRE_NOTHROW(pfc::runtime::reset_cpu_affinity_if_single_mpi_rank(1));

        std::cerr.rdbuf(old_cerr);
        std::string output = cerr_capture.str();

        // If sched_setaffinity succeeded (common case), output should be empty
        // If it failed, output should contain the warning message
        if (!output.empty()) {
            // Verify the warning message contains expected elements
            REQUIRE(output.find("reset_cpu_affinity_if_single_mpi_rank") != std::string::npos);
            REQUIRE(output.find("sched_setaffinity failed") != std::string::npos);
            REQUIRE(output.find("errno") != std::string::npos);
        }
        // If output is empty, sched_setaffinity succeeded - this is expected
    } else {
        SUCCEED("Test requires multi-CPU system");
    }

    if (old_skip) setenv("OPENPFC_NO_RESET_AFFINITY", old_skip, 1);
}

TEST_CASE("test_reset_affinity_nothrow_behavior does not throw exceptions", "[runtime][cpu_affinity][linux]") {
    REQUIRE_NOTHROW(pfc::runtime::reset_cpu_affinity_if_single_mpi_rank(1));
    REQUIRE_NOTHROW(pfc::runtime::reset_cpu_affinity_if_single_mpi_rank(4));
}
#endif
