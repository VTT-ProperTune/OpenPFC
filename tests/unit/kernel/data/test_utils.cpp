// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <mpi.h>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>

using namespace pfc;

TEST_CASE("utils - string formatting", "[utils][unit]") {
  SECTION("Format string with integer") {
    std::string result = utils::string_format("file_%03d.dat", 42);
    REQUIRE(result == "file_042.dat");
  }

  SECTION("Format string with multiple integers") {
    std::string result = utils::string_format("step_%d_rank_%d.dat", 100, 3);
    REQUIRE(result == "step_100_rank_3.dat");
  }

  SECTION("Format string with floating point") {
    std::string result = utils::string_format("value_%.2f.txt", 3.14159);
    REQUIRE(result == "value_3.14.txt");
  }

  SECTION("Format plain string without placeholders") {
    std::string result = utils::string_format("plain_file.dat");
    REQUIRE(result == "plain_file.dat");
  }

  SECTION("Handles encoding errors") {
    // Use an invalid format specifier that triggers snprintf to return negative
    // Different snprintf implementations may behave differently; this is a common case
    REQUIRE_THROWS_AS(
      utils::string_format("%lc", static_cast<wchar_t>(0xD800)),  // invalid UTF-16 surrogate
      std::runtime_error
    );
  }
}

TEST_CASE("utils - format with number", "[utils][unit]") {
  SECTION("Applies formatting when placeholder present") {
    std::string result = utils::format_with_number("output_%04d.dat", 7);
    REQUIRE(result == "output_0007.dat");
  }

  SECTION("Returns filename unchanged when no placeholder") {
    std::string result = utils::format_with_number("output.dat", 999);
    REQUIRE(result == "output.dat");
  }

  SECTION("Works with different integer values") {
    std::string result1 = utils::format_with_number("step_%d.dat", 0);
    REQUIRE(result1 == "step_0.dat");

    std::string result2 = utils::format_with_number("step_%d.dat", 999);
    REQUIRE(result2 == "step_999.dat");
  }
}

TEST_CASE("utils - sizeof_vec", "[utils][unit]") {
  SECTION("Computes correct size for int vector") {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    REQUIRE(utils::sizeof_vec(vec) == 5 * sizeof(int));
  }

  SECTION("Computes correct size for double vector") {
    std::vector<double> vec(100);
    REQUIRE(utils::sizeof_vec(vec) == 100 * sizeof(double));
  }

  SECTION("Returns zero for empty vector") {
    std::vector<float> vec;
    REQUIRE(utils::sizeof_vec(vec) == 0);
  }
}

TEST_CASE("utils - format_with_number validation rejects dangerous patterns", "[utils][unit]") {
  SECTION("Rejects %n specifier") {
    REQUIRE_THROWS_AS(utils::format_with_number("output_%n.dat", 7), std::invalid_argument);
    try {
      utils::format_with_number("output_%n.dat", 7);
      FAIL("Should have thrown std::invalid_argument");
    } catch (const std::invalid_argument &e) {
      std::string msg = e.what();
      REQUIRE(msg.find("invalid filename pattern") != std::string::npos);
      REQUIRE(msg.find("%n") != std::string::npos);
    }
  }

  SECTION("Rejects %s specifier") {
    REQUIRE_THROWS_AS(utils::format_with_number("file_%s.txt", 1), std::invalid_argument);
  }

  SECTION("Rejects %f specifier") {
    REQUIRE_THROWS_AS(utils::format_with_number("data_%f.bin", 42), std::invalid_argument);
  }

  SECTION("Rejects multiple %d specifiers") {
    REQUIRE_THROWS_AS(utils::format_with_number("step_%d_rank_%d.dat", 1), std::invalid_argument);
  }

  SECTION("Rejects mixed specifiers (%d and %f)") {
    REQUIRE_THROWS_AS(utils::format_with_number("out_%d_%f.dat", 1), std::invalid_argument);
  }

  SECTION("Rejects %x specifier") {
    REQUIRE_THROWS_AS(utils::format_with_number("hex_%x.dat", 10), std::invalid_argument);
  }

  SECTION("Rejects %p specifier") {
    REQUIRE_THROWS_AS(utils::format_with_number("ptr_%p.dat", 1), std::invalid_argument);
  }
}

TEST_CASE("utils - format_with_number validation accepts valid patterns", "[utils][unit]") {
  SECTION("Accepts %d and produces correct output") {
    std::string result = utils::format_with_number("step_%d.dat", 123);
    REQUIRE(result == "step_123.dat");
  }

  SECTION("Accepts %04d (zero-padded) and produces correct output") {
    std::string result = utils::format_with_number("frame_%04d.vti", 7);
    REQUIRE(result == "frame_0007.vti");
  }

  SECTION("Accepts %-5d (left-align) and produces correct output") {
    std::string result = utils::format_with_number("result_%-5d.bin", 42);
    REQUIRE(result == "result_42   .bin");  // Note: left-align with space padding to width 5
  }

  SECTION("Accepts pattern with no % specifier unchanged") {
    std::string result = utils::format_with_number("output_fixed.dat", 999);
    REQUIRE(result == "output_fixed.dat");
  }

  SECTION("Accepts pattern with % in middle of text") {
    std::string result = utils::format_with_number("data_frame_%d_raw", 5);
    REQUIRE(result == "data_frame_5_raw");
  }
}

TEST_CASE("mpi - get comm rank and size", "[mpi][unit]") {
  SECTION("Get rank returns valid rank") {
    int rank = mpi::get_comm_rank(MPI_COMM_WORLD);
    REQUIRE(rank >= 0);
  }

  SECTION("Get size returns positive size") {
    int size = mpi::get_comm_size(MPI_COMM_WORLD);
    REQUIRE(size >= 1);
  }

  SECTION("Rank is less than size") {
    int rank = mpi::get_comm_rank(MPI_COMM_WORLD);
    int size = mpi::get_comm_size(MPI_COMM_WORLD);
    REQUIRE(rank < size);
  }
}
