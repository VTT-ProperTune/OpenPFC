// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_halo_persistent_mpi_errors.cpp
 * @brief Contract tests: PersistentHaloExchanger start/wait propagate MPI errors.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

std::string read_file(const std::filesystem::path &path) {
  std::ifstream in(path);
  REQUIRE(in.good());
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
}

/// Extract the brace-balanced body of a method whose signature starts at
/// `sig` (e.g. "void start_exchange()"). Returns the interior of the braces.
std::string extract_method_body(std::string_view src, std::string_view sig) {
  const auto sig_pos = src.find(sig);
  REQUIRE(sig_pos != std::string_view::npos);
  const auto brace = src.find('{', sig_pos + sig.size());
  REQUIRE(brace != std::string_view::npos);
  int depth = 0;
  for (std::size_t i = brace; i < src.size(); ++i) {
    if (src[i] == '{') {
      ++depth;
    } else if (src[i] == '}') {
      --depth;
      if (depth == 0) {
        return std::string(src.substr(brace + 1, i - brace - 1));
      }
    }
  }
  FAIL("unbalanced braces after " << sig);
  return {};
}

} // namespace

TEST_CASE("throw_on_mpi_error throws on non-success and no-ops on success",
          "[halo][persistent][mpi_errors]") {
  REQUIRE_NOTHROW(pfc::mpi::throw_on_mpi_error(MPI_SUCCESS, "MPI_Waitall"));

  REQUIRE_THROWS_AS(pfc::mpi::throw_on_mpi_error(MPI_ERR_OTHER, "MPI_Waitall"),
                    std::runtime_error);
  try {
    pfc::mpi::throw_on_mpi_error(MPI_ERR_OTHER, "MPI_Waitall");
    FAIL("expected throw");
  } catch (const std::runtime_error &e) {
    REQUIRE_THAT(e.what(), Catch::Matchers::ContainsSubstring("MPI_Waitall"));
  }
}

TEST_CASE("PersistentHaloExchanger start/wait propagate MPI errors (source "
          "contract)",
          "[halo][persistent][mpi_errors]") {
  const auto header =
      std::filesystem::path(__FILE__).parent_path() /
      "../../../../include/openpfc/kernel/decomposition/halo_persistent.hpp";
  REQUIRE(std::filesystem::exists(header));

  const std::string src = read_file(header);
  const std::string start_body = extract_method_body(src, "void start_exchange()");
  const std::string wait_body = extract_method_body(src, "void wait_exchange()");

  REQUIRE(start_body.find("MPI_Startall") != std::string::npos);
  REQUIRE(start_body.find("throw_on_mpi_error") != std::string::npos);

  REQUIRE(wait_body.find("MPI_Waitall") != std::string::npos);
  REQUIRE(wait_body.find("throw_on_mpi_error") != std::string::npos);
}
