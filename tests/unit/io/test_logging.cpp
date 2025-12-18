// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <sstream>

#include "openpfc/logging.hpp"

using namespace pfc;

struct StreamRedirect {
  std::ostream &os;
  std::streambuf *old_buf;
  std::ostringstream captured;
  explicit StreamRedirect(std::ostream &target)
      : os(target), old_buf(target.rdbuf()) {
    os.rdbuf(captured.rdbuf());
  }
  ~StreamRedirect() { os.rdbuf(old_buf); }
};

TEST_CASE("logging - level filtering", "[logging][unit]") {
  Logger lg{LogLevel::Info, 0};

  // Debug should be filtered when min level is Info
  {
    StreamRedirect redirect(std::clog);
    log(lg, LogLevel::Debug, "debug message");
    REQUIRE(redirect.captured.str().empty());
  }

  // Info should pass
  {
    StreamRedirect redirect(std::clog);
    log_info(lg, "hello info");
    auto out = redirect.captured.str();
    REQUIRE_FALSE(out.empty());
    REQUIRE(out.find("[INFO]") != std::string::npos);
    REQUIRE(out.find("hello info") != std::string::npos);
  }
}

TEST_CASE("logging - stream targets", "[logging][unit]") {
  Logger lg{LogLevel::Debug, 3};

  // Info/Debug to std::clog
  {
    StreamRedirect redirect(std::clog);
    log_debug(lg, "dbg");
    log_info(lg, "inf");
    auto out = redirect.captured.str();
    REQUIRE(out.find("[DEBUG]") != std::string::npos);
    REQUIRE(out.find("[INFO]") != std::string::npos);
  }

  // Warning/Error to std::cerr
  {
    StreamRedirect redirect(std::cerr);
    log_warning(lg, "warn");
    log_error(lg, "err");
    auto out = redirect.captured.str();
    REQUIRE(out.find("[WARN]") != std::string::npos);
    REQUIRE(out.find("[ERROR]") != std::string::npos);
  }
}

TEST_CASE("logging - rank prefix", "[logging][unit]") {
  // With rank
  {
    Logger lg{LogLevel::Info, 7};
    StreamRedirect redirect(std::clog);
    log_info(lg, "message");
    auto out = redirect.captured.str();
    REQUIRE(out.find("rank 7:") != std::string::npos);
  }

  // Without rank (negative indicates unknown)
  {
    Logger lg{LogLevel::Info, -1};
    StreamRedirect redirect(std::clog);
    log_info(lg, "message");
    auto out = redirect.captured.str();
    REQUIRE(out.find("rank ") == std::string::npos);
  }
}
