// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <catch2/catch_config.hpp>
#include <catch2/catch_session.hpp>

namespace pfc::test {

// Every rank must visit the same cases, sections and random generators.
// Parse CLI options after these defaults so explicit order/seed choices still work.
// MPI initialization and finalization remain the caller's responsibility.
inline int run(int argc, char *argv[]) {
  Catch::Session session;
  session.configData().runOrder = Catch::TestRunOrder::Declared;
  session.configData().rngSeed = 1u;
  return session.run(argc, argv);
}

} // namespace pfc::test
