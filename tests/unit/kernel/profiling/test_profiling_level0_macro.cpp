// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// Undo compile-definition from OpenPFC so this TU exercises level-0 macros.
#ifdef OPENPFC_PROFILING_LEVEL
#undef OPENPFC_PROFILING_LEVEL
#endif
#define OPENPFC_PROFILING_LEVEL 0

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/profiling/profile_scope_macro.hpp>

TEST_CASE("PFC_PROFILE_SCOPE is no-op at level 0", "[profiling]") {
  PFC_PROFILE_SCOPE("test_zone");
  REQUIRE(true);
}
