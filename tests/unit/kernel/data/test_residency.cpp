// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Host-only unit tests for the M2.2 residency state machine. These pin the
// host<->device coherence logic that Field consults before transferring, so
// the correctness core is verified without a GPU (CI compiles but never runs
// device code).

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/residency.hpp>

using pfc::data::Residency;

TEST_CASE("Residency: a host-only field never needs a transfer",
          "[residency][unit]") {
  auto r = Residency::host_only();
  REQUIRE_FALSE(r.two_sided());
  REQUIRE(r.host_valid());
  REQUIRE_FALSE(r.host_needs_refresh());
  REQUIRE_FALSE(r.device_needs_refresh());

  // Writes on a one-sided field never mark anything stale.
  r.note_host_write();
  REQUIRE(r.host_valid());
  REQUIRE_FALSE(r.host_needs_refresh());
  REQUIRE_FALSE(r.device_needs_refresh());
}

TEST_CASE("Residency: a fresh device-backed field must push host->device",
          "[residency][unit]") {
  // The audit-4.1 scenario: ICs are seeded on the host mirror; the device copy
  // is stale, so integrating on the device WITHOUT a push evolves garbage.
  auto r = Residency::device_backed();
  REQUIRE(r.two_sided());
  REQUIRE(r.host_valid());
  REQUIRE_FALSE(r.device_valid());
  REQUIRE(r.device_needs_refresh()); // <- the sync the bug omitted
  REQUIRE_FALSE(r.host_needs_refresh());
}

TEST_CASE("Residency: a device write makes the host mirror stale",
          "[residency][unit]") {
  auto r = Residency::device_backed();
  r.note_synced(); // both agree after the initial push
  REQUIRE_FALSE(r.device_needs_refresh());
  REQUIRE_FALSE(r.host_needs_refresh());

  // The device integrates a step -> host mirror now stale.
  r.note_device_write();
  REQUIRE(r.device_valid());
  REQUIRE_FALSE(r.host_valid());
  REQUIRE(r.host_needs_refresh()); // a writer/host read must pull first
  REQUIRE_FALSE(r.device_needs_refresh());
}

TEST_CASE("Residency: alternating writes keep exactly one side authoritative",
          "[residency][unit]") {
  auto r = Residency::device_backed();
  r.note_synced();

  r.note_host_write(); // host modifier applied
  REQUIRE(r.host_valid());
  REQUIRE_FALSE(r.device_valid());
  REQUIRE(r.device_needs_refresh());

  r.note_synced();       // pushed
  r.note_device_write(); // device step
  REQUIRE(r.device_valid());
  REQUIRE_FALSE(r.host_valid());
  REQUIRE(r.host_needs_refresh());
}

TEST_CASE("Residency: note_synced clears both refresh flags", "[residency][unit]") {
  auto r = Residency::device_backed();
  r.note_device_write();
  REQUIRE(r.host_needs_refresh());
  r.note_synced();
  REQUIRE(r.host_valid());
  REQUIRE(r.device_valid());
  REQUIRE_FALSE(r.host_needs_refresh());
  REQUIRE_FALSE(r.device_needs_refresh());
}

// Compile-time sanity: the whole machine is constexpr-usable.
static_assert(Residency::host_only().host_valid());
static_assert(!Residency::host_only().two_sided());
static_assert(Residency::device_backed().device_needs_refresh());
