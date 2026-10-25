// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "fixtures/allocation_provider.hpp"
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <optional>
#include <type_traits>
namespace dg = pfc::diagnostics;
static_assert(!std::is_move_assignable_v<dg::AllocationToken>);
static_assert(std::is_move_constructible_v<dg::AllocationToken>);

TEST_CASE("allocation intervals retain exact live ownership and overlap",
          "[diagnostics][memory_traffic]") {
  dg::Traffic d;
  auto first = d.allocation(dg::Space::Device, 5);
  auto second = d.allocation(dg::Space::Device, 7);
  auto totals = d.allocations(dg::Space::Device);
  REQUIRE(totals.requests == 2);
  REQUIRE(totals.live_bytes == 12);
  REQUIRE(totals.peak_live_bytes == 12);
  first.release();
  d.reset_interval();
  totals = d.allocations(dg::Space::Device);
  REQUIRE(totals.requests == 0);
  REQUIRE(totals.initial_live_bytes == 7);
  REQUIRE(totals.peak_live_bytes == 7);
  auto replacement = d.allocation(dg::Space::Device, 11);
  second.release();
  replacement.release();
  totals = d.allocations(dg::Space::Device);
  REQUIRE(totals.requests == 1);
  REQUIRE(totals.frees == 2);
  REQUIRE(totals.peak_live_bytes == 18);
  REQUIRE(totals.live_bytes == 0);
  REQUIRE(d.allocations(dg::Space::Host).requests == 0);
  std::optional<dg::AllocationToken> survivor;
  {
    dg::Traffic temporary;
    survivor.emplace(temporary.allocation(dg::Space::Host, 13));
  }
  survivor->release(); // Stable ledger remains owned after Traffic dies.
}

TEST_CASE("consumer ordinary aligned sized and failed new payload is exact",
          "[diagnostics][memory_traffic]") {
  dg::Traffic d;
  dg::AllocationTotals allocated, released;
  void *first = nullptr, *second = nullptr;
  {
    dg::Scope scope(&d);
    first = ::operator new(5);
    second = ::operator new(7, std::align_val_t{64});
    allocated = d.allocations(dg::Space::Host);
    ::operator delete(first, std::size_t{5});
    ::operator delete(second, std::size_t{7}, std::align_val_t{64});
    released = d.allocations(dg::Space::Host);
  }
  REQUIRE(allocated.requests == 2);
  REQUIRE(allocated.requested_bytes == 12);
  REQUIRE(allocated.peak_live_bytes == 12);
  REQUIRE(reinterpret_cast<std::uintptr_t>(second) % 64 == 0);
  REQUIRE(released.frees == 2);
  REQUIRE(released.freed_bytes == 12);
  REQUIRE(released.live_bytes == 0);
  d.reset_interval();
  void *failed = nullptr;
  {
    dg::Scope scope(&d);
    failed = ::operator new(std::numeric_limits<std::size_t>::max(), std::nothrow);
  }
  REQUIRE(failed == nullptr);
  REQUIRE(d.allocations(dg::Space::Host).requests == 0);
  d.reset_interval();
  void *persistent = nullptr;
  {
    dg::Scope scope(&d);
    persistent = ::operator new(17);
  }
  d.reset_interval();
  REQUIRE(d.allocations(dg::Space::Host).initial_live_bytes == 17);
  ::operator delete(persistent);
  REQUIRE(d.allocations(dg::Space::Host).frees == 1);
  REQUIRE(d.allocations(dg::Space::Host).live_bytes == 0);
}

TEST_CASE("scopes nest and pause restores the enclosing observation",
          "[diagnostics][memory_traffic]") {
  dg::Traffic outer_traffic, inner_traffic;
  REQUIRE(dg::current == nullptr);
  {
    dg::Scope outer(&outer_traffic);
    REQUIRE(dg::current == &outer_traffic);
    {
      dg::Scope inner(&inner_traffic);
      REQUIRE(dg::current == &inner_traffic);
      dg::copy(dg::Direction::HostToDevice, 3);
    }
    REQUIRE(dg::current == &outer_traffic);
    {
      dg::Scope off(nullptr);
      REQUIRE(dg::current == nullptr);
      dg::copy(dg::Direction::HostToDevice, 5);
      auto unobserved = dg::successful_allocation(dg::Space::Device, 5);
      unobserved.release(); // A null token has no ledger to update.
    }
    REQUIRE(dg::current == &outer_traffic);
    {
      dg::PauseObservation paused;
      REQUIRE(dg::current == nullptr);
    }
    REQUIRE(dg::current == &outer_traffic);
  }
  REQUIRE(dg::current == nullptr);
  const auto h2d = static_cast<std::size_t>(dg::Direction::HostToDevice);
  REQUIRE(inner_traffic.copies.calls[h2d] == 1);
  REQUIRE(inner_traffic.copies.bytes[h2d] == 3);
  REQUIRE(outer_traffic.copies.calls[h2d] == 0);
  REQUIRE(outer_traffic.allocations(dg::Space::Device).requests == 0);
}

TEST_CASE("copies are counted per direction and reset with the interval",
          "[diagnostics][memory_traffic]") {
  dg::Traffic d;
  const auto index = [](dg::Direction direction) {
    return static_cast<std::size_t>(direction);
  };
  dg::copy(dg::Direction::HostToDevice, 8); // No current observation.
  {
    dg::Scope scope(&d);
    dg::copy(dg::Direction::HostToDevice, 8);
    dg::copy(dg::Direction::HostToDevice, 16);
    dg::copy(dg::Direction::DeviceToHost, 4);
    dg::copy(dg::Direction::DeviceToDevice, 0); // Zero length: no call.
  }
  REQUIRE(d.copies.calls[index(dg::Direction::HostToDevice)] == 2);
  REQUIRE(d.copies.bytes[index(dg::Direction::HostToDevice)] == 24);
  REQUIRE(d.copies.calls[index(dg::Direction::DeviceToHost)] == 1);
  REQUIRE(d.copies.bytes[index(dg::Direction::DeviceToHost)] == 4);
  REQUIRE(d.copies.calls[index(dg::Direction::DeviceToDevice)] == 0);
  REQUIRE(d.copies.calls[index(dg::Direction::HostToHost)] == 0);
  d.reset_interval();
  for (std::size_t i = 0; i < dg::direction_count; ++i) {
    REQUIRE(d.copies.calls[i] == 0);
    REQUIRE(d.copies.bytes[i] == 0);
  }
}

TEST_CASE("failed free keeps live ownership and completeness sticky",
          "[diagnostics][memory_traffic]") {
  dg::Traffic d;
  auto token = d.allocation(dg::Space::Device, 19);
  token.failed_release();
  REQUIRE_FALSE(d.allocations(dg::Space::Device).complete);
  REQUIRE(d.allocations(dg::Space::Device).live_bytes == 19);
  d.reset_interval();
  REQUIRE_FALSE(d.allocations(dg::Space::Device).complete);
  REQUIRE(d.allocations(dg::Space::Device).initial_live_bytes == 19);
  token.release();
  REQUIRE(d.allocations(dg::Space::Device).live_bytes == 0);
  REQUIRE_FALSE(d.allocations(dg::Space::Device).complete);
  REQUIRE(d.allocations(dg::Space::Host).complete);
}
