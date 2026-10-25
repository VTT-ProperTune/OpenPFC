// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_traffic.hpp
 * @brief Optional scoped allocation and explicit copy accounting.
 *
 * A caller-owned `pfc::diagnostics::Traffic` observation, made current on
 * the calling thread with `Scope`, counts successful host/device
 * allocations and explicit host/device copies. GPU `DataBuffer` reports its
 * device allocations and uploads/downloads; host allocations are reported
 * only through a consumer-owned global `new`/`delete` provider.
 *
 * @see docs/reference/memory_traffic.md
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace pfc::diagnostics {

class Traffic;

enum class Space { Host, Device };
enum class Direction {
  HostToHost,
  HostToDevice,
  DeviceToHost,
  DeviceToDevice,
  Count
};
inline constexpr auto direction_count = static_cast<std::size_t>(Direction::Count);

struct AllocationTotals {
  std::uint64_t requests = 0, frees = 0;
  std::uint64_t requested_bytes = 0, freed_bytes = 0;
  std::uint64_t initial_live_bytes = 0, live_bytes = 0, peak_live_bytes = 0;
  bool complete = true; ///< False after an observed runtime free failure.
};

struct AllocationLedger {
  struct Counters {
    std::atomic<std::uint64_t> requests{0}, frees{0}, requested{0}, freed{0};
    std::atomic<std::uint64_t> initial{0}, live{0}, peak{0};
    std::atomic<bool> complete{true};
    void allocate(std::size_t bytes) noexcept {
      ++requests;
      requested += bytes;
      const auto now = live.fetch_add(bytes) + bytes;
      auto old = peak.load();
      while (old < now && !peak.compare_exchange_weak(old, now)) {
      }
    }
    void release(std::size_t bytes) noexcept {
      ++frees;
      freed += bytes;
      live -= bytes;
    }
    AllocationTotals snapshot() const noexcept {
      return {requests.load(), frees.load(), requested.load(), freed.load(),
              initial.load(),  live.load(),  peak.load(),      complete.load()};
    }
    // Only at a quiescent observation boundary; never erase live ownership.
    void reset_interval() noexcept {
      requests = 0;
      frees = 0;
      requested = 0;
      freed = 0;
      const auto baseline = live.load();
      initial = baseline;
      peak = baseline;
    }
  };
  Counters host, device;
};

/// A successful allocation captures stable ledger ownership. This token may
/// outlive Traffic; it never stores a pointer into a stack observation.
/// Call release() only after the matching actual free succeeds.
class AllocationToken {
  std::shared_ptr<AllocationLedger> ledger_;
  Space space_ = Space::Host;
  std::size_t bytes_ = 0;

public:
  AllocationToken() = default;
  AllocationToken(std::shared_ptr<AllocationLedger> ledger, Space space,
                  std::size_t bytes)
      : ledger_(std::move(ledger)), space_(space), bytes_(bytes) {
    if (ledger_)
      (space_ == Space::Host ? ledger_->host : ledger_->device).allocate(bytes_);
  }
  AllocationToken(const AllocationToken &) = delete;
  AllocationToken &operator=(const AllocationToken &) = delete;
  AllocationToken(AllocationToken &&) noexcept = default;
  AllocationToken &operator=(AllocationToken &&) = delete;
  void failed_release() noexcept {
    if (ledger_)
      (space_ == Space::Host ? ledger_->host : ledger_->device).complete = false;
  }
  void release() noexcept {
    if (!ledger_) return;
    (space_ == Space::Host ? ledger_->host : ledger_->device).release(bytes_);
    ledger_.reset();
    bytes_ = 0;
  }
};

/// Observation that receives allocations and copies on this thread.
inline thread_local Traffic *current = nullptr;

/// Disable observation for the lifetime of this object, then restore it.
class PauseObservation {
  Traffic *previous_;

public:
  PauseObservation() : previous_(current) { current = nullptr; }
  ~PauseObservation() { current = previous_; }
  PauseObservation(const PauseObservation &) = delete;
  PauseObservation &operator=(const PauseObservation &) = delete;
};

/// Make an observation current for the lifetime of this object.
class Scope {
  Traffic *previous_;

public:
  /// Null explicitly disables this scope, including an enclosing observation.
  explicit Scope(Traffic *observation) : previous_(current) {
    current = observation;
  }
  ~Scope() { current = previous_; }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
};

/// Successful explicit copy calls and bytes, indexed by Direction.
struct Copies {
  std::uint64_t calls[direction_count]{};
  std::uint64_t bytes[direction_count]{};
};

/// Optional caller-owned observation. A full host allocation claim requires
/// a consumer new/delete provider using successful_allocation() and tokens.
/// reset_interval() retains the exact live baseline. Bookkeeping and
/// runtime-reserved memory are excluded.
class Traffic {
  std::shared_ptr<AllocationLedger> ledger_;

public:
  Copies copies{};
  Traffic() {
    PauseObservation pause;
    ledger_ = std::make_shared<AllocationLedger>();
  }
  AllocationTotals allocations(Space space) const noexcept {
    return (space == Space::Host ? ledger_->host : ledger_->device).snapshot();
  }
  AllocationToken allocation(Space space, std::size_t bytes) {
    return {ledger_, space, bytes};
  }
  void reset_interval() noexcept {
    ledger_->host.reset_interval();
    ledger_->device.reset_interval();
    copies = {};
  }
};

/// Provider ABI: call only after an actual successful allocation; retain the
/// move-only token with that allocation and release it after successful free.
inline AllocationToken successful_allocation(Space space, std::size_t bytes) {
  return current ? current->allocation(space, bytes) : AllocationToken{};
}

/// Record one successful explicit copy. Zero-length copies are not counted.
inline void copy(Direction direction, std::size_t bytes) noexcept {
  if (!current || !bytes) return;
  const auto d = static_cast<std::size_t>(direction);
  ++current->copies.calls[d];
  current->copies.bytes[d] += bytes;
}

} // namespace pfc::diagnostics
