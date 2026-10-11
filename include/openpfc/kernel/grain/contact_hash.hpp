// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pfc::grain {

/// Empty open-addressed slot. Packed keys never use 0: endpoints are ordered
/// `low < high`, so a self-contact cannot be inserted.
inline constexpr unsigned long long contact_hash_empty = 0;

/// Open-addressed unique-contact table size for a bounded edge list.
/// Four slots per reserved edge, at least eight, rounded up to a power of two.
/// This is O(edge capacity), not O(grains²).
inline std::size_t contact_hash_slots(std::size_t edge_capacity) {
  constexpr std::size_t kMin = 8;
  if (edge_capacity > std::numeric_limits<std::size_t>::max() / 4) {
    throw std::overflow_error("contact hash table size overflow");
  }
  const std::size_t want = std::max(edge_capacity * 4, kMin);
  std::size_t slots = kMin;
  while (slots < want) {
    if (slots > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error("contact hash table size overflow");
    }
    slots *= 2;
  }
  return slots;
}

inline std::size_t contact_hash_bytes(std::size_t edge_capacity) {
  const auto slots = contact_hash_slots(edge_capacity);
  if (slots > std::numeric_limits<std::size_t>::max() / sizeof(unsigned long long)) {
    throw std::overflow_error("contact hash table byte count overflow");
  }
  return slots * sizeof(unsigned long long);
}

/// Dense unsigned G×G matrix the GPU path used to allocate.
inline std::size_t dense_adjacency_bytes(std::size_t grains) {
  if (grains && grains > std::numeric_limits<std::size_t>::max() / grains) {
    throw std::overflow_error("dense adjacency matrix size overflow");
  }
  if (grains * grains > std::numeric_limits<std::size_t>::max() / sizeof(unsigned)) {
    throw std::overflow_error("dense adjacency matrix byte count overflow");
  }
  return grains * grains * sizeof(unsigned);
}

inline unsigned long long pack_contact_key(std::size_t low, std::size_t high) {
  if (low >= high || high > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("contact hash key requires low < high in uint32");
  }
  return (static_cast<unsigned long long>(low) << 32) |
         static_cast<unsigned long long>(high);
}

inline std::pair<std::size_t, std::size_t>
unpack_contact_key(unsigned long long key) {
  return {static_cast<std::size_t>(key >> 32),
          static_cast<std::size_t>(key & 0xffffffffull)};
}

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
/// Insert ordered registry indices `(low, high)` into a power-of-two table.
/// Returns false only when every slot is occupied by a different key.
__device__ inline bool insert_contact_key(unsigned long long *table,
                                          std::size_t slots, std::size_t low,
                                          std::size_t high) {
  if (low > high) {
    auto t = low;
    low = high;
    high = t;
  }
  if (low == high || slots == 0) return true;
  const unsigned long long key = (static_cast<unsigned long long>(low) << 32) |
                                 static_cast<unsigned long long>(high);
  const std::size_t mask = slots - 1;
  std::size_t slot = static_cast<std::size_t>(key * 0x9e3779b97f4a7c15ull) & mask;
  for (std::size_t n = 0; n < slots; ++n) {
    const unsigned long long cur = table[slot];
    if (cur == key) return true;
    if (cur == contact_hash_empty) {
      const unsigned long long old =
          ::atomicCAS(table + slot, contact_hash_empty, key);
      if (old == contact_hash_empty || old == key) return true;
    }
    slot = (slot + 1) & mask;
  }
  return false;
}
#endif

} // namespace pfc::grain
