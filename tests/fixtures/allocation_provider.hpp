// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
// Consumer-owned observer for ordinary/aligned/sized global new/delete. The
// stable public token records requested object bytes, not allocator headers,
// padding, runtime reserves, RSS or memory-bus transactions.
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <new>
#include <openpfc/kernel/diagnostics/memory_traffic.hpp>
namespace allocation_observer {
struct Header {
  void *raw;
  pfc::diagnostics::AllocationToken token;
};
void *allocate(std::size_t bytes, std::size_t alignment) {
  alignment = std::max(alignment, alignof(std::max_align_t));
  if (bytes > std::numeric_limits<std::size_t>::max() - sizeof(Header) - alignment)
    throw std::bad_alloc();
  void *raw =
      std::malloc(std::max<std::size_t>(bytes, 1) + sizeof(Header) + alignment - 1);
  if (!raw) throw std::bad_alloc();
  auto address =
      (reinterpret_cast<std::uintptr_t>(raw) + sizeof(Header) + alignment - 1) &
      ~(alignment - 1);
  auto *header = reinterpret_cast<Header *>(address - sizeof(Header));
  new (header) Header{raw, pfc::diagnostics::successful_allocation(
                               pfc::diagnostics::Space::Host, bytes)};
  return reinterpret_cast<void *>(address);
}
void release(void *pointer) noexcept {
  if (!pointer) return;
  auto *header = reinterpret_cast<Header *>(
      reinterpret_cast<std::uintptr_t>(pointer) - sizeof(Header));
  void *raw = header->raw;
  auto token = std::move(header->token);
  header->~Header();
  std::free(raw);
  token.release(); // Count release after the actual successful deallocation.
}
} // namespace allocation_observer
void *operator new(std::size_t bytes) {
  return allocation_observer::allocate(bytes, alignof(std::max_align_t));
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void *operator new(std::size_t bytes, std::align_val_t alignment) {
  return allocation_observer::allocate(bytes, static_cast<std::size_t>(alignment));
}
void *operator new[](std::size_t bytes, std::align_val_t alignment) {
  return ::operator new(bytes, alignment);
}
void *operator new(std::size_t bytes, const std::nothrow_t &) noexcept {
  try {
    return ::operator new(bytes);
  } catch (...) {
    return nullptr;
  }
}
void *operator new[](std::size_t bytes, const std::nothrow_t &tag) noexcept {
  return ::operator new(bytes, tag);
}
void *operator new(std::size_t bytes, std::align_val_t alignment,
                   const std::nothrow_t &) noexcept {
  try {
    return ::operator new(bytes, alignment);
  } catch (...) {
    return nullptr;
  }
}
void *operator new[](std::size_t bytes, std::align_val_t alignment,
                     const std::nothrow_t &tag) noexcept {
  return ::operator new(bytes, alignment, tag);
}
void operator delete(void *p) noexcept { allocation_observer::release(p); }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete(void *p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, std::size_t) noexcept { ::operator delete(p); }
void operator delete(void *p, std::align_val_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, std::align_val_t) noexcept { ::operator delete(p); }
void operator delete(void *p, std::size_t, std::align_val_t) noexcept {
  ::operator delete(p);
}
void operator delete[](void *p, std::size_t, std::align_val_t) noexcept {
  ::operator delete(p);
}
void operator delete(void *p, const std::nothrow_t &) noexcept {
  ::operator delete(p);
}
void operator delete[](void *p, const std::nothrow_t &) noexcept {
  ::operator delete(p);
}
void operator delete(void *p, std::align_val_t, const std::nothrow_t &) noexcept {
  ::operator delete(p);
}
void operator delete[](void *p, std::align_val_t, const std::nothrow_t &) noexcept {
  ::operator delete(p);
}
