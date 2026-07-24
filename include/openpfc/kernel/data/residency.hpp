// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef PFC_KERNEL_DATA_RESIDENCY_HPP
#define PFC_KERNEL_DATA_RESIDENCY_HPP

/**
 * @file residency.hpp
 * @brief Host/device coherence bookkeeping for the canonical `Field` (M2.2).
 *
 * Pure host-side state: which side of a two-sided (device-backed) field holds
 * the authoritative data, and therefore whether a host<->device transfer is
 * needed before the next access. It performs NO device calls, so the whole
 * transition machine is unit-testable without a GPU -- which matters because
 * OpenPFC CI compiles GPU code but never runs it (Audit §4.2 / the "not tested
 * at runtime" TODOs). `Field<T, MemorySpace>` owns one `Residency` and consults
 * it to decide when to actually invoke `DataBuffer::copy_{from,to}_host`.
 *
 * This is the framework-level replacement for the per-app, per-vendor
 * hand-rolled `m_cpu_buffer_valid` + `sync_cpu_to_gpu`/`sync_gpu_to_cpu` state
 * machines (Audit §4.1) whose omission let the App-driven GPU tungsten binary
 * integrate an unseeded device field.
 *
 * Coherence rule: a write on one side makes that side authoritative and marks
 * the other side stale; a read on a stale side must refresh (transfer) first;
 * a completed transfer makes both sides agree.
 */

namespace pfc::data {

/**
 * @brief Tracks host/device validity for a field's storage.
 *
 * A **one-sided** field (host-only: `MemorySpace == HostSpace`) is permanently
 * host-coherent and never needs a transfer. A **two-sided** field (device
 * memory space, carrying a host mirror) tracks which side is current.
 */
class Residency {
public:
  /// One-sided host field: permanently host-coherent, no device side.
  static constexpr Residency host_only() noexcept {
    return Residency(/*host=*/true, /*device=*/true, /*two_sided=*/false);
  }

  /**
   * @brief Two-sided device-backed field, seeded on the host mirror first.
   *
   * Initial conditions are applied on the host, so the host side starts
   * authoritative and the device copy is stale -- a device-side use must push
   * host->device first. This is exactly the sync the audit-4.1 bug omitted.
   */
  static constexpr Residency device_backed() noexcept {
    return Residency(/*host=*/true, /*device=*/false, /*two_sided=*/true);
  }

  constexpr bool two_sided() const noexcept { return m_two_sided; }
  constexpr bool host_valid() const noexcept { return m_host_valid; }
  constexpr bool device_valid() const noexcept { return m_device_valid; }

  /// True iff a host read must pull device->host before it is safe.
  constexpr bool host_needs_refresh() const noexcept {
    return m_two_sided && !m_host_valid;
  }
  /// True iff a device use must push host->device before it is safe.
  constexpr bool device_needs_refresh() const noexcept {
    return m_two_sided && !m_device_valid;
  }

  /// Record a host-side write: host becomes authoritative, device goes stale.
  constexpr void note_host_write() noexcept {
    m_host_valid = true;
    if (m_two_sided) m_device_valid = false;
  }
  /// Record a device-side write: device becomes authoritative, host goes stale.
  constexpr void note_device_write() noexcept {
    m_device_valid = true;
    if (m_two_sided) m_host_valid = false;
  }
  /// Record a completed transfer: both sides now agree.
  constexpr void note_synced() noexcept {
    m_host_valid = true;
    m_device_valid = true;
  }

private:
  constexpr Residency(bool host_valid, bool device_valid, bool two_sided) noexcept
      : m_host_valid(host_valid), m_device_valid(device_valid),
        m_two_sided(two_sided) {}

  bool m_host_valid;
  bool m_device_valid;
  bool m_two_sided;
};

} // namespace pfc::data

#endif // PFC_KERNEL_DATA_RESIDENCY_HPP
