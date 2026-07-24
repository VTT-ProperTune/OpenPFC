// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef PFC_KERNEL_DATA_GRID_FIELD_HPP
#define PFC_KERNEL_DATA_GRID_FIELD_HPP

/**
 * @file grid_field.hpp
 * @brief M2 canonical owning field container: `pfc::data::Field<T, MemorySpace>`.
 *
 * The single owning grid-data container for OpenPFC 0.2 (Audit §13.3). It
 * unifies the legacy zoo:
 *   - `field::LocalField<T>`  == this with **halo width 0** (unpadded);
 *   - `field::PaddedBrick<T>` == this with **halo width n** (n-cell halo);
 *   - `field::Field<T>`       == this over the whole domain (single rank).
 *
 * Storage is the surviving `DataBuffer<MemorySpace, T>` primitive; layout is an
 * owned `Box3i` (the local owned index box, in *global* index coordinates) plus
 * a halo width and a **geometry POD held by value** (a `Domain`, giving spacing
 * and origin). There are deliberately NO `const World&` / `const Domain&`
 * members -- that reference-to-external-geometry pattern is the dangling hazard
 * Audit defect #10/§82 calls out. One row-major (x-fastest) linearization
 * `idx()` serves the padded and unpadded cases; `apply(f(x,y,z))` is defined
 * once.
 *
 * ADDITIVE M2.1: introduced alongside the legacy types so the build stays
 * green with no consumer changes. It lives in `pfc::data` for now because the
 * final name `pfc::Field` is still occupied by a Gen-1 `std::vector<double>`
 * alias (`kernel/data/model_types.hpp`) and `field::Field`. At M2.final, once
 * the legacy container zoo (and that alias) is deleted, this collapses into
 * `pfc::Field` in `kernel/data/field.hpp`.
 *
 * Residency tracking (M2.2): a device-backed field (device `MemorySpace`) also
 * owns a host mirror and a `Residency` (residency.hpp) recording which side is
 * current. `with_host_view(fn)` brackets host access (pulling device->host when
 * stale), `sync_to_device()` pushes host->device before a device kernel, and
 * `note_device_write()` records a device-side write. For a host-space field the
 * buffer is the host data and these collapse to no-ops. This is the framework
 * residency protocol that replaces the per-app `m_cpu_buffer_valid` +
 * `sync_*` hacks (Audit §4.1).
 */

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/residency.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>

namespace pfc::data {

/**
 * @brief The one owning field container: `DataBuffer` + `Box3i` + halo +
 *        geometry-by-value.
 *
 * @tparam T           Element type (e.g. `double`, `std::complex<double>`).
 * @tparam MemorySpace Placement tag (`HostSpace` default; `CudaSpace`/`HipSpace`
 *                     from the runtime headers). Selects the `DataBuffer`
 *                     backend via `memory_space_to_backend_t`.
 *
 * Logical indices `(i, j, k)` are **local** (0-based on the owned box). Halo
 * cells are addressable in `[-halo, n + halo)` on every axis, exactly as the
 * legacy `PaddedBrick`.
 */
template <typename T, typename MemorySpace = pfc::HostSpace> class Field {
public:
  using value_type = T;
  using memory_space = MemorySpace;
  using backend_tag = pfc::memory_space_to_backend_t<MemorySpace>;
  using storage_type = pfc::core::DataBuffer<backend_tag, T>;

  /// True when this field's memory space is host-accessible (no device mirror).
  static constexpr bool is_host_space = std::is_same_v<MemorySpace, pfc::HostSpace>;

  Field() = default;

  /**
   * @brief Construct from the global `Domain`, the local owned index box, and
   *        a halo width. Allocates `prod(size + 2*halo)` zero-initialized cells.
   * @throws std::invalid_argument on a negative halo or an inconsistent box.
   */
  Field(const pfc::Domain &domain, const pfc::Box3i &owned_box, int halo_width = 0)
      : m_domain(domain), m_box(owned_box), m_halo(halo_width),
        m_buffer(padded_volume_(owned_box, halo_width)) {
    if constexpr (!is_host_space) {
      // Device-backed field: seed on the host mirror (ICs land there), push to
      // the device on first device use. See residency.hpp / Audit §4.1.
      m_residency = Residency::device_backed();
      m_host_mirror.assign(m_buffer.size(), T{});
    }
  }

  // ---- storage ----------------------------------------------------------
  T *data() noexcept { return m_buffer.data(); }
  const T *data() const noexcept { return m_buffer.data(); }
  /// Total allocated cells, including halo padding.
  std::size_t size() const noexcept { return m_buffer.size(); }
  storage_type &buffer() noexcept { return m_buffer; }
  const storage_type &buffer() const noexcept { return m_buffer; }

  // ---- geometry ---------------------------------------------------------
  const pfc::Domain &domain() const noexcept { return m_domain; }
  /// Owned (interior) index box, in global index coordinates.
  const pfc::Box3i &box() const noexcept { return m_box; }
  int halo_width() const noexcept { return m_halo; }
  /// Per-axis count of owned cells (halo excluded).
  pfc::Int3 local_size() const noexcept { return m_box.size; }
  const pfc::Real3 &spacing() const noexcept {
    return pfc::domain::get_spacing(m_domain);
  }
  const pfc::Real3 &origin() const noexcept {
    return pfc::domain::get_origin(m_domain);
  }
  /// Per-axis padded extent (owned + both halo slabs).
  int padded_extent(int axis) const noexcept {
    return m_box.size[axis] + 2 * m_halo;
  }

  // ---- indexing (x-fastest row-major; matches LocalField/PaddedBrick) ----
  std::size_t idx(int i, int j, int k) const noexcept {
    const auto npx = static_cast<std::size_t>(padded_extent(0));
    const auto npy = static_cast<std::size_t>(padded_extent(1));
    const auto hw = static_cast<std::size_t>(m_halo);
    return (static_cast<std::size_t>(i) + hw) +
           (static_cast<std::size_t>(j) + hw) * npx +
           (static_cast<std::size_t>(k) + hw) * npx * npy;
  }
  std::size_t idx(const pfc::Int3 &c) const noexcept {
    return idx(c[0], c[1], c[2]);
  }

  T &operator()(int i, int j, int k) noexcept { return m_buffer[idx(i, j, k)]; }
  const T &operator()(int i, int j, int k) const noexcept {
    return m_buffer[idx(i, j, k)];
  }
  T &operator()(const pfc::Int3 &c) noexcept { return m_buffer[idx(c)]; }
  const T &operator()(const pfc::Int3 &c) const noexcept { return m_buffer[idx(c)]; }

  // ---- coordinate queries (match LocalField exactly) --------------------
  /// Global index `(gi, gj, gk)` of local logical `(i, j, k)`.
  pfc::Int3 global(int i, int j, int k) const noexcept {
    return {m_box.low[0] + i, m_box.low[1] + j, m_box.low[2] + k};
  }
  /// Physical coordinates of local logical `(i, j, k)`.
  pfc::Real3 coords(int i, int j, int k) const noexcept {
    const auto &o = origin();
    const auto &s = spacing();
    const auto &lo = m_box.low;
    return {o[0] + static_cast<double>(lo[0] + i) * s[0],
            o[1] + static_cast<double>(lo[1] + j) * s[1],
            o[2] + static_cast<double>(lo[2] + k) * s[2]};
  }

  // ---- iteration --------------------------------------------------------
  /// Visit every owned cell `(i, j, k)` in x-fastest order.
  template <typename Fn> void for_each_owned(Fn &&fn) const {
    const int nx = m_box.size[0];
    const int ny = m_box.size[1];
    const int nz = m_box.size[2];
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) fn(i, j, k);
  }

  /// Fill every owned cell by sampling `fn(x, y, z)` at its physical coords.
  /// A host-side write, so the device copy (if any) is marked stale.
  template <typename Fn> void apply(Fn &&fn) {
    for_each_owned([&](int i, int j, int k) {
      const auto c = coords(i, j, k);
      (*this)(i, j, k) = fn(c[0], c[1], c[2]);
    });
    m_residency.note_host_write();
  }

  // ---- residency (M2.2) -------------------------------------------------
  /// Host/device coherence state (see residency.hpp).
  const Residency &residency() const noexcept { return m_residency; }

  /// Record that a device kernel wrote the device buffer (host mirror stale).
  void note_device_write() noexcept { m_residency.note_device_write(); }

  /**
   * @brief Push the host mirror to the device buffer when the device copy is
   *        stale. No-op for a host-space field. Call before a device kernel
   *        reads this field -- the sync the audit-4.1 bug omitted.
   */
  void sync_to_device() {
    if constexpr (!is_host_space) {
      if (m_residency.device_needs_refresh()) {
        m_buffer.copy_from_host(m_host_mirror.data(), m_host_mirror.size());
        m_residency.note_synced();
      }
    }
  }

  /**
   * @brief Bracket a host-side access. Ensures the host data is current
   *        (pulling device->host for a device field), invokes
   *        `fn(T* data, std::size_t size)` over the padded host buffer, then
   *        marks the host side authoritative (device copy stale).
   */
  template <typename Fn> void with_host_view(Fn &&fn) {
    if constexpr (is_host_space) {
      fn(m_buffer.data(), m_buffer.size());
    } else {
      if (m_residency.host_needs_refresh()) {
        m_buffer.copy_to_host(m_host_mirror.data(), m_host_mirror.size());
        m_residency.note_synced();
      }
      fn(m_host_mirror.data(), m_host_mirror.size());
    }
    m_residency.note_host_write();
  }

private:
  static std::size_t padded_volume_(const pfc::Box3i &box, int halo) {
    if (halo < 0) throw std::invalid_argument("Field: halo width must be >= 0");
    if (!box.is_consistent())
      throw std::invalid_argument("Field: owned box is not consistent");
    std::size_t v = 1;
    for (int d = 0; d < 3; ++d)
      v *= static_cast<std::size_t>(box.size[d] + 2 * halo);
    return v;
  }

  pfc::Domain m_domain{};
  pfc::Box3i m_box{};
  int m_halo{0};
  storage_type m_buffer{};
  Residency m_residency{Residency::host_only()};
  // Host mirror for a device-backed field; stays empty for a host-space field.
  std::vector<T> m_host_mirror{};
};

} // namespace pfc::data

#endif // PFC_KERNEL_DATA_GRID_FIELD_HPP
