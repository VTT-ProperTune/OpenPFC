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
 * `pfc::data::Field` is the definition; `pfc::Field<T, MemorySpace>` (alias at
 * the end of this header) is the public name. The Gen-1 `std::vector<double>`
 * alias that used to occupy `pfc::Field` is deleted; conditions, writers, and
 * checkpoints take the non-owning `field::FieldOutput<T>` / `field::FieldView<T>`
 * (`view()` / `output()` below) instead of bare vectors.
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
#include <openpfc/kernel/field/state_access.hpp>

namespace pfc::data {

/**
 * @brief The one owning field container: `DataBuffer` + `Box3i` + halo +
 *        geometry-by-value.
 *
 * @tparam T           Element type (e.g. `double`, `std::complex<double>`).
 * @tparam MemorySpace Placement tag (`HostSpace` default; `CUDASpace`/`HIPSpace`
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
   *
   * `halo_width` is both the **storage** padding and the default **iteration**
   * halo (PaddedBrick convention). Prefer the four-argument overload when
   * migrating unpadded `LocalField` face-halo layouts (storage 0, iteration n).
   */
  Field(const pfc::Domain &domain, const pfc::Box3i &owned_box, int halo_width = 0)
      : Field(domain, owned_box, halo_width, halo_width) {}

  /**
   * @brief Construct with independent storage padding and iteration halo.
   *
   * - `storage_halo` sizes the buffer (`prod(size + 2*storage_halo)`).
   * - `iteration_halo` is what `halo_width()` / `for_each_interior` report —
   *   matching `LocalField`'s metadata halo on an unpadded buffer.
   *
   * Face-halo FD stacks use `storage_halo=0` and `iteration_halo=order/2`.
   * PaddedBrick-style fields use equal values for both.
   */
  Field(const pfc::Domain &domain, const pfc::Box3i &owned_box, int storage_halo,
        int iteration_halo)
      : m_domain(domain), m_box(owned_box), m_halo(storage_halo),
        m_iteration_halo(iteration_halo),
        m_buffer(padded_volume_(owned_box, storage_halo)) {
    if (iteration_halo < 0) {
      throw std::invalid_argument("Field: iteration halo width must be >= 0");
    }
    if constexpr (!is_host_space) {
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

  /**
   * @brief Host `std::vector` view of the buffer (FFT / legacy APIs).
   * @note HostSpace only — device-backed fields have no host `std::vector`
   *       primary storage.
   */
  std::vector<T> &vec() {
    static_assert(is_host_space,
                  "Field::vec() is only available for HostSpace fields");
    return m_buffer.as_vector();
  }
  const std::vector<T> &vec() const {
    static_assert(is_host_space,
                  "Field::vec() is only available for HostSpace fields");
    return m_buffer.as_vector();
  }

  // ---- geometry ---------------------------------------------------------
  /**
   * @brief Read-only non-owning view (host space only): data + local extents +
   *        spacing + origin of the local box's low corner.
   */
  [[nodiscard]] pfc::field::FieldView<T> view() const noexcept
    requires is_host_space
  {
    return pfc::field::FieldView<T>(m_buffer.data(), m_buffer.size(), padded_extents_(),
                                    spacing(), local_origin_());
  }

  /** @brief Mutable non-owning view over the storage (host space only). */
  [[nodiscard]] pfc::field::FieldOutput<T> output() noexcept
    requires is_host_space
  {
    return pfc::field::FieldOutput<T>(m_buffer.data(), m_buffer.size());
  }

  const pfc::Domain &domain() const noexcept { return m_domain; }
  /// Owned (interior) index box, in global index coordinates.
  const pfc::Box3i &box() const noexcept { return m_box; }
  /// Storage padding width (cells added on each side of the owned box).
  int storage_halo() const noexcept { return m_halo; }
  /**
   * @brief Iteration / stencil halo (LocalField-compatible).
   *
   * For padded fields this equals `storage_halo()`. For unpadded face-halo
   * layouts it is the metadata width used by `for_each_interior` and
   * `FDGradient` factories, while storage stays tightly packed.
   */
  int halo_width() const noexcept { return m_iteration_halo; }
  /// Per-axis count of owned cells (halo excluded).
  pfc::Int3 local_size() const noexcept { return m_box.size; }
  /// LocalField-compatible alias of `local_size()`.
  pfc::Int3 size3() const noexcept { return m_box.size; }
  /// Global domain extents `{Nx, Ny, Nz}` (LocalField-compatible).
  pfc::Int3 global_size() const noexcept {
    return pfc::domain::get_size(m_domain);
  }
  /// Global index of local `(0,0,0)` (LocalField-compatible).
  pfc::Int3 lower_global() const noexcept { return m_box.low; }
  const pfc::Real3 &spacing() const noexcept {
    return pfc::domain::get_spacing(m_domain);
  }
  const pfc::Real3 &origin() const noexcept {
    return pfc::domain::get_origin(m_domain);
  }
  /// Per-axis padded extent (owned + both storage-halo slabs).
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
  /**
   * @brief Visit every owned cell in x-fastest order.
   *
   * Callable may be any of (auto-detected):
   *  - `void(int i, int j, int k)`
   *  - `void(double x, double y, double z, T value)`  (LocalField-compatible)
   *  - `void(const Real3& x, T value)`                (LocalField-compatible)
   */
  template <typename Fn> void for_each_owned(Fn &&fn) const {
    const int nx = m_box.size[0];
    const int ny = m_box.size[1];
    const int nz = m_box.size[2];
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          // Prefer LocalField-compatible coord signatures first: int promotes
          // to double, so checking (int,int,int) first would mis-dispatch.
          if constexpr (std::is_invocable_v<Fn &, double, double, double, T>) {
            const auto c = coords(i, j, k);
            fn(c[0], c[1], c[2], (*this)(i, j, k));
          } else if constexpr (std::is_invocable_v<Fn &, const pfc::Real3 &,
                                                   T>) {
            fn(coords(i, j, k), (*this)(i, j, k));
          } else {
            fn(i, j, k);
          }
        }
      }
    }
  }

  /**
   * @brief Iterate interior `[hw, n-hw)` per axis (LocalField-compatible).
   *
   * Callable may be either of:
   *  - `void(double x, double y, double z, T value)`
   *  - `void(const Real3& x, T value)`
   */
  template <typename Fn> void for_each_interior(Fn &&fn) const {
    const int hw = m_iteration_halo;
    const int nx = m_box.size[0];
    const int ny = m_box.size[1];
    const int nz = m_box.size[2];
    const int imin = hw, imax = nx - hw;
    const int jmin = hw, jmax = ny - hw;
    const int kmin = hw, kmax = nz - hw;
    if (imin >= imax || jmin >= jmax || kmin >= kmax) return;
    for (int k = kmin; k < kmax; ++k) {
      for (int j = jmin; j < jmax; ++j) {
        for (int i = imin; i < imax; ++i) {
          const auto c = coords(i, j, k);
          if constexpr (std::is_invocable_v<Fn &, double, double, double, T>) {
            fn(c[0], c[1], c[2], (*this)(i, j, k));
          } else {
            fn(c, (*this)(i, j, k));
          }
        }
      }
    }
  }

  /// Fill every owned cell by sampling `fn` at its physical coords.
  /// Accepts `T(double,double,double)` or `T(const Real3&)`.
  /// A host-side write, so the device copy (if any) is marked stale.
  template <typename Fn> void apply(Fn &&fn) {
    const int nx = m_box.size[0];
    const int ny = m_box.size[1];
    const int nz = m_box.size[2];
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const auto c = coords(i, j, k);
          if constexpr (std::is_invocable_v<Fn &, double, double, double>) {
            (*this)(i, j, k) = fn(c[0], c[1], c[2]);
          } else {
            (*this)(i, j, k) = fn(c);
          }
        }
      }
    }
    m_residency.note_host_write();
  }

  // ---- residency (M2.2) -------------------------------------------------
  /// Host/device coherence state (see residency.hpp).
  const Residency &residency() const noexcept { return m_residency; }

  /// Record that a device kernel wrote the device buffer (host mirror stale).
  void note_device_write() noexcept { m_residency.note_device_write(); }

  /// Record a host-side write (device mirror stale). Public for axpy helpers.
  void note_host_write() noexcept { m_residency.note_host_write(); }

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
  pfc::Int3 padded_extents_() const noexcept {
    return {m_box.size[0] + 2 * m_halo, m_box.size[1] + 2 * m_halo,
            m_box.size[2] + 2 * m_halo};
  }
  pfc::Real3 local_origin_() const noexcept {
    const auto &o = origin();
    const auto &s = spacing();
    return {o[0] + static_cast<double>(m_box.low[0] - m_halo) * s[0],
            o[1] + static_cast<double>(m_box.low[1] - m_halo) * s[1],
            o[2] + static_cast<double>(m_box.low[2] - m_halo) * s[2]};
  }

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
  int m_halo{0};             ///< storage padding
  int m_iteration_halo{0};   ///< LocalField-compatible stencil / interior halo
  storage_type m_buffer{};
  Residency m_residency{Residency::host_only()};
  // Host mirror for a device-backed field; stays empty for a host-space field.
  std::vector<T> m_host_mirror{};
};

} // namespace pfc::data

namespace pfc {
/// Public name of the canonical owning field (`pfc::data::Field` is the definition).
template <class T, class MemorySpace = pfc::HostSpace>
using Field = data::Field<T, MemorySpace>;
} // namespace pfc

#endif // PFC_KERNEL_DATA_GRID_FIELD_HPP
