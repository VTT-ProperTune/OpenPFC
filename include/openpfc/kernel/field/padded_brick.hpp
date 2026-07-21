// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file padded_brick.hpp
 * @brief A halo-padded brick for laboratory-style explicit stencil loops.
 *
 * @details
 * `pfc::field::PaddedBrick<T>` is the layout the **laboratory-style** FD
 * driver wants: a single contiguous buffer sized
 * `(nx + 2*hw) * (ny + 2*hw) * (nz + 2*hw)` so the user can write
 *
 *     u(i+1, j, k) - 2 * u(i, j, k) + u(i-1, j, k)
 *
 * for owned cell `i in [0, nx)` and have `u(-1, j, k)` and `u(nx, j, k)`
 * legitimately reach into the **left/right halo ring** that the in-place
 * `pfc::communication::PaddedHaloExchanger<T>` writes into.
 *
 * Compared to its older siblings:
 *
 * - `pfc::field::LocalField<T>` carries no halo storage at all (size is
 *   exactly `nx*ny*nz`); the FD path keeps its halos in six separate
 *   face vectors and exchanges them with `pfc::SparseHaloExchanger`
 *   (typically built via `pfc::halo::make_structured_halos`).
 * - `pfc::HaloExchanger<T>` is the existing in-place exchanger but its
 *   "no extra padding" face-type spec (see
 *   `kernel/decomposition/halo_mpi_types.hpp` line 105) **overwrites the
 *   outermost owned cells with neighbor data** — there is no ghost ring,
 *   so `u(-1, ...)` is meaningless.
 *
 * `PaddedBrick<T>` therefore stores both the **owned core**
 * `[0, n) x [0, n) x [0, n)` and a **halo ring** of width `hw` on each
 * side of every axis in **one** contiguous buffer, and exposes
 * `T &operator()(int i, int j, int k)` valid for any
 * `i,j,k in [-hw, n+hw)`.
 *
 * The internal storage is row-major with **x fastest**, matching the rest
 * of OpenPFC (`kernel/field/fd_gradient.hpp`, `kernel/field/fd_apply.hpp`,
 * `kernel/decomposition/halo_mpi_types.hpp`, FFT inbox iteration, ...).
 *
 * @see kernel/decomposition/padded_halo_mpi_types.hpp for the matching
 *      MPI subarray helper that handles the padded layout.
 * @see kernel/decomposition/padded_halo_exchange.hpp for the in-place
 *      non-blocking halo exchanger built around it.
 * @see kernel/field/brick_iteration.hpp for `for_each_owned/inner/border`
 *      helpers that yield `(i, j, k)` triples over a `PaddedBrick`.
 */

#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/data/world_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/scaled_field.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pfc::field {

namespace detail {

/**
 * @brief Compute padded extent `n + 2*hw` with overflow check.
 *
 * @param n Base extent.
 * @param hw Halo width (non-negative).
 * @return Padded extent as a signed 64-bit integer.
 * @throws std::overflow_error if `n + 2*hw` would overflow `int`.
 * @throws std::invalid_argument if `hw < 0`.
 */
inline long long checked_padded_extent(int n, int hw) {
  if (hw < 0) {
    throw std::invalid_argument("padded extent: halo width must be non-negative (got " +
                                std::to_string(hw) + ")");
  }
  // Use long long to detect overflow
  const long long result = static_cast<long long>(n) + 2LL * static_cast<long long>(hw);
  if (result > static_cast<long long>(std::numeric_limits<int>::max()) ||
      result < static_cast<long long>(std::numeric_limits<int>::min())) {
    throw std::overflow_error("padded extent overflow: " + std::to_string(n) +
                              " + 2*" + std::to_string(hw) + " exceeds int range");
  }
  return result;
}

/**
 * @brief Compute 3D product with overflow check.
 *
 * @param nx First dimension.
 * @param ny Second dimension.
 * @param nz Third dimension.
 * @return Product as std::size_t.
 * @throws std::overflow_error if product would overflow std::size_t.
 * @throws std::invalid_argument if any dimension is negative.
 */
inline std::size_t checked_product_3d(long long nx, long long ny, long long nz) {
  if (nx < 0 || ny < 0 || nz < 0) {
    throw std::invalid_argument("product overflow: dimensions must be non-negative (" +
                                std::to_string(nx) + ", " + std::to_string(ny) +
                                ", " + std::to_string(nz) + ")");
  }
  // Stepwise unsigned multiply (same pattern as vtk_writer_validate) — avoids
  // signed long long overflow UB when intermediate products exceed LLONG_MAX.
  unsigned long long n = 1;
  const auto max_sz =
      static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)());
  for (const unsigned long long dim :
       {static_cast<unsigned long long>(nx), static_cast<unsigned long long>(ny),
        static_cast<unsigned long long>(nz)}) {
    if (dim != 0ULL && n > max_sz / dim) {
      throw std::overflow_error("3D product overflow: " + std::to_string(nx) + " * " +
                                std::to_string(ny) + " * " + std::to_string(nz) +
                                " exceeds std::size_t range");
    }
    n *= dim;
  }
  return static_cast<std::size_t>(n);
}

/**
 * @brief Forward iterator yielding `pfc::Int3{i, j, k}` over a half-open
 *        cuboid `[lo[0], hi[0]) x [lo[1], hi[1]) x [lo[2], hi[2])` in
 *        **k-outer / j-middle / i-inner** (row-major, x-fastest) order.
 *
 * Used by `PaddedBrick<T>::indices()` and `indices_inner(r)` to drive
 * `pfc::field::for_each(...)` over the owned region in the same order
 * the storage is laid out, so the inner loop is cache-friendly.
 */
class OwnedIndexIterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = pfc::Int3;
  using reference = pfc::Int3;
  using pointer = void;
  using difference_type = std::ptrdiff_t;

  OwnedIndexIterator() noexcept = default;
  OwnedIndexIterator(pfc::Int3 lo, pfc::Int3 hi, pfc::Int3 cur) noexcept
      : m_lo(lo), m_hi(hi), m_cur(cur) {}

  pfc::Int3 operator*() const noexcept { return m_cur; }

  OwnedIndexIterator &operator++() noexcept {
    if (++m_cur[0] >= m_hi[0]) {
      m_cur[0] = m_lo[0];
      if (++m_cur[1] >= m_hi[1]) {
        m_cur[1] = m_lo[1];
        ++m_cur[2];
      }
    }
    return *this;
  }
  OwnedIndexIterator operator++(int) noexcept {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  friend bool operator==(const OwnedIndexIterator &a,
                         const OwnedIndexIterator &b) noexcept {
    return a.m_cur == b.m_cur;
  }
  friend bool operator!=(const OwnedIndexIterator &a,
                         const OwnedIndexIterator &b) noexcept {
    return !(a == b);
  }

private:
  pfc::Int3 m_lo{};
  pfc::Int3 m_hi{};
  pfc::Int3 m_cur{};
};

/**
 * @brief Half-open cuboid `[lo, hi)` exposed as a forward range whose
 *        elements are `pfc::Int3{i, j, k}` triples in k-outer / j-middle
 *        / i-inner order. Empty whenever any axis has `lo[d] >= hi[d]`.
 */
class OwnedIndexRange {
public:
  OwnedIndexRange() noexcept = default;
  OwnedIndexRange(pfc::Int3 lo, pfc::Int3 hi) noexcept : m_lo(lo), m_hi(hi) {
    if (m_lo[0] >= m_hi[0] || m_lo[1] >= m_hi[1] || m_lo[2] >= m_hi[2]) {
      m_hi = m_lo; // collapse to empty range
    }
  }

  OwnedIndexIterator begin() const noexcept { return {m_lo, m_hi, m_lo}; }
  OwnedIndexIterator end() const noexcept {
    // After incrementing past `(hi[0]-1, hi[1]-1, hi[2]-1)` the iterator
    // reaches `(lo[0], lo[1], hi[2])` — the canonical sentinel.
    return {m_lo, m_hi, pfc::Int3{m_lo[0], m_lo[1], m_hi[2]}};
  }

  [[nodiscard]] bool empty() const noexcept {
    return m_lo[0] == m_hi[0] || m_lo[1] == m_hi[1] || m_lo[2] == m_hi[2];
  }

  [[nodiscard]] std::size_t size() const noexcept {
    if (empty()) return 0;
    return static_cast<std::size_t>(m_hi[0] - m_lo[0]) *
           static_cast<std::size_t>(m_hi[1] - m_lo[1]) *
           static_cast<std::size_t>(m_hi[2] - m_lo[2]);
  }

  [[nodiscard]] pfc::Int3 lower() const noexcept { return m_lo; }
  [[nodiscard]] pfc::Int3 upper() const noexcept { return m_hi; }

private:
  pfc::Int3 m_lo{};
  pfc::Int3 m_hi{};
};

} // namespace detail

/**
 * @brief Halo-padded contiguous brick: one buffer, `[-hw, n+hw)` indexing.
 *
 * Storage layout (row-major, x fastest):
 *   `linear = (i + hw) + (j + hw) * nx_pad + (k + hw) * nx_pad * ny_pad`
 * where `nx_pad = nx + 2*hw`, etc. The MPI subarray types built by
 * `pfc::halo::create_padded_face_types_6` use the same convention.
 *
 * Each brick is **self-contained**: it owns the `Decomposition` (by value),
 * its `rank`, and `halo_width` so downstream consumers — exchangers,
 * gradient evaluators, iteration helpers — can pick everything they need
 * from the brick alone. This avoids drift between e.g. an exchanger and
 * an evaluator constructed against the same field but with mismatched
 * halo widths.
 *
 * The brick itself is **MPI-unaware**: the `MPI_Comm` lives on the
 * exchanger (`pfc::PaddedHaloExchanger<T>`), not on the storage.
 *
 * @tparam T Element type (`double` for the heat equation; `float` /
 *           `std::complex<double>` are equally valid).
 */
template <class T> class PaddedBrick {
public:
  /**
   * @brief Construct a padded brick from an existing decomposition.
   *
   * Geometry comes from `decomposition::get_subworld(decomp, rank)` for
   * the owned size + lower global index, and from the **global world**
   * for the physical origin/spacing (so `global_coords(i, j, k)`
   * returns the right physical position even for halo cells `i = -1`,
   * which conceptually live at the rank's left neighbor).
   *
   * The returned buffer is value-initialized (`T{}`).
   *
   * @throws std::invalid_argument if `halo_width < 0`. Unlike
   *         `LocalField::from_subdomain` we do **not** require the
   *         owned region to exceed `2*halo_width` per axis — a
   *         padded brick with a tiny owned core is still valid; the
   *         user just doesn't get a non-empty inner region for that
   *         halo width.
   */
  PaddedBrick(const pfc::decomposition::Decomposition &decomp, int rank,
              int halo_width)
      : m_decomp(decomp), m_rank(rank), m_halo(halo_width) {
    if (halo_width < 0) {
      throw std::invalid_argument(
          "pfc::field::PaddedBrick: halo_width must be non-negative (got " +
          std::to_string(halo_width) + ")");
    }
    const auto &gw = pfc::decomposition::get_world(m_decomp);
    const auto &local = pfc::decomposition::get_subworld(m_decomp, m_rank);
    m_size = pfc::world::get_size(local);
    m_lower = pfc::world::get_lower(local);
    m_global_size = pfc::world::get_size(gw);
    m_origin = pfc::world::get_origin(gw);
    m_spacing = pfc::world::get_spacing(gw);

    // Check for overflow before computing padded extents
    const auto npx_ll = detail::checked_padded_extent(m_size[0], m_halo);
    const auto npy_ll = detail::checked_padded_extent(m_size[1], m_halo);
    const auto npz_ll = detail::checked_padded_extent(m_size[2], m_halo);

    // Check for overflow in the total element count
    const std::size_t total_elements = detail::checked_product_3d(npx_ll, npy_ll, npz_ll);

    m_data.assign(total_elements, T{});
  }

  // ---- Storage ------------------------------------------------------------

  /**
   * @brief Pointer to the **start of the padded buffer** (i.e. the
   *        `(-hw, -hw, -hw)` cell, **not** the `(0, 0, 0)` owned cell).
   *
   * This is the pointer that `pfc::PaddedHaloExchanger<T>` and the
   * matching MPI subarray types in
   * `kernel/decomposition/padded_halo_mpi_types.hpp` operate on.
   */
  T *data() noexcept { return m_data.data(); }
  const T *data() const noexcept { return m_data.data(); }

  /// Total number of elements in the padded buffer (`nx_pad*ny_pad*nz_pad`).
  std::size_t size() const noexcept { return m_data.size(); }

  /** Underlying vector — pass to allocators, exchangers, raw kernels. */
  std::vector<T> &vec() noexcept { return m_data; }
  const std::vector<T> &vec() const noexcept { return m_data; }

  // ---- Geometry queries ---------------------------------------------------

  /// Local owned size `(nx, ny, nz)` (excludes the halo ring).
  pfc::Int3 size3() const noexcept { return m_size; }
  /// Local **padded** size `(nx + 2hw, ny + 2hw, nz + 2hw)`.
  pfc::Int3 padded_size3() const noexcept {
    return pfc::Int3{padded_extent_(m_size[0]), padded_extent_(m_size[1]),
                     padded_extent_(m_size[2])};
  }
  pfc::Int3 lower_global() const noexcept { return m_lower; }
  pfc::Int3 global_size() const noexcept { return m_global_size; }
  pfc::Real3 origin() const noexcept { return m_origin; }
  pfc::Real3 spacing() const noexcept { return m_spacing; }
  int halo_width() const noexcept { return m_halo; }

  /// Decomposition that produced this brick (carried by value, MPI-free).
  const pfc::decomposition::Decomposition &decomposition() const noexcept {
    return m_decomp;
  }
  /// Rank in the parent decomposition the brick was built for.
  int rank() const noexcept { return m_rank; }

  /// Owned-x extent.
  int nx() const noexcept { return m_size[0]; }
  /// Owned-y extent.
  int ny() const noexcept { return m_size[1]; }
  /// Owned-z extent.
  int nz() const noexcept { return m_size[2]; }
  /// Padded-x extent (`nx + 2*hw`).
  int nx_padded() const noexcept { return padded_extent_(m_size[0]); }
  /// Padded-y extent (`ny + 2*hw`).
  int ny_padded() const noexcept { return padded_extent_(m_size[1]); }
  /// Padded-z extent (`nz + 2*hw`).
  int nz_padded() const noexcept { return padded_extent_(m_size[2]); }

  // ---- Indexing helpers ---------------------------------------------------

  /**
   * @brief Flat row-major index for local `(i, j, k)`. Valid for any
   *        `i,j,k in [-hw, n+hw)`. Halo cells map to the outer ring.
   *
   * @note No bounds checking in the release build. Out-of-range arguments
   *       are undefined behaviour; pair with `for_each_owned` / `_inner`
   *       / `_border` to guarantee the iteration stays in-range.
   */
  std::size_t idx(int i, int j, int k) const noexcept {
    const auto npx = static_cast<std::size_t>(padded_extent_(m_size[0]));
    const auto npy = static_cast<std::size_t>(padded_extent_(m_size[1]));
    const auto hw = static_cast<std::size_t>(m_halo);
    return (static_cast<std::size_t>(i) + hw) +
           (static_cast<std::size_t>(j) + hw) * npx +
           (static_cast<std::size_t>(k) + hw) * npx * npy;
  }

  /// Element access for any `i,j,k in [-hw, n+hw)`.
  T &operator()(int i, int j, int k) noexcept { return m_data[idx(i, j, k)]; }
  const T &operator()(int i, int j, int k) const noexcept {
    return m_data[idx(i, j, k)];
  }

  /**
   * @brief `pfc::Int3` overloads of `idx` / `operator()`.
   *
   * Lets generic code call `brick(idx)` after receiving a triple from
   * `for_each` or `indices()`, instead of unpacking it manually.
   */
  std::size_t idx(const pfc::Int3 &c) const noexcept {
    return idx(c[0], c[1], c[2]);
  }
  T &operator()(const pfc::Int3 &c) noexcept { return m_data[idx(c)]; }
  const T &operator()(const pfc::Int3 &c) const noexcept { return m_data[idx(c)]; }

  /// @brief Same as `operator()(c)`; allows `brick[idx]` with `idx` from `for_each`.
  T &operator[](const pfc::Int3 &c) noexcept { return m_data[idx(c)]; }
  const T &operator[](const pfc::Int3 &c) const noexcept { return m_data[idx(c)]; }

  // ---- Owned-cell index ranges -------------------------------------------

  /**
   * @brief Forward range of `pfc::Int3{i, j, k}` over the **owned**
   *        region `[0, nx) x [0, ny) x [0, nz)`, in k-outer / j-middle
   *        / i-inner order.
   *
   * Pair with `pfc::field::for_each(brick, fn)` for a cache-friendly
   * sweep over every owned cell, or use the iterators directly:
   *
   * ```cpp
   * for (const auto idx : brick.indices()) {
   *   brick(idx) = some_value(idx);
   * }
   * ```
   */
  detail::OwnedIndexRange indices() const noexcept {
    return {pfc::Int3{0, 0, 0}, m_size};
  }

  /**
   * @brief Forward range of `pfc::Int3{i, j, k}` over the inner region
   *        `[r, n - r)` per axis (no `r`-thick boundary slab).
   *
   * Returns an empty range whenever any axis has `n <= 2*r`, so the
   * caller never iterates past the owned region. Stencils that read
   * `±r` neighbors entirely inside the owned core (without halo data)
   * use this to bound the sweep.
   */
  detail::OwnedIndexRange indices_inner(int r) const noexcept {
    return {pfc::Int3{r, r, r},
            pfc::Int3{m_size[0] - r, m_size[1] - r, m_size[2] - r}};
  }

  /**
   * @brief Global cell index `(gi, gj, gk)` of local `(i, j, k)`.
   *
   * For halo cells (e.g. `i = -1`) this returns the conceptual global
   * index **before** any periodic wrap (so callers can implement either
   * periodic or Dirichlet semantics in their own filling code).
   */
  pfc::Int3 global(int i, int j, int k) const noexcept {
    return pfc::Int3{m_lower[0] + i, m_lower[1] + j, m_lower[2] + k};
  }

  /**
   * @brief Physical coordinates `(x, y, z)` as scalars for local `(i, j, k)`.
   *
   * Same geometry as `global_coords`; pair with structured bindings, e.g.
   * `auto [x, y, z] = brick.global_xyz(i, j, k);`.
   */
  [[nodiscard]] std::tuple<double, double, double> global_xyz(int i, int j,
                                                              int k) const noexcept {
    return {m_origin[0] + static_cast<double>(m_lower[0] + i) * m_spacing[0],
            m_origin[1] + static_cast<double>(m_lower[1] + j) * m_spacing[1],
            m_origin[2] + static_cast<double>(m_lower[2] + k) * m_spacing[2]};
  }

  /**
   * @brief Physical coordinates `(x, y, z)` of local `(i, j, k)`.
   *
   * Computed as `origin + (lower + i) * spacing` — same as
   * `LocalField::coords` but valid across the halo ring as well.
   */
  pfc::Real3 global_coords(int i, int j, int k) const noexcept {
    const auto [x, y, z] = global_xyz(i, j, k);
    return pfc::Real3{x, y, z};
  }

  // ---- Convenience iteration ---------------------------------------------
  //
  // The richer `for_each_owned/inner/border` helpers that yield raw
  // `(i, j, k)` triples live in `kernel/field/brick_iteration.hpp` so
  // this header stays focused on the data layout.

  /**
   * @brief Fill **every owned cell** by sampling `fn` at its physical
   *        coordinates. Halo cells are left untouched (zero-initialized).
   *
   * Lambda may be either of these signatures (auto-detected):
   *  - `T(double x, double y, double z)`
   *  - `T(const Real3& x)`
   */
  template <class Fn> void apply(Fn &&fn) {
    for (int k = 0; k < m_size[2]; ++k) {
      for (int j = 0; j < m_size[1]; ++j) {
        for (int i = 0; i < m_size[0]; ++i) {
          const auto x = global_coords(i, j, k);
          m_data[idx(i, j, k)] = static_cast<T>(invoke_with_coords_(fn, x));
        }
      }
    }
  }

  /**
   * @brief In-place axpy over the full padded buffer:
   *        `m_data[k] += s.alpha * s.data[k]`.
   *
   * Intended for `u += dt * du` when `du` is a `DuField` whose residual
   * buffer matches `size()` and leaves halo lanes at zero (refreshed each
   * step by `PaddedHaloExchanger`).
   *
   * @throws std::invalid_argument if `s.size != size()`.
   */
  template <class U = T, class = std::enable_if_t<std::is_same_v<U, double>>>
  PaddedBrick &operator+=(const ScaledField &s) {
    if (s.size != m_data.size()) {
      throw std::invalid_argument(
          "pfc::field::PaddedBrick::operator+=: ScaledField size " +
          std::to_string(s.size) + " does not match field size " +
          std::to_string(m_data.size()));
    }
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(m_data.size());
    const double alpha = s.alpha;
    const double *src = s.data;
    double *dst = m_data.data();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) {
      dst[i] += alpha * src[i];
    }
    return *this;
  }

private:
  int padded_extent_(int n) const noexcept { return n + 2 * m_halo; }

  template <class Fn> static auto invoke_with_coords_(Fn &&fn, const pfc::Real3 &x) {
    if constexpr (std::is_invocable_v<Fn, double, double, double>) {
      return fn(x[0], x[1], x[2]);
    } else {
      return fn(x);
    }
  }

  pfc::decomposition::Decomposition m_decomp;
  int m_rank{0};
  std::vector<T> m_data{};
  pfc::Int3 m_size{};
  pfc::Int3 m_lower{};
  pfc::Int3 m_global_size{};
  pfc::Real3 m_origin{};
  pfc::Real3 m_spacing{};
  int m_halo{0};
};

/**
 * @brief Build a `ScaledField` proxy from a scalar and a padded brick.
 *
 * Enables `u += dt * du` when `u` is `PaddedBrick<double>` and `du` is a
 * matching `DuField` (same flattened padded length).
 */
inline ScaledField operator*(double alpha, const PaddedBrick<double> &f) noexcept {
  return ScaledField{alpha, f.data(), f.size()};
}

} // namespace pfc::field
