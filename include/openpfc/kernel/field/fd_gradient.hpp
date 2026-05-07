// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_gradient.hpp
 * @brief Point-wise FD evaluator parameterized on a model-owned grads type.
 *
 * @details
 * `FdGradient<G>` is a thin point-wise façade over the per-axis central
 * first- and second-derivative primitives `apply_d1_along` /
 * `apply_d2_along` (see `fd_apply.hpp`, backed by the tabulated
 * `EvenCentralD1<Order>` / `EvenCentralD2<Order>` stencils in
 * `fd_stencils.hpp`). It is **templated on the grads aggregate `G`**
 * (whatever the model defines) and uses the per-member detection
 * concepts in `grad_concepts.hpp` to fill only the slots `G` declares.
 * Slots that this backend cannot provide (today: mixed second
 * derivatives `xy / xz / yz`, which require **corner-filled halos**)
 * trigger a compile-time error rather than silently producing zeros.
 *
 * **Backend capability** (matches the table in
 * `docs/extending_openpfc/per_point_grads.md`):
 *
 * | Member       | Status                                                          |
 * |--------------|------------------------------------------------------------------|
 * | `value`      | yes (direct read of the centre cell) | | `x, y, z`    | yes — D1
 * orders 2..14 (see `EvenCentralD1<Order>`)               | | `xx, yy, zz` | yes —
 * D2 orders 2..20 (see `EvenCentralD2<Order>`)               | | `xy, xz, yz` |
 * **rejected** at compile time (mixed seconds need corner halos;   | | | use
 * `pfc::cuda::FullPaddedDeviceHalo` + a future GPU evaluator,  | |              | or
 * `SpectralGradient<G>`).                                       |
 *
 * Construction binds to a contiguous local `core` array of shape
 * `nx*ny*nz` (x varies fastest) and the per-axis grid spacing. `prepare()`
 * is a no-op for FD: the application is responsible for any halo exchange
 * before iterating. Face halos are unused in the interior `[hw, n-hw)`
 * because, with `halo_width >= order/2`, all stencil neighbours stay in
 * the local core.
 *
 * **Error handling**: if the model's `G` declares a derivative member
 * (e.g. `g.x`) whose stencil order is not tabulated (e.g. order 16 for
 * D1 today), the constructor throws `std::invalid_argument` with the
 * offending order in the message — strictly better behaviour than
 * silently producing zeros.
 *
 * @note `operator()` is `const`/`noexcept` and intentionally inlines the
 *       stencil arithmetic so the surrounding `for_each_interior` loop
 *       fuses into a single tight kernel.
 *
 * @see grad_concepts.hpp for the per-member detection concepts
 * @see grad_point.hpp for the convenience default catalog struct
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the driver loop
 * @see fd_apply.hpp for the per-axis apply primitives
 * @see fd_stencils.hpp for the underlying stencil tables
 * @see runtime/cuda/full_padded_device_halo.hpp for the corner-filled halo
 *      policy required to unblock `xy/xz/yz` on a future GPU evaluator
 */

#include <cstddef>
#include <stdexcept>
#include <string>

#include <openpfc/kernel/field/fd_apply.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <openpfc/kernel/field/grad_concepts.hpp>
#include <openpfc/kernel/field/local_field.hpp>

namespace pfc::field {

/**
 * @brief Even-order central FD point evaluator (first + second derivatives).
 *
 * @tparam G Model-owned per-point grads aggregate. The constructor and
 *           `operator()` consult `pfc::field::has_*<G>` to decide which
 *           members of `G` to populate. A `G` that asks for mixed second
 *           derivatives (`xy / xz / yz`) is rejected at compile time
 *           (see the `static_assert` below).
 */
template <class G> class FdGradient {
public:
  /**
   * @brief Construct an FD point evaluator over a contiguous brick.
   *
   * @param core         Pointer to the local `nx*ny*nz` row-major buffer
   *                     (x fastest); must outlive the evaluator.
   * @param nx,ny,nz     Local extents in cells.
   * @param dx,dy,dz     Per-axis grid spacing.
   * @param halo_width   Local halo width (`order / 2` is required for the
   *                     central FD contract).
   * @param order        Even spatial order. Must be tabulated for every
   *                     derivative `G` declares: D2 orders 2..20 are
   *                     supported; D1 orders 2..14 are supported.
   *
   * @throws std::invalid_argument if `order` is outside the tabulated
   *         range for any derivative member declared by `G`.
   */
  FdGradient(const double *core, int nx, int ny, int nz, double dx, double dy,
             double dz, int halo_width, int order)
      : m_core(core), m_nx(nx), m_ny(ny), m_nz(nz),
        m_sxy(static_cast<std::ptrdiff_t>(nx) * static_cast<std::ptrdiff_t>(ny)),
        m_hw(halo_width) {
    static_assert(!has_xy<G> && !has_xz<G> && !has_yz<G>,
                  "FdGradient: mixed second derivatives (xy/xz/yz) need "
                  "corner-filled halos. The CPU exchanger is axis-aligned "
                  "today; use `pfc::cuda::FullPaddedDeviceHalo` plus the "
                  "GPU evaluator (when available), or `SpectralGradient<G>`, "
                  "for these members.");

    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double inv_dz = 1.0 / dz;
    const double inv_dx2 = inv_dx * inv_dx;
    const double inv_dy2 = inv_dy * inv_dy;
    const double inv_dz2 = inv_dz * inv_dz;

    if constexpr (has_xx<G> || has_yy<G> || has_zz<G>) {
      pfc::field::fd::EvenCentralD2View st2{};
      if (!pfc::field::fd::lookup_even_central_d2(order, &st2)) {
        throw std::invalid_argument(
            "FdGradient: order " + std::to_string(order) +
            " has no central second-derivative stencil table "
            "(supported even orders: 2..20).");
      }
      m_d2_stencil = st2;
      const double inv_den2 = 1.0 / static_cast<double>(st2.denom);
      m_sx2 = inv_dx2 * inv_den2;
      m_sy2 = inv_dy2 * inv_den2;
      m_sz2 = inv_dz2 * inv_den2;
    }

    if constexpr (has_x<G> || has_y<G> || has_z<G>) {
      pfc::field::fd::EvenCentralD1View st1{};
      if (!pfc::field::fd::lookup_even_central_d1(order, &st1)) {
        throw std::invalid_argument("FdGradient: order " + std::to_string(order) +
                                    " has no central first-derivative stencil table "
                                    "(supported even orders: 2..14).");
      }
      m_d1_stencil = st1;
      const double inv_den1 = 1.0 / static_cast<double>(st1.denom);
      m_sx1 = inv_dx * inv_den1;
      m_sy1 = inv_dy * inv_den1;
      m_sz1 = inv_dz * inv_den1;
    }
  }

  void prepare() noexcept {}

  int imin() const noexcept { return m_hw; }
  int imax() const noexcept { return m_nx - m_hw; }
  int jmin() const noexcept { return m_hw; }
  int jmax() const noexcept { return m_ny - m_hw; }
  int kmin() const noexcept { return m_hw; }
  int kmax() const noexcept { return m_nz - m_hw; }

  [[nodiscard]] std::size_t idx(int ix, int iy, int iz) const noexcept {
    return static_cast<std::size_t>(ix) +
           static_cast<std::size_t>(iy) * static_cast<std::size_t>(m_nx) +
           static_cast<std::size_t>(iz) * static_cast<std::size_t>(m_sxy);
  }

  [[nodiscard]] G operator()(int ix, int iy, int iz) const noexcept {
    G g{};
    const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(idx(ix, iy, iz));

    if constexpr (has_value<G>) {
      g.value = m_core[c];
    }

    if constexpr (has_x<G>) {
      g.x = m_sx1 * pfc::field::fd::apply_d1_along<0>(
                        m_d1_stencil, m_core, c,
                        /*sx=*/static_cast<std::ptrdiff_t>(1),
                        /*sy=*/static_cast<std::ptrdiff_t>(m_nx), m_sxy);
    }
    if constexpr (has_y<G>) {
      g.y = m_sy1 * pfc::field::fd::apply_d1_along<1>(
                        m_d1_stencil, m_core, c,
                        /*sx=*/static_cast<std::ptrdiff_t>(1),
                        /*sy=*/static_cast<std::ptrdiff_t>(m_nx), m_sxy);
    }
    if constexpr (has_z<G>) {
      g.z = m_sz1 * pfc::field::fd::apply_d1_along<2>(
                        m_d1_stencil, m_core, c,
                        /*sx=*/static_cast<std::ptrdiff_t>(1),
                        /*sy=*/static_cast<std::ptrdiff_t>(m_nx), m_sxy);
    }

    if constexpr (has_xx<G>) {
      g.xx = m_sx2 * pfc::field::fd::apply_d2_along<0>(
                         m_d2_stencil, m_core, c,
                         /*sx=*/static_cast<std::ptrdiff_t>(1),
                         /*sy=*/static_cast<std::ptrdiff_t>(m_nx), m_sxy);
    }
    if constexpr (has_yy<G>) {
      g.yy = m_sy2 * pfc::field::fd::apply_d2_along<1>(
                         m_d2_stencil, m_core, c,
                         /*sx=*/static_cast<std::ptrdiff_t>(1),
                         /*sy=*/static_cast<std::ptrdiff_t>(m_nx), m_sxy);
    }
    if constexpr (has_zz<G>) {
      g.zz = m_sz2 * pfc::field::fd::apply_d2_along<2>(
                         m_d2_stencil, m_core, c,
                         /*sx=*/static_cast<std::ptrdiff_t>(1),
                         /*sy=*/static_cast<std::ptrdiff_t>(m_nx), m_sxy);
    }

    return g;
  }

private:
  const double *m_core{nullptr};
  int m_nx{0};
  int m_ny{0};
  int m_nz{0};
  std::ptrdiff_t m_sxy{0};
  int m_hw{0};

  // D2 (xx, yy, zz) per-axis combined scale `1 / (h_i^2 * denom_2)`.
  double m_sx2{0.0};
  double m_sy2{0.0};
  double m_sz2{0.0};
  pfc::field::fd::EvenCentralD2View m_d2_stencil{};

  // D1 (x, y, z) per-axis combined scale `1 / (h_i * denom_1)`.
  double m_sx1{0.0};
  double m_sy1{0.0};
  double m_sz1{0.0};
  pfc::field::fd::EvenCentralD1View m_d1_stencil{};
};

/**
 * @brief Free-function factory: build an `FdGradient<G>` from a `LocalField`.
 *
 * Mirrors the `world::create`, `decomposition::create`, `fft::create` family:
 * derives `nx, ny, nz`, the per-axis grid spacings, and the halo width
 * directly from `u`. The caller supplies the spatial accuracy `order` and
 * the model-owned grads type as an explicit template argument.
 *
 * Example:
 * @code
 * struct HeatGrads { double xx{}, yy{}, zz{}; };
 * auto grad = pfc::field::create<HeatGrads>(stack.u(), 4);
 * @endcode
 *
 * @tparam G     Model-owned grads aggregate (see `grad_concepts.hpp`).
 * @param u      Local field bound to the FD subdomain (must outlive the
 *               returned evaluator; the evaluator reads `u.data()`).
 * @param order  Even spatial order of the central stencil. Must be
 *               tabulated for every derivative `G` declares: D2 orders
 *               2..20 are supported; D1 orders 2..14 are supported. For
 *               the standard "halo width = stencil half-width" contract,
 *               `order/2 == u.halo_width()` should hold on the field.
 *
 * @return An `FdGradient<G>` ready to be passed to
 *         `pfc::sim::for_each_interior` (or `pfc::sim::steppers::create`).
 *
 * @throws std::invalid_argument if `order` is outside the tabulated range
 *         for any derivative member declared by `G`.
 */
template <class G>
[[nodiscard]] inline FdGradient<G> create(const LocalField<double> &u, int order) {
  const auto sz = u.size3();
  const auto sp = u.spacing();
  return FdGradient<G>(u.data(), sz[0], sz[1], sz[2], sp[0], sp[1], sp[2],
                       u.halo_width(), order);
}

} // namespace pfc::field
