// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_gradient.hpp
 * @brief Point-wise FD evaluator parameterized on a model-owned grads type.
 *
 * @details
 * `FdGradient<G>` is a thin point-wise façade over the even-order central
 * second-derivative stencil tables in
 * `pfc::field::fd::detail::EvenFdStencil1d` (see `finite_difference.hpp`). It
 * is **templated on the grads aggregate `G`** (whatever the model defines)
 * and uses the per-member detection concepts in `grad_concepts.hpp` to fill
 * only the slots `G` declares. Slots that this backend cannot provide (first
 * derivatives, mixed second derivatives) trigger a compile-time error rather
 * than silently producing zeros.
 *
 * Construction binds to a contiguous local `core` array of shape
 * `nx*ny*nz` (x varies fastest) and the per-axis grid metric. `prepare()`
 * is a no-op for FD: the application is responsible for any halo exchange
 * before iterating. Face halos are unused in the interior `[hw, n-hw)`
 * because, with `halo_width >= order/2`, all stencil neighbours stay in
 * the local core.
 *
 * @note `operator()` is `const`/`noexcept` and intentionally inlines the
 *       stencil arithmetic so the surrounding `for_each_interior` loop
 *       fuses into a single tight kernel.
 *
 * @see grad_concepts.hpp for the per-member detection concepts
 * @see grad_point.hpp for the convenience default catalog struct
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the driver loop
 * @see finite_difference.hpp for the stencil tables
 */

#include <cstddef>
#include <cstdint>

#include <openpfc/kernel/field/finite_difference.hpp>
#include <openpfc/kernel/field/grad_concepts.hpp>
#include <openpfc/kernel/field/local_field.hpp>

namespace pfc::field {

/**
 * @brief Even-order central FD second-derivative point evaluator.
 *
 * @tparam G Model-owned per-point grads aggregate. The constructor and
 *           `operator()` consult `pfc::field::has_*<G>` to decide which
 *           members of `G` to populate. A `G` that asks for first
 *           derivatives or mixed second derivatives is rejected at
 *           compile time (see the `static_assert`s below).
 */
template <class G> class FdGradient {
public:
  FdGradient(const double *core, int nx, int ny, int nz, double inv_dx2,
             double inv_dy2, double inv_dz2, int halo_width, int order)
      : m_core(core), m_nx(nx), m_ny(ny), m_nz(nz),
        m_sxy(static_cast<std::ptrdiff_t>(nx) * static_cast<std::ptrdiff_t>(ny)),
        m_hw(halo_width), m_inv_dx2(inv_dx2), m_inv_dy2(inv_dy2),
        m_inv_dz2(inv_dz2) {
    pfc::field::fd::detail::EvenFdStencil1d st{};
    if (pfc::field::fd::detail::fd_even_order_lookup(order, &st)) {
      m_stencil = st;
      const double inv_den = 1.0 / static_cast<double>(st.denom);
      m_sx = inv_dx2 * inv_den;
      m_sy = inv_dy2 * inv_den;
      m_sz = inv_dz2 * inv_den;
    }
  }

  void prepare() noexcept {}

  int imin() const noexcept { return m_hw; }
  int imax() const noexcept { return m_nx - m_hw; }
  int jmin() const noexcept { return m_hw; }
  int jmax() const noexcept { return m_ny - m_hw; }
  int kmin() const noexcept { return m_hw; }
  int kmax() const noexcept { return m_nz - m_hw; }

  inline std::size_t idx(int ix, int iy, int iz) const noexcept {
    return static_cast<std::size_t>(ix) +
           static_cast<std::size_t>(iy) * static_cast<std::size_t>(m_nx) +
           static_cast<std::size_t>(iz) * static_cast<std::size_t>(m_sxy);
  }

  inline G operator()(int ix, int iy, int iz) const noexcept {
    static_assert(!has_x<G> && !has_y<G> && !has_z<G>,
                  "FdGradient: first derivatives are not implemented yet "
                  "(would need an extended halo and 1st-order stencil tables).");
    static_assert(!has_xy<G> && !has_xz<G> && !has_yz<G>,
                  "FdGradient: mixed second derivatives need corner halos "
                  "and are not yet supported. Use SpectralGradient for now.");

    G g{};
    const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(idx(ix, iy, iz));
    const double uc = m_core[c];

    if constexpr (has_value<G>) {
      g.value = uc;
    }

    if constexpr (has_xx<G> || has_yy<G> || has_zz<G>) {
      const int M = m_stencil.half_width;
      const std::int64_t *coeff = m_stencil.coeffs;

      double dxx{}, dyy{}, dzz{};
      if constexpr (has_xx<G>) dxx = static_cast<double>(coeff[0]) * uc;
      if constexpr (has_yy<G>) dyy = static_cast<double>(coeff[0]) * uc;
      if constexpr (has_zz<G>) dzz = static_cast<double>(coeff[0]) * uc;
      for (int k = 1; k <= M; ++k) {
        const double ck = static_cast<double>(coeff[k]);
        const std::ptrdiff_t kx = static_cast<std::ptrdiff_t>(k);
        const std::ptrdiff_t ky = kx * static_cast<std::ptrdiff_t>(m_nx);
        const std::ptrdiff_t kz = kx * m_sxy;
        if constexpr (has_xx<G>) dxx += ck * (m_core[c - kx] + m_core[c + kx]);
        if constexpr (has_yy<G>) dyy += ck * (m_core[c - ky] + m_core[c + ky]);
        if constexpr (has_zz<G>) dzz += ck * (m_core[c - kz] + m_core[c + kz]);
      }
      if constexpr (has_xx<G>) g.xx = m_sx * dxx;
      if constexpr (has_yy<G>) g.yy = m_sy * dyy;
      if constexpr (has_zz<G>) g.zz = m_sz * dzz;
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
  double m_inv_dx2{0.0};
  double m_inv_dy2{0.0};
  double m_inv_dz2{0.0};
  double m_sx{0.0};
  double m_sy{0.0};
  double m_sz{0.0};
  pfc::field::fd::detail::EvenFdStencil1d m_stencil{};
};

/**
 * @brief Free-function factory: build an `FdGradient<G>` from a `LocalField`.
 *
 * Mirrors the `world::create`, `decomposition::create`, `fft::create` family:
 * derives `nx, ny, nz`, the inverse-square grid spacings, and the halo width
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
 * @param order  Even spatial order of the central second-derivative stencil
 *               (2, 4, ..., 20). Must satisfy `order/2 == u.halo_width()`
 *               for the standard "halo width = stencil half-width" contract.
 *
 * @return An `FdGradient<G>` ready to be passed to
 *         `pfc::sim::for_each_interior` (or `pfc::sim::steppers::create`).
 */
template <class G>
inline FdGradient<G> create(const LocalField<double> &u, int order) {
  const auto sz = u.size3();
  const auto sp = u.spacing();
  const double inv_dx2 = 1.0 / (sp[0] * sp[0]);
  const double inv_dy2 = 1.0 / (sp[1] * sp[1]);
  const double inv_dz2 = 1.0 / (sp[2] * sp[2]);
  return FdGradient<G>(u.data(), sz[0], sz[1], sz[2], inv_dx2, inv_dy2, inv_dz2,
                       u.halo_width(), order);
}

} // namespace pfc::field
