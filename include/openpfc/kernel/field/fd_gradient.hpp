// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_gradient.hpp
 * @brief Point-wise FD evaluator parameterized on a model-owned grads type.
 *
 * @details
 * `pfc::gradient::FDGradient<G>` is a thin point-wise façade over the
 * per-axis central first- and second-derivative primitives
 * `apply_d1_along` / `apply_d2_along` (see `fd_apply.hpp`, backed by
 * the tabulated `EvenCentralD1<Order>` / `EvenCentralD2<Order>` stencils
 * in `fd_stencils.hpp`). It is **templated on the grads aggregate `G`**
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
 * |--------------|-----------------------------------------------------------------|
 * | `value`      | yes (direct read of the centre cell)                            |
 * | `x, y, z`    | yes — D1 orders 2..14 (see `EvenCentralD1<Order>`)              |
 * | `xx, yy, zz` | yes — D2 orders 2..20 (see `EvenCentralD2<Order>`)              |
 * | `xy, xz, yz` | **rejected** at compile time (mixed seconds need corner halos)  |
 *
 * Construction is normally done through the brick-binding constructor
 *
 *     pfc::gradient::FDGradient<HeatGrads> grad(u);            // order = 2
 *     pfc::gradient::FDGradient<HeatGrads> grad(u, 4);         // explicit order
 *
 * which reads the geometry (`nx, ny, nz`, spacings, halo width) directly
 * from the `pfc::field::PaddedBrick<double>`. Legacy callers that hand a
 * raw pointer + extents are still supported, and the
 * `pfc::field::create<G>(LocalField, order)` factory keeps working
 * (for the unpadded `LocalField` path used by `FdCpuStack`).
 *
 * Drive a sweep with the free `pfc::gradient::evaluate(grad, idx)`
 * helper — it accepts a `pfc::Int3` so the iteration code does not
 * have to know whether the underlying brick is 1D, 2D, or 3D:
 *
 *     pfc::field::for_each(du, [&](const auto& idx) {
 *       const auto g = pfc::gradient::evaluate(grad, idx);
 *       du(idx) = heat3d::kD * (g.xx + g.yy + g.zz);
 *     });
 *
 * `prepare()` is a no-op for FD: the application is responsible for any
 * halo exchange before iterating. Face halos are unused in the interior
 * `[hw, n-hw)` because, with `halo_width >= order/2`, all stencil
 * neighbours stay in the local core.
 *
 * **Error handling**: if the model's `G` declares a derivative member
 * (e.g. `g.x`) whose stencil order is not tabulated (e.g. order 16 for
 * D1 today), the constructor throws `std::invalid_argument` with the
 * offending order in the message — strictly better behaviour than
 * silently producing zeros.
 *
 * @note `operator()` is `const`/`noexcept` and intentionally inlines the
 *       stencil arithmetic so the surrounding `for_each_interior` /
 *       `pfc::field::for_each` loop fuses into a single tight kernel.
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
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <openpfc/kernel/data/world_types.hpp>
#include <openpfc/kernel/field/fd_apply.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <openpfc/kernel/field/grad_concepts.hpp>
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

namespace pfc::gradient {

/**
 * @brief Even-order central FD point evaluator (first + second derivatives).
 *
 * The optional halo_prepare_callback enables multi-stage Runge-Kutta steppers
 * to coordinate halo exchange with stage evaluation through the prepare() hook.
 * When no callback is provided, prepare() is a no-op (backward compatible).
 *
 * @tparam G Model-owned per-point grads aggregate. The constructor and
 *           `operator()` consult `pfc::field::has_*<G>` to decide which
 *           members of `G` to populate. A `G` that asks for mixed second
 *           derivatives (`xy / xz / yz`) is rejected at compile time
 *           (see the `static_assert` below).
 */
template <class G> class FDGradient {
public:
  /**
   * @brief Construct an FD point evaluator over a contiguous brick.
   *
   * Prefer the `(PaddedBrick&, order)` overload below for the typical
   * case; this raw-pointer constructor is for power users that build the
   * evaluator from a `LocalField` or a buffer the kernel does not own.
   *
   * Treats `core` as a tightly-packed `nx*ny*nz` row-major buffer
   * (x fastest), with iteration bounds `[halo_width, n-halo_width)` per
   * axis — the `LocalField` convention.
   *
   * @param core                  Pointer to the local `nx*ny*nz` row-major buffer
   *                             (x fastest); must outlive the evaluator.
   * @param nx,ny,nz              Local extents in cells.
   * @param dx,dy,dz              Per-axis grid spacing.
   * @param halo_width            Local halo width (`order / 2` is required for the
   *                             central FD contract).
   * @param order                 Even spatial order. Must be tabulated for every
   *                             derivative `G` declares: D2 orders 2..20 are
   *                             supported; D1 orders 2..14 are supported.
   *                             Defaults to 2 (the cheapest 7-point Laplacian).
   * @param halo_prepare_callback Optional function invoked by prepare() to
   *                             trigger halo exchange before each stage in
   *                             multi-stage methods.
   *
   * @throws std::invalid_argument if `order` is outside the tabulated
   *         range for any derivative member declared by `G`.
   */
  FDGradient(const double *core, int nx, int ny, int nz, double dx, double dy,
             double dz, int halo_width, int order = 2,
             std::function<void()> halo_prepare_callback = {})
      : FDGradient(core, nx, ny, nz, /*sy=*/static_cast<std::ptrdiff_t>(nx),
                   /*sxy=*/static_cast<std::ptrdiff_t>(nx) *
                       static_cast<std::ptrdiff_t>(ny),
                   dx, dy, dz, /*imin=*/halo_width, /*imax=*/nx - halo_width,
                   /*jmin=*/halo_width, /*jmax=*/ny - halo_width,
                   /*kmin=*/halo_width, /*kmax=*/nz - halo_width, halo_width,
                   order, std::move(halo_prepare_callback)) {}

  /**
   * @brief Bind the evaluator to a `pfc::field::PaddedBrick<double>`.
   *
   * Pre-offsets the stored core pointer to the brick's **owned** `(0, 0, 0)`
   * cell so that `grad(i, j, k)` (and `pfc::gradient::evaluate(grad, idx)`)
   * indexes by **owned coordinates** `i, j, k ∈ [0, nx_owned)`. Every owned
   * cell is stencil-safe because the halo ring is filled by the matching
   * `pfc::communication::PaddedHaloExchanger<T>` before the sweep starts. The
   * strides used by the stencil reads are still the **padded** strides
   * `(1, nx_pad, nx_pad·ny_pad)` so reads at the boundary reach into the
   * halo correctly.
   *
   * @param u                     Padded brick to evaluate over (must outlive `*this`).
   * @param order                 Even spatial order (defaults to 2).
   * @param halo_prepare_callback Optional function invoked by prepare() to
   *                             trigger halo exchange before each stage in
   *                             multi-stage methods.
   */
  explicit FDGradient(const pfc::field::PaddedBrick<double> &u, int order = 2,
                    std::function<void()> halo_prepare_callback = {})
      : FDGradient(brick_owned_origin_(u), u.nx(), u.ny(), u.nz(),
                   /*sy=*/static_cast<std::ptrdiff_t>(u.padded_size3()[0]),
                   /*sxy=*/static_cast<std::ptrdiff_t>(u.padded_size3()[0]) *
                       static_cast<std::ptrdiff_t>(u.padded_size3()[1]),
                   u.spacing()[0], u.spacing()[1], u.spacing()[2],
                   /*imin=*/0, /*imax=*/u.nx(),
                   /*jmin=*/0, /*jmax=*/u.ny(),
                   /*kmin=*/0, /*kmax=*/u.nz(), u.halo_width(), order,
                   std::move(halo_prepare_callback)) {}

  /**
   * @brief Prepare for gradient evaluation.
   *
   * If a halo_prepare_callback was provided to the constructor, invokes it to
   * trigger halo exchange before each stage in multi-stage methods. Otherwise,
   * does nothing (backward compatible with existing FD code).
   *
   * @note This method is not `noexcept` because the user-provided callback may
   *       throw, matching the behavior of `SpectralGradient::prepare()` which
   *       can throw during FFT operations.
   */
  void prepare() {
    if (m_halo_prepare_callback) {
      m_halo_prepare_callback();
    }
  }

  int imin() const noexcept { return m_imin; }
  int imax() const noexcept { return m_imax; }
  int jmin() const noexcept { return m_jmin; }
  int jmax() const noexcept { return m_jmax; }
  int kmin() const noexcept { return m_kmin; }
  int kmax() const noexcept { return m_kmax; }

  [[nodiscard]] std::size_t idx(int ix, int iy, int iz) const noexcept {
    return static_cast<std::size_t>(ix) +
           static_cast<std::size_t>(iy) * static_cast<std::size_t>(m_sy) +
           static_cast<std::size_t>(iz) * static_cast<std::size_t>(m_sxy);
  }

  [[nodiscard]] G operator()(int ix, int iy, int iz) const noexcept {
    G g{};
    const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(idx(ix, iy, iz));

    if constexpr (pfc::field::has_value<G>) {
      g.value = m_core[c];
    }

    if constexpr (pfc::field::has_x<G>) {
      g.x = m_sx1 *
            pfc::field::fd::apply_d1_along<0>(m_d1_stencil, m_core, c,
                                              /*sx=*/static_cast<std::ptrdiff_t>(1),
                                              /*sy=*/m_sy, m_sxy);
    }
    if constexpr (pfc::field::has_y<G>) {
      g.y = m_sy1 *
            pfc::field::fd::apply_d1_along<1>(m_d1_stencil, m_core, c,
                                              /*sx=*/static_cast<std::ptrdiff_t>(1),
                                              /*sy=*/m_sy, m_sxy);
    }
    if constexpr (pfc::field::has_z<G>) {
      g.z = m_sz1 *
            pfc::field::fd::apply_d1_along<2>(m_d1_stencil, m_core, c,
                                              /*sx=*/static_cast<std::ptrdiff_t>(1),
                                              /*sy=*/m_sy, m_sxy);
    }

    if constexpr (pfc::field::has_xx<G>) {
      g.xx = m_sx2 *
             pfc::field::fd::apply_d2_along<0>(m_d2_stencil, m_core, c,
                                               /*sx=*/static_cast<std::ptrdiff_t>(1),
                                               /*sy=*/m_sy, m_sxy);
    }
    if constexpr (pfc::field::has_yy<G>) {
      g.yy = m_sy2 *
             pfc::field::fd::apply_d2_along<1>(m_d2_stencil, m_core, c,
                                               /*sx=*/static_cast<std::ptrdiff_t>(1),
                                               /*sy=*/m_sy, m_sxy);
    }
    if constexpr (pfc::field::has_zz<G>) {
      g.zz = m_sz2 *
             pfc::field::fd::apply_d2_along<2>(m_d2_stencil, m_core, c,
                                               /*sx=*/static_cast<std::ptrdiff_t>(1),
                                               /*sy=*/m_sy, m_sxy);
    }

    return g;
  }

  /// Convenience overload: evaluate at a `pfc::Int3` index directly.
  [[nodiscard]] G operator()(const pfc::Int3 &c) const noexcept {
    return (*this)(c[0], c[1], c[2]);
  }

private:
  /// Pointer to the brick's **owned** `(0, 0, 0)` cell inside its padded
  /// buffer. Used by the brick-binding constructor so the public
  /// `operator()` indexes by owned coordinates while the stencil reads
  /// still hop along the padded strides.
  static const double *
  brick_owned_origin_(const pfc::field::PaddedBrick<double> &u) noexcept {
    const std::size_t hw = static_cast<std::size_t>(u.halo_width());
    const std::size_t nxp = static_cast<std::size_t>(u.padded_size3()[0]);
    const std::size_t nyp = static_cast<std::size_t>(u.padded_size3()[1]);
    return u.data() + hw + hw * nxp + hw * nxp * nyp;
  }

  /// Common private constructor — every public ctor delegates here.
  FDGradient(const double *core, int nx, int ny, int nz, std::ptrdiff_t sy,
             std::ptrdiff_t sxy, double dx, double dy, double dz, int imin, int imax,
             int jmin, int jmax, int kmin, int kmax, int halo_width, int order,
             std::function<void()> halo_prepare_callback)
      : m_core(core), m_nx(nx), m_ny(ny), m_nz(nz), m_sy(sy), m_sxy(sxy),
        m_imin(imin), m_imax(imax), m_jmin(jmin), m_jmax(jmax), m_kmin(kmin),
        m_kmax(kmax), m_hw(halo_width),
        m_halo_prepare_callback(std::move(halo_prepare_callback)) {
    static_assert(!pfc::field::has_xy<G> && !pfc::field::has_xz<G> &&
                      !pfc::field::has_yz<G>,
                  "FDGradient: mixed second derivatives (xy/xz/yz) need "
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

    if constexpr (pfc::field::has_xx<G> || pfc::field::has_yy<G> ||
                  pfc::field::has_zz<G>) {
      pfc::field::fd::EvenCentralD2View st2{};
      if (!pfc::field::fd::lookup_even_central_d2(order, &st2)) {
        throw std::invalid_argument(
            "FDGradient: order " + std::to_string(order) +
            " has no central second-derivative stencil table "
            "(supported even orders: 2..20).");
      }
      m_d2_stencil = st2;
      const double inv_den2 = 1.0 / static_cast<double>(st2.denom);
      m_sx2 = inv_dx2 * inv_den2;
      m_sy2 = inv_dy2 * inv_den2;
      m_sz2 = inv_dz2 * inv_den2;
    }

    if constexpr (pfc::field::has_x<G> || pfc::field::has_y<G> ||
                  pfc::field::has_z<G>) {
      pfc::field::fd::EvenCentralD1View st1{};
      if (!pfc::field::fd::lookup_even_central_d1(order, &st1)) {
        throw std::invalid_argument("FDGradient: order " + std::to_string(order) +
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

  const double *m_core{nullptr};
  int m_nx{0};
  int m_ny{0};
  int m_nz{0};
  std::ptrdiff_t m_sy{0};
  std::ptrdiff_t m_sxy{0};
  int m_imin{0}, m_imax{0};
  int m_jmin{0}, m_jmax{0};
  int m_kmin{0}, m_kmax{0};
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
  std::function<void()> m_halo_prepare_callback{};
};

/**
 * @brief Evaluate `grad` at index `idx` (free function form).
 *
 * Mirrors `pfc::communication::exchange(halo)` (or `start_exchange` /
 * `finish_exchange`) so
 * call sites read declaratively: gradient evaluation, halo control, and
 * iteration are three separate concerns.
 *
 * Works with any evaluator that exposes `operator()(int, int, int) -> G`,
 * i.e. `FDGradient<G>` and the upcoming `SpectralGradient<G>` after the
 * relocation. The generic form keeps user code agnostic of which backend
 * it is wired up to.
 */
template <class Eval>
[[nodiscard]] inline auto
evaluate(const Eval &grad,
         const pfc::Int3 &idx) noexcept(noexcept(grad(idx[0], idx[1], idx[2])))
    -> decltype(grad(idx[0], idx[1], idx[2])) {
  return grad(idx[0], idx[1], idx[2]);
}

/**
 * @brief Run any evaluator-internal preparation step (no-op for FD).
 *
 * Spectral evaluators use this to launch their FFTs once per timestep
 * before the per-cell loop reads `grad(idx)`. For `FDGradient` this is
 * a noexcept no-op; the call site stays uniform either way.
 */
template <class Eval>
inline auto prepare(Eval &grad) noexcept(noexcept(grad.prepare()))
    -> decltype(grad.prepare()) {
  return grad.prepare();
}

} // namespace pfc::gradient

namespace pfc::field {

/**
 * @brief Deprecated alias for the canonical `pfc::gradient::FDGradient<G>`.
 *
 * Kept so existing call sites — `FdCpuStack`, the per-point heat3d
 * tutorial, the device evaluator wrappers, and friends — continue to
 * compile while users migrate to the new name. New code should use
 * `pfc::gradient::FDGradient<G>` directly.
 */
template <class G> using FdGradient = pfc::gradient::FDGradient<G>;

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
 *               Defaults to 2.
 *
 * @return An `FdGradient<G>` ready to be passed to
 *         `pfc::sim::for_each_interior` (or `pfc::sim::steppers::create`).
 *
 * @throws std::invalid_argument if `order` is outside the tabulated range
 *         for any derivative member declared by `G`.
 */
template <class G>
[[nodiscard]] inline FdGradient<G> create(const LocalField<double> &u,
                                          int order = 2) {
  const auto sz = u.size3();
  const auto sp = u.spacing();
  return FdGradient<G>(u.data(), sz[0], sz[1], sz[2], sp[0], sp[1], sp[2],
                       u.halo_width(), order);
}

/**
 * @brief Build an `FdGradient<G>` over a halo-padded brick.
 *
 * The evaluator indexes the **full** `(nx+2hw)×(ny+2hw)×(nz+2hw)` storage with
 * interior bounds `[hw, nx_pad-hw)` per axis, matching ghost cells populated by
 * `pfc::communication::PaddedHaloExchanger<T>` on `u.data()`.
 *
 * Equivalent to `pfc::gradient::FDGradient<G>(u, order)`; kept for symmetry
 * with the `LocalField` factory above.
 *
 * @tparam G     Model-owned grads aggregate (see `grad_concepts.hpp`).
 * @param u      Padded brick (must outlive the returned evaluator).
 * @param order  Even spatial order; should satisfy `order / 2 == u.halo_width()`.
 *               Defaults to 2.
 */
template <class G>
[[nodiscard]] inline FdGradient<G> create(const PaddedBrick<double> &u,
                                          int order = 2) {
  return FdGradient<G>(u, order);
}

} // namespace pfc::field
